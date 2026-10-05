"""Draw the 20-report HCY deep-dive sample, and pre-register the draw.

The sample is **purposive, not representative**: seven cells, each answering a different question
about where a method's error comes from. No aggregate metric may be quoted from it, 20 reports is
~200 annotated pairs, which cannot separate methods whose HCY micro-F1 differs by 0.009. What it
produces is mechanisms with a report id attached.

    cell            n   pool
    ─────────────────────────────────────────────────────────────────────────
    pb_best         5   the 10 reports PhenoBERT scores highest on (annotated)
    pb_worst        5   the 10 it scores lowest on (annotated)
    no_annotation   2   reports no source annotated, pairwise non-duplicate
    family          2   reports carrying a ``family`` annotation
    lab_value       2   reports carrying a ``lab_value`` annotation
    implicit        2   reports carrying an ``implicit`` annotation
    negated         2   reports carrying a ``negated`` annotation

Two properties make this a frame rather than a shortlist:

1. **It is written before anything is read.** The seed, the input digests, the full ranked pools
   and the draw all land in one directory. The deep dive then happens against that file. A report
   that turns out to be interesting cannot be added afterwards without the diff showing it.
2. **The draw is random within each pool, not extremal.** Taking the top 5 of the 10 would select
   on the very quantity the deep dive is meant to explain. A seeded draw from the pool keeps the
   cell definition ("PhenoBERT does well here") without also selecting for the extremity.

The 20 are **distinct**: a report satisfying two cells is spent on one of them. Cells are filled
most-constrained-first (least slack between pool size and remaining need), recomputed after every
pick, so a scarce cell is never starved by a plentiful one drawing its last candidate.

Patient-data boundary
---------------------
This reads HCY. Report text is used for **one** purpose, near-duplicate detection among the
unannotated reports, and never leaves the process: the outputs carry ids, counts, scores and
text *digests* only. :func:`assert_no_text` re-checks that on the way out. Run it on the cluster. What you copy off is the frame.

Usage
-----
    python apps/compare_ui/select_hcy_documents.py \
        --predictions <output>/baseline_phenobert/hcy/phenobert_predictions.jsonl \
        --dataset     <hcy>/curated_ground_truth_2026-09-08 \
        --hcy-dir     <hcy> \
        --out         <output>/hcy_deepdive_frame

    python apps/compare_ui/select_hcy_documents.py --selftest    # no data needed
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from datetime import date
from pathlib import Path

from hpo_extraction.paths import REPO_ROOT as _REPO  # noqa: E402
for _p in (str(_REPO),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: The dataset files ``hcy_ground_truth``'s ``dataset`` stage writes into ``<hcy>/curated_gold_<date>/``.
ANNOTATIONS_FILE = "hcy_curated_annotations.csv"
REPORTS_FILE = "hcy_curated_reports.csv"
GOLD_FILE = "hcy_ground_truth_curated.csv"

#: The four annotation qualifiers that get a cell of their own, and the size of each cell.
LABEL_CELLS: tuple[str, ...] = ("family", "lab_value", "implicit", "negated")

#: ``{cell: n}``, the shape of the sample. Changing this changes the pre-registration, so it is a
#: constant, not a flag.
CELL_SIZES: dict[str, int] = {
    "pb_best": 5, "pb_worst": 5, "no_annotation": 2,
    **{name: 2 for name in LABEL_CELLS},
}

#: Reports below this many curated annotated terms are excluded from the PhenoBERT ranking. Per-report
#: F1 over a 1-term ground-truth set is 0 or 1, so without a floor both ends of the ranking fill with
#: trivial reports and the cell stops meaning "PhenoBERT does well/badly here".
DEFAULT_MIN_GOLD = 3

#: Token-set Jaccard above which two reports count as the same report for the ``no_annotation``
#: cell. HCY carries follow-up letters that differ in a date and a sentence.
DEFAULT_DUP_JACCARD = 0.90


# ──────────────────────────────────────────────────────────────────────────────
# inputs
# ──────────────────────────────────────────────────────────────────────────────
def read_csv(path: str) -> list[dict]:
    """Rows of a CSV as dicts."""
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path: str) -> str:
    """SHA-256 of an input, so the frame names the exact files it was drawn from."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def load_predictions(path: str) -> dict[str, set]:
    """``{report_id: {hpo_id, …}}`` from a predictions file's ``summary: true`` lines.

    The summary line is the contract every earlier method writes. Falling back to the per-row
    ``prediction`` flag would silently disagree with it wherever a method post-filters.
    """
    predicted: dict[str, set] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            report_id = str(row.get("report_id", ""))
            if row.get("summary"):
                predicted[report_id] = set(row.get("predicted_set") or ())
            elif report_id not in predicted and row.get("prediction"):
                predicted.setdefault(report_id, set()).add(row["hpo_id"])
    return predicted


def load_gold(path: str) -> dict[str, set]:
    """``{patient_id: {hpo_code, …}}`` from the curated two-column ground truth."""
    gold = {}
    for row in read_csv(path):
        codes = {code for code in (row.get("hpo_codes") or "").split(";") if code}
        gold[str(row["patient_id"])] = codes
    return gold


# ──────────────────────────────────────────────────────────────────────────────
# scoring
# ──────────────────────────────────────────────────────────────────────────────
def score(predicted: set, gold: set) -> dict:
    """Per-report counts and P/R/F1. An empty ground truth scores ``None``, it is not a 0."""
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    if not gold:
        return {"tp": tp, "fp": fp, "fn": fn, "precision": None, "recall": None, "f1": None}
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def rank_reports(scored: dict, min_gold: int, metric: str = "f1") -> tuple[list, list]:
    """``(best_first, worst_first)`` over reports with at least *min_gold* annotated terms.

    Ties break on gold-set size, descending, at **both** ends: 8/8 is a stronger best case than
    3/3, and 0/12 is a more informative worst case than 0/3.
    """
    eligible = [pid for pid, row in scored.items()
                if row["n_gold"] >= min_gold and row[metric] is not None]
    best = sorted(eligible, key=lambda pid: (-scored[pid][metric], -scored[pid]["n_gold"], pid))
    worst = sorted(eligible, key=lambda pid: (scored[pid][metric], -scored[pid]["n_gold"], pid))
    return best, worst


# ──────────────────────────────────────────────────────────────────────────────
# near-duplicate detection, for the unannotated pair
# ──────────────────────────────────────────────────────────────────────────────
_WORD = re.compile(r"[a-z0-9]+")


def signature(text: str) -> tuple[str, frozenset]:
    """``(exact digest, token set)`` of a normalised report. Neither is reversible to the text."""
    tokens = _WORD.findall((text or "").lower())
    return hashlib.sha256(" ".join(tokens).encode()).hexdigest()[:16], frozenset(tokens)


def duplicate_clusters(signatures: dict, threshold: float) -> list[list]:
    """Group report ids whose token sets overlap at or above *threshold* (single-link)."""
    ids = sorted(signatures)
    parent = {pid: pid for pid in ids}

    def find(pid):
        while parent[pid] != pid:
            parent[pid] = parent[parent[pid]]
            pid = parent[pid]
        return pid

    for i, left in enumerate(ids):
        _, left_tokens = signatures[left]
        for right in ids[i + 1:]:
            _, right_tokens = signatures[right]
            union = len(left_tokens | right_tokens)
            if not union:
                continue
            if len(left_tokens & right_tokens) / union >= threshold:
                parent[find(left)] = find(right)
    clusters: dict[str, list] = {}
    for pid in ids:
        clusters.setdefault(find(pid), []).append(pid)
    return sorted(clusters.values(), key=lambda group: (-len(group), group[0]))


def load_texts(hcy_dir: str, report_ids: list) -> dict:
    """``{pid: report text}`` through the curation app's own loaders, or ``{}`` without *hcy_dir*.

    Imported lazily: the selector must still run, minus duplicate detection, on a machine that
    only has the exported dataset and the predictions.
    """
    if not hcy_dir:
        return {}
    sys.path.insert(0, str(_REPO / "experiments" / "03_setup" / "ground_truth"))
    import curated_ground_truth as curated_gold  # noqa: E402

    corpus = curated_gold.load_corpus(curated_gold.resolve_paths(hcy_dir))
    texts = corpus.get("texts", {})
    wanted = set(report_ids)
    resolved = {pid: texts.get(pid, "") for pid in wanted if texts.get(pid)}
    for pid in wanted - set(resolved):
        segments = corpus.get("segments", {}).get(pid) or []
        if segments:
            resolved[pid] = " ".join(segments)
    return resolved


# ──────────────────────────────────────────────────────────────────────────────
# The draw
# ──────────────────────────────────────────────────────────────────────────────
def fill(pools: dict, sizes: dict, seed: int) -> tuple[dict, list]:
    """Draw *sizes[cell]* distinct reports from *pools[cell]*, most-constrained cell first.

    Slack (``len(pool) - still needed``) is recomputed after every single pick, so a cell whose
    pool has just been raided by another cell is served next. Returns ``(picks, trace)``. A cell
    that runs out is left short and says so in the trace, not raising, the frame should
    record an unfillable cell, not vanish.
    """
    rng = random.Random(seed)
    remaining = {cell: dict.fromkeys(pool) for cell, pool in pools.items()}
    need = dict(sizes)
    picks: dict[str, list] = {cell: [] for cell in pools}
    taken: set = set()
    trace: list = []

    while any(need.values()):
        open_cells = [cell for cell, count in need.items() if count > 0]
        available = {cell: [pid for pid in remaining[cell] if pid not in taken]
                     for cell in open_cells}
        starved = [cell for cell in open_cells if len(available[cell]) < need[cell]]
        cell = min(open_cells, key=lambda name: (len(available[name]) - need[name], name))
        if not available[cell]:
            trace.append({"cell": cell, "report_id": None, "pool_left": 0,
                          "note": "pool exhausted, cell left short"})
            need[cell] = 0
            continue
        pid = rng.choice(available[cell])
        picks[cell].append(pid)
        taken.add(pid)
        need[cell] -= 1
        trace.append({"cell": cell, "report_id": pid, "pool_left": len(available[cell]) - 1,
                      "slack": len(available[cell]) - need[cell] - 1,
                      "note": f"starved cells at this step: {','.join(starved)}" if starved else ""})
    return {cell: sorted(ids) for cell, ids in picks.items()}, trace


#: Consecutive tokens that must coincide before a frame counts as quoting a report. Single words
#: are useless as evidence, this file's own prose says "analysis", "annotation" and "predictions",
#: and so do the reports. Five in a row does not happen between clinical text and boilerplate, but
#: it cannot be avoided by a leak: any quoted segment is far longer than five tokens.
LEAK_NGRAM = 5


def _ngrams(text: str, n: int = LEAK_NGRAM) -> set:
    tokens = _WORD.findall((text or "").lower())
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def assert_no_text(payload, texts: dict) -> None:
    """Hard abort if any report text leaked into what is about to be written.

    The frame is the artifact that travels. A segment quoted in it would move patient data to
    wherever the frame goes. The test is a shared **run of consecutive words**, so it survives
    reformatting and punctuation changes while ignoring the vocabulary the frame and the reports
    inevitably share.
    """
    blob = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    frame_ngrams = _ngrams(blob)
    if not frame_ngrams:
        return
    for pid, text in texts.items():
        overlap = _ngrams(text) & frame_ngrams
        if overlap:
            raise AssertionError(
                f"report text from {pid} appears in the frame "
                f"({len(overlap)} shared {LEAK_NGRAM}-word run(s)), refusing to write")


# ──────────────────────────────────────────────────────────────────────────────
# assembly
# ──────────────────────────────────────────────────────────────────────────────
def build(*, predicted: dict, gold: dict, reports: list, annotations: list,
          texts: dict, seed: int, min_gold: int, metric: str, label_pool_top: int,
          dup_threshold: float) -> dict:
    """Everything the frame records: the per-report table, the seven pools, and the draw."""
    by_report = {row["patient_id"]: row for row in reports}

    label_counts: dict[str, dict] = {}
    for row in annotations:
        pid = row["patient_id"]
        bucket = label_counts.setdefault(pid, dict.fromkeys(LABEL_CELLS, 0))
        for name in LABEL_CELLS:
            if row.get(f"q_{name}") == "1":
                bucket[name] += 1

    scored: dict[str, dict] = {}
    for pid, codes in gold.items():
        row = score(predicted.get(pid, set()), codes)
        row["n_gold"] = len(codes)
        row["n_pred"] = len(predicted.get(pid, set()))
        row["in_predictions"] = pid in predicted
        row.update({name: label_counts.get(pid, {}).get(name, 0) for name in LABEL_CELLS})
        scored[pid] = row

    best, worst = rank_reports(scored, min_gold, metric)
    pools = {"pb_best": best[:10], "pb_worst": worst[:10]}

    for name in LABEL_CELLS:
        carriers = sorted((pid for pid, counts in label_counts.items() if counts[name]),
                          key=lambda pid: (-label_counts[pid][name], pid))
        pools[name] = carriers[:label_pool_top] if label_pool_top else carriers

    unannotated = sorted(pid for pid, row in by_report.items()
                         if row.get("in_cohort") == "0" and int(row.get("n_annotations") or 0) == 0)
    clusters = duplicate_clusters({pid: signature(texts[pid]) for pid in unannotated if pid in texts},
                                  dup_threshold) if texts else []
    # One representative per cluster: two reports from the same cluster are the same report twice.
    if clusters:
        representatives = sorted(group[0] for group in clusters)
        unplaced = [pid for pid in unannotated if pid not in texts]
    else:
        representatives, unplaced = unannotated, []
    pools["no_annotation"] = representatives

    picks, trace = fill(pools, CELL_SIZES, seed)
    return {
        "scored": scored, "pools": pools, "picks": picks, "trace": trace,
        "label_counts": label_counts,
        "n_predicted": {pid: len(codes) for pid, codes in predicted.items()},
        "duplicates": [group for group in clusters if len(group) > 1],
        "unannotated": unannotated, "unannotated_without_text": unplaced,
        "params": {"seed": seed, "min_gold": min_gold, "metric": metric,
                   "label_pool_top": label_pool_top, "dup_jaccard": dup_threshold},
    }


def render(result: dict, inputs: dict) -> str:
    """The human-readable pre-registration, the file a reader of the thesis is pointed at."""
    scored, picks, pools = result["scored"], result["picks"], result["pools"]
    params = result["params"]
    selected = sorted({pid for ids in picks.values() for pid in ids})
    lines = [
        "# HCY deep-dive sampling frame",
        "",
        f"Drawn {date.today().isoformat()} · seed **{params['seed']}** · "
        f"{len(selected)} distinct reports over {len(CELL_SIZES)} cells.",
        "",
        "**This is a purposive sample. No aggregate metric may be quoted from it.** Cells are "
        "defined on outcomes (PhenoBERT's per-report score) and on annotation qualifiers, so any "
        "rate computed over these 20 reports is a rate over the selection rule. Mechanisms travel; "
        "numbers stay on the full 118.",
        "",
        "## Inputs",
        "",
        "| file | sha256 |",
        "|---|---|",
    ]
    lines += [f"| `{name}` | `{value}` |" for name, value in inputs.items()]
    lines += [
        "",
        "## Parameters",
        "",
        f"- ranking metric: **{params['metric']}** per report, against the curated ground truth",
        f"- minimum annotated terms to be rankable: **{params['min_gold']}** "
        "(below it per-report F1 is 0 or 1 and both tails fill with trivial reports)",
        f"- label pool: the **{params['label_pool_top'] or 'all'}** richest carriers of each "
        "qualifier",
        f"- near-duplicate threshold: token-set Jaccard **{params['dup_jaccard']}**",
        "",
        "## The draw",
        "",
        "| cell | n | pool | drawn |",
        "|---|--:|--:|---|",
    ]
    for cell, size in CELL_SIZES.items():
        lines.append(f"| `{cell}` | {size} | {len(pools[cell])} | "
                     f"{', '.join(picks[cell]) or '**unfilled**'} |")

    lines += ["", "## Selected reports", "",
              "| report | cell | ground truth | pred | TP | FP | FN | P | R | F1 | fam | lab | impl | neg |",
              "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    cell_of = {pid: cell for cell, ids in picks.items() for pid in ids}
    for pid in selected:
        row = scored.get(pid)
        if row is None:
            # No ground truth, so no TP/FP/FN column means anything, except `pred`, which is the whole
            # point of the cell: every term PhenoBERT emits here is a false positive by definition.
            counts = result["label_counts"].get(pid, {})
            n_pred = result["n_predicted"].get(pid, 0)
            lines.append(f"| {pid} | `{cell_of[pid]}` | 0 | {n_pred} |, | {n_pred} |, |, |, "
                         "|, | "
                         + " | ".join(str(counts.get(name, 0)) for name in LABEL_CELLS) + " |")
            continue
        lines.append(
            f"| {pid} | `{cell_of[pid]}` | {row['n_gold']} | {row['n_pred']} | {row['tp']} | "
            f"{row['fp']} | {row['fn']} | {row['precision']:.2f} | {row['recall']:.2f} | "
            f"{row['f1']:.2f} | " + " | ".join(str(row[name]) for name in LABEL_CELLS) + " |")

    if result["duplicates"]:
        lines += ["", "## Near-duplicate clusters among the unannotated reports", "",
                  "Only one report per cluster was eligible for the `no_annotation` cell.", ""]
        lines += [f"- {' ≡ '.join(group)}" for group in result["duplicates"]]
    if result["unannotated_without_text"]:
        lines += ["", f"**{len(result['unannotated_without_text'])} unannotated report(s) carry no "
                  "text and could not be duplicate-checked**: "
                  + ", ".join(result["unannotated_without_text"])]

    lines += ["", "## Notes carried into the analysis", "",
              "- `family` annotations are **excluded from the curated ground truth** "
              "(`curated_gold.DEFAULT_EXCLUDE_LABELS`), so a method predicting one scores a false "
              "positive it arguably should not. That is the point of the cell, not a bug in it.",
              "- The `no_annotation` reports have an empty ground truth, so every prediction on them is a "
              "false positive by design, read them as a precision probe only.",
              "- Reports below the ground truth floor never enter the PhenoBERT ranking; they are neither "
              "best nor worst cases, they are unrankable.",
              "- Methods with learned components (`an earlier exploratory run`/`_14`/`_16`) are cross-validated on "
              "HCY. Check each selected report against its fold before comparing them here.",
              ""]
    return "\n".join(lines)


def write_outputs(out_dir: str, result: dict, inputs: dict, texts: dict) -> list:
    """Write the HCY selection frame (JSON and CSV) to *out_dir* and return the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = sorted({pid for ids in result["picks"].values() for pid in ids})
    payload = {
        "generated": date.today().isoformat(), "inputs": inputs, "params": result["params"],
        "cell_sizes": CELL_SIZES, "pools": result["pools"], "picks": result["picks"],
        "selected": selected, "trace": result["trace"],
        "duplicates": result["duplicates"], "unannotated": result["unannotated"],
    }
    assert_no_text(payload, texts)
    written = []

    (out / "frame.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    written.append(str(out / "frame.json"))

    text = render(result, inputs)
    assert_no_text(text, texts)
    (out / "FRAME.md").write_text(text, encoding="utf-8")
    written.append(str(out / "FRAME.md"))

    fields = ["patient_id", "selected", "cell", "n_gold", "n_pred", "tp", "fp", "fn",
              "precision", "recall", "f1", *LABEL_CELLS]
    cell_of = {pid: cell for cell, ids in result["picks"].items() for pid in ids}
    with open(out / "pools.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pid, row in sorted(result["scored"].items()):
            writer.writerow({"patient_id": pid, "selected": int(pid in cell_of),
                             "cell": cell_of.get(pid, ""),
                             **{key: row.get(key) for key in fields[3:]}})
    written.append(str(out / "pools.csv"))
    return written


# ──────────────────────────────────────────────────────────────────────────────
# selftest
# ──────────────────────────────────────────────────────────────────────────────
def selftest() -> int:
    """Exercise scoring, ranking, duplicate collapse and the greedy fill on synthetic data."""
    failures = []

    def check(name, condition):
        if not condition:
            failures.append(name)

    row = score({"a", "b", "c"}, {"b", "c", "d"})
    check("score counts", (row["tp"], row["fp"], row["fn"]) == (2, 1, 1))
    check("score f1", abs(row["f1"] - 2 / 3) < 1e-9)
    check("empty ground truth is None", score({"a"}, set())["f1"] is None)

    scored = {"p1": {"f1": 1.0, "n_gold": 3}, "p2": {"f1": 1.0, "n_gold": 9},
              "p3": {"f1": 0.0, "n_gold": 2}, "p4": {"f1": 0.5, "n_gold": 5}}
    best, worst = rank_reports(scored, min_gold=3)
    check("ground truth floor excludes p3", "p3" not in best and "p3" not in worst)
    check("ties break on ground truth size", best[0] == "p2")
    check("worst first", worst[0] == "p4")

    sigs = {"a": signature("the patient had seizures and hypotonia"),
            "b": signature("The patient had seizures and hypotonia."),
            "c": signature("entirely different words appear in this letter")}
    clusters = duplicate_clusters(sigs, 0.9)
    check("case/punctuation duplicates collapse", any(set(g) == {"a", "b"} for g in clusters))
    check("distinct report stays alone", ["c"] in clusters)

    pools = {"pb_best": [f"b{i}" for i in range(10)], "pb_worst": [f"w{i}" for i in range(10)],
             "no_annotation": ["n0", "n1", "n2"],
             **{name: [f"b{i}" for i in range(3)] + [f"L{name}{i}" for i in range(7)]
                for name in LABEL_CELLS}}
    picks, _ = fill(pools, CELL_SIZES, seed=7)
    flat = [pid for ids in picks.values() for pid in ids]
    check("draw is the right size", len(flat) == sum(CELL_SIZES.values()))
    check("reports are distinct", len(set(flat)) == len(flat))
    check("every cell filled", all(len(picks[c]) == n for c, n in CELL_SIZES.items()))
    check("picks come from their pool", all(pid in pools[c] for c, ids in picks.items()
                                            for pid in ids))
    again, _ = fill(pools, CELL_SIZES, seed=7)
    check("draw is reproducible", again == picks)
    check("a different seed draws differently", fill(pools, CELL_SIZES, seed=8)[0] != picks)

    short = fill({**pools, "no_annotation": ["n0"]}, CELL_SIZES, seed=7)
    check("an exhausted pool is recorded, not raised", len(short[0]["no_annotation"]) == 1)

    report = "The patient was referred for analysis of recurrent seizures and marked hypotonia."
    try:
        assert_no_text({"note": f"quoted: {report}"}, {"p1": report})
        check("a quoted segment is refused", False)
    except AssertionError:
        pass
    # The regression: the frame's own prose shares vocabulary with every report it describes
    # ("analysis", "annotation", "predictions"). Shared words are not a leak. Shared runs are.
    boilerplate = ("Notes carried into the analysis. The annotation qualifiers and the "
                   "predictions contract are described above; no patient text appears here.")
    try:
        assert_no_text({"note": boilerplate}, {"p1": report})
    except AssertionError as exc:
        check(f"shared vocabulary is not a leak ({exc})", False)

    for name in failures:
        print(f"FAIL  {name}")
    print(f"{'FAILED' if failures else 'ok'}, {len(failures)} failure(s)")
    return 1 if failures else 0


# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    """Select the HCY reports shown in the Compare UI. Returns the exit status."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", help="baseline_phenobert/hcy/phenobert_predictions.jsonl")
    parser.add_argument("--dataset", help="<hcy>/curated_ground_truth_<date>/ from the ground-truth build")
    parser.add_argument("--hcy-dir", default="", help="for near-duplicate detection (optional)")
    parser.add_argument("--out", default="output/hcy_deepdive_frame")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--metric", default="f1", choices=("f1", "recall", "precision"))
    parser.add_argument("--min-gold", type=int, default=DEFAULT_MIN_GOLD)
    parser.add_argument("--label-pool-top", type=int, default=10,
                        help="draw each qualifier cell from the N richest carriers; 0 = all")
    parser.add_argument("--dup-jaccard", type=float, default=DEFAULT_DUP_JACCARD)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if not (args.predictions and args.dataset):
        parser.error("--predictions and --dataset are required (or use --selftest)")

    dataset = Path(args.dataset)
    paths = {"predictions": args.predictions,
             "annotations": str(dataset / ANNOTATIONS_FILE),
             "reports": str(dataset / REPORTS_FILE),
             "gold": str(dataset / GOLD_FILE)}
    missing = [name for name, path in paths.items() if not os.path.isfile(path)]
    if missing:
        print(f"missing input(s): {', '.join(f'{n} ({paths[n]})' for n in missing)}",
              file=sys.stderr)
        return 2

    reports = read_csv(paths["reports"])
    annotations = read_csv(paths["annotations"])
    gold = load_gold(paths["gold"])
    predicted = load_predictions(paths["predictions"])
    texts = load_texts(args.hcy_dir, [row["patient_id"] for row in reports])

    absent = sorted(set(gold) - set(predicted))
    if absent:
        print(f"warning: {len(absent)} cohort report(s) have no PhenoBERT prediction "
              f"(scored as empty): {', '.join(absent[:5])}…", file=sys.stderr)

    result = build(predicted=predicted, gold=gold, reports=reports, annotations=annotations,
                   texts=texts, seed=args.seed, min_gold=args.min_gold, metric=args.metric,
                   label_pool_top=args.label_pool_top, dup_threshold=args.dup_jaccard)

    inputs = {name: digest(path) for name, path in paths.items()}
    written = write_outputs(args.out, result, inputs, texts)

    short = [cell for cell, size in CELL_SIZES.items() if len(result["picks"][cell]) < size]
    print(render(result, inputs))
    print("\nwrote:\n  " + "\n  ".join(written))
    if short:
        print(f"\nUNFILLED CELLS: {', '.join(short)}, the pools are too small or overlap too "
              "much. Raise --label-pool-top, lower --min-gold, or accept a smaller sample.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
