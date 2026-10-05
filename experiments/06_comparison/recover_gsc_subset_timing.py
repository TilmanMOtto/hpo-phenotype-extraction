"""Recover per-document runtimes on GSC+ from traces the runs already left, and sum them per subset.

Every GSC+ system was run ONCE over all 228 abstracts. The two scored subsets (RAG-HPO's 114,
AutoPCR's 206) are cut from that run's predictions offline. The cost stage of the comparison therefore
divides a run total by 228 and prints the same seconds/report under both subset headings. Nobody
timed a document -- but most runs wrote something per document with a clock attached, and this
script reads those traces back:

    system         component   trace                                              resolution
    ─────────────────────────────────────────────────────────────────────────────────────────
    AutoPCR 8B/70B annotate    tqdm lines in the SLURM stderr, one per abstract     1 s
                   parse       tqdm lines of the shared neural++ parser job          1 s
    RAG-HPO 8B/70B stage 1     tqdm "Extract+Retrieve", one line per abstract        1 s
                   stage 2     tqdm "LLM HPO mapping", one line per PHRASE -- no     APPROXIMATE
                               abstract id, so its seconds are apportioned by the
                               abstract's non-exact findings (see `_raghpo_stage2_weights`)
    PhenoBERT      annotate    output-file mtimes, one file per abstract, written    ns
                               in sequence by one annotate.py process
    PhenoJury      generation  run.log DEBUG line per (juror, abstract, sentence)    1 s
                   grounding   mtimes of phenobert_output_<juror>/, one file per     ns
                               abstract, from the last (complete) grounding pass

TreePhenoRAG is not here: its GSC+ run was already restricted to the 114 RAG-HPO ids (a cohort
allowlist), so its cost is subset-specific as it stands, and it never ran on the 206.

**Overhead is per run, not per document.** Model load and any setup outside the per-document
loop, plus the FIRST document of every trace, are kept as `overhead_s` and charged ONCE to each
subset -- what running that subset alone would pay. It is never spread over documents. The first
document goes with it because it absorbs lazy initialisation: the neural++ parser's first abstract
took 135 s of the job's 226 s, and a trace whose clock starts at process start (file mtimes, a
resumed generation pass) cannot separate the load from it at all.

**Every trace is checked against the total the cost table already uses** (the driver's timing
file, or the juror's `Extract done` line), and the check is written out in `verification.csv`.
A trace whose loop total does not match its timing file is from a different run and aborts.

GSC+ only, by design: every input path is refused if it names `hcy`, and every tqdm trace
must count 228 documents (HCY has 118). Runs on the cluster, where the SLURM logs are:

    python experiments/06_comparison/recover_gsc_subset_timing.py \\
        --output-root <results_dir> \\
        --log-root <logs_dir>

Writes `<output-root>/gsc_subset_timing/gsc/{per_document,overhead,subset_seconds,verification}.csv`
-- under a `gsc/` directory, which marks it as public-corpus output that may leave LeoMed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from hpo_extraction.paths import REPO_ROOT as REPO  # noqa: E402
N_DOCS = 228
SUBSETS = {
    "gsc_raghpo_ann": "resources/data/GSC_RAGHPO/document_ids.txt",
    "gsc_2024_eval_206": "resources/data/GSC_2024/eval_206_ids.txt",
}

#: The SLURM jobs whose runs wrote the timing files the comparison reads. Found with
#: `sacct --name=the AutoPCR 8B baseline,...`. Each is verified against its timing file below.
JOBS = {
    "raghpo_8b": 9612572, "raghpo_70b": 9612574,
    "autopcr_8b": 9612576, "autopcr_70b": 9612578,
    "autopcr_parse": 9611114,
}
EXP = {
    "raghpo_8b": "baseline_raghpo_8b", "raghpo_70b": "baseline_raghpo_70b",
    "autopcr_8b": "baseline_autopcr_8b", "autopcr_70b": "baseline_autopcr_70b",
    "phenobert": "baseline_phenobert", "phenojury": "phenojury_generation_free_listing",
}
JURORS = ["apertus", "deepseek", "intelligent_internet", "llama", "medgemma", "medpsy",
          "openbiollm", "phi4"]
CONDITIONS = {"phenojury": JURORS, "phenojury_core3": ["medgemma", "medpsy", "phi4"]}

#: A loop total may fall short of the timing file's duration by the setup outside the loop, never
#: exceed it. Seconds of slack for tqdm's 1 s rounding.
SLACK_S = 2.0

TQDM = re.compile(r"(?:(?P<desc>[A-Za-z+ ]+):\s*)?\d+%\|[^|]*\|\s*(?P<n>\d+)/(?P<total>\d+)\s*"
                  r"\[(?P<elapsed>[\d:]+)[<\]]")
LOG_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\w+)\s+(\S+)\s+(.*)$")
SENT = re.compile(r"^(?P<model>\w+) \| (?P<report>\S+) \| sent (?P<sent>\d+) \|")


def refuse_patient_data(*paths: Path) -> None:
    """Stop if any path looks like HCY data: this script reads GSC+ traces only."""
    for p in paths:
        if "hcy" in str(p).lower():
            raise SystemExit(f"refusing {p}: this script reads GSC+ traces only")


def _seconds(elapsed: str) -> int:
    out = 0
    for part in elapsed.split(":"):
        out = out * 60 + int(part)
    return out


def tqdm_elapsed(path: Path, desc: str, total: int) -> list[int]:
    """Elapsed seconds at n = 0..total for the LAST loop in *path* with this desc and total."""
    refuse_patient_data(path)
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    seqs, cur = [], None
    for line in text.split("\n"):
        m = TQDM.search(line)
        if not m or int(m["total"]) != total or (m["desc"] or "").strip() != desc:
            continue
        n = int(m["n"])
        if n == 0:
            cur = {}
            seqs.append(cur)
        if cur is not None:
            cur.setdefault(n, _seconds(m["elapsed"]))
    if not seqs:
        raise SystemExit(f"{path}: no tqdm loop {desc!r} over {total}")
    last = seqs[-1]
    missing = [n for n in range(total + 1) if n not in last]
    if missing:
        raise SystemExit(f"{path}: loop {desc!r} lacks {len(missing)} step(s), e.g. {missing[:5]}")
    return [last[n] for n in range(total + 1)]


def timing_file(path: Path) -> dict:
    """The last record of a timing JSON Lines file."""
    refuse_patient_data(path)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows[-1]


def check(ver: list, system: str, component: str, traced: float, reference: float, what: str,
          allow_short: bool = True) -> None:
    """Append a check of a traced duration against a reference duration (seconds) to *ver*."""
    gap = reference - traced
    ok = -SLACK_S <= gap if allow_short else abs(gap) <= SLACK_S
    ver.append(dict(system=system, component=component, traced_s=round(traced, 1),
                    reference_s=round(reference, 1), gap_s=round(gap, 1), reference=what,
                    ok=ok))
    if not ok:
        raise SystemExit(f"{system}/{component}: traced {traced:.1f}s vs {what} {reference:.1f}s "
                         "-- the trace is not from the run the cost table reads")


# ── AutoPCR ──────────────────────────────────────────────────────────────────────────────────
def corpus_order(exp_dir: Path) -> list[str]:
    """Document ids in the order AutoPCR processed its staged GSC+ corpus."""
    path = exp_dir / "gsc/autopcr_corpus/corpus_test.tsv"
    refuse_patient_data(path)
    blocks = path.read_text(encoding="utf-8").strip().split("\n\n")
    ids = [b.split("\n", 1)[0].strip() for b in blocks]
    if len(ids) != N_DOCS:
        raise SystemExit(f"{path}: {len(ids)} documents, expected {N_DOCS}")
    return ids


def autopcr(root: Path, logs: Path, docs: list, over: list, ver: list) -> None:
    """Per-document seconds of both AutoPCR runs, from their logs."""
    order = corpus_order(root / EXP["autopcr_8b"])
    for system in ("autopcr_8b", "autopcr_70b"):
        if corpus_order(root / EXP[system]) != order:
            raise SystemExit(f"{system}: staged corpus order differs from exp13_19's")
        el = tqdm_elapsed(logs / f"{JOBS[system]}_stderr.log", "", N_DOCS)
        for i, rid in enumerate(order[1:], start=1):
            docs.append(dict(system=system, component="annotate", report_id=rid,
                             seconds=el[i + 1] - el[i], exact=True))
        t = timing_file(root / EXP[system] / "gsc/autopcr_timing.jsonl")
        over.append(dict(system=system, component="annotate",
                         seconds=round(float(t["annotate_s"]) - el[-1] + el[1], 3),
                         what=f"annotate_s outside the loop + the first abstract ({el[1]} s)"))
        check(ver, system, "annotate", el[-1], float(t["annotate_s"]), "timing annotate_s")
    # The neural++ parse ran once, as its own job, and both linkers load its cache. The cost table
    # does not include it. It is written as its own component so the reader can decide.
    el = tqdm_elapsed(logs / f"{JOBS['autopcr_parse']}_stderr.log", "", N_DOCS)
    out = (logs / f"{JOBS['autopcr_parse']}_stdout.log").read_text(errors="replace")
    m = re.search(r"parsed (\d+) document\(s\) in (\d+)s", out)
    if not m or int(m[1]) != N_DOCS:
        raise SystemExit("parser job: no 'parsed 228 document(s)' line")
    for i, rid in enumerate(order[1:], start=1):
        docs.append(dict(system="autopcr_parse", component="parse", report_id=rid,
                         seconds=el[i + 1] - el[i], exact=True))
    over.append(dict(system="autopcr_parse", component="parse",
                     seconds=round(float(m[2]) - el[-1] + el[1], 3),
                     what=f"outside the loop + the first abstract ({el[1]} s, parser warm-up)"))
    check(ver, "autopcr_parse", "parse", el[-1], float(m[2]), "parser job 'parsed … in Xs'")


# ── RAG-HPO ──────────────────────────────────────────────────────────────────────────────────
def prediction_order(path: Path) -> list[str]:
    """Document ids in the order a RAG-HPO run wrote its predictions."""
    refuse_patient_data(path)
    ids = [json.loads(l)["report_id"] for l in path.read_text().splitlines()
           if l.strip() and json.loads(l).get("summary")]
    if len(ids) != N_DOCS:
        raise SystemExit(f"{path}: {len(ids)} summary rows, expected {N_DOCS}")
    return ids


def _raghpo_stage2_weights(path: Path, order: list) -> tuple[dict, str]:
    """Findings per abstract that went to LLM mapping, as far as the saved rows can tell.

    The driver writes its findings as `exact_df` then `non_ex`, each in abstract order, one row
    per candidate (rank 1..5). So the rank-1 rows run through the abstracts TWICE, and the second
    run is the non-exact findings -- the ones stage 2 maps, bar those not categorised "Abnormal"
    (the category is not saved). The split is where the abstract order first goes backwards. If
    the rows do not show one such reset, fall back to all findings and say so.
    """
    refuse_patient_data(path)
    pos = {rid: i for i, rid in enumerate(order)}
    seq = [json.loads(l)["report_id"] for l in path.read_text().splitlines()
           if l.strip() and json.loads(l).get("rank") == 1]
    resets = [i for i in range(1, len(seq)) if pos[seq[i]] < pos[seq[i - 1]]]
    if len(resets) == 1:
        rows, how = seq[resets[0]:], "non-exact findings (rank-1 rows after the order resets)"
    else:
        rows, how = seq, f"all findings ({len(resets)} order resets, expected 1)"
    counts = defaultdict(int)
    for rid in rows:
        counts[rid] += 1
    return counts, how


def raghpo(root: Path, logs: Path, docs: list, over: list, ver: list, notes: list) -> None:
    """Per-document seconds of both RAG-HPO runs, from their logs."""
    for system in ("raghpo_8b", "raghpo_70b"):
        d = root / EXP[system] / "gsc"
        order = prediction_order(d / "rag_hpo_predictions.jsonl")
        if order != sorted(order):
            raise SystemExit(f"{system}: prediction order is not the driver's sorted order")
        err = logs / f"{JOBS[system]}_stderr.log"
        el = tqdm_elapsed(err, "Extract+Retrieve", N_DOCS)
        for i, rid in enumerate(order[1:], start=1):
            docs.append(dict(system=system, component="stage1", report_id=rid,
                             seconds=el[i + 1] - el[i], exact=True))
        t = timing_file(d / "rag_hpo_timing.jsonl")
        n_phrases = int(t["n_llm_calls"]) - N_DOCS          # one extraction call per abstract
        el2 = tqdm_elapsed(err, "LLM HPO mapping", n_phrases)
        weights, how = _raghpo_stage2_weights(d / "rag_hpo_retrieved_segments.jsonl", order)
        total_w = sum(weights.values())
        for rid in order:
            docs.append(dict(system=system, component="stage2", report_id=rid,
                             seconds=round(el2[-1] * weights.get(rid, 0) / total_w, 3),
                             exact=False))
        notes.append(f"{system} stage 2 ({el2[-1]} s over {n_phrases} phrases) apportioned by "
                     f"{how}: {total_w} rows")
        over.append(dict(system=system, component="stage1+2",
                         seconds=round(float(t["duration_s"]) - el[-1] - el2[-1] + el[1], 3),
                         what=f"duration_s outside both loops + stage 1's first abstract "
                              f"({el[1]} s)"))
        check(ver, system, "stage1+2", el[-1] + el2[-1], float(t["duration_s"]),
              "timing duration_s")


# ── PhenoBERT and PhenoJury grounding: one output file per abstract ─────────────────────────
def mtime_trace(directory: Path) -> tuple[list, float, float]:
    """(per-abstract seconds for files 2..n, first mtime, last mtime), in write order."""
    refuse_patient_data(directory)
    files = sorted((p.stat().st_mtime, p.stem) for p in directory.iterdir() if p.is_file())
    if len(files) != N_DOCS:
        raise SystemExit(f"{directory}: {len(files)} files, expected {N_DOCS}")
    per = [(stem, t - files[i - 1][0]) for i, (t, stem) in enumerate(files) if i]
    return per, files[0][0], files[-1][0]


def phenobert(root: Path, docs: list, over: list, ver: list) -> None:
    """Per-document seconds of the PhenoBERT run, from its output file times."""
    d = root / EXP["phenobert"] / "gsc"
    per, first, last = mtime_trace(d / "phenobert_output")
    for stem, s in per:
        docs.append(dict(system="phenobert", component="annotate", report_id=stem,
                         seconds=round(s, 3), exact=True))
    t = timing_file(d / "phenobert_timing.jsonl")
    over.append(dict(system="phenobert", component="annotate",
                     seconds=round(float(t["annotate_s"]) - (last - first), 3),
                     what="annotate_s before the first output file (model load + first abstract)"))
    check(ver, "phenobert", "annotate", last - first, float(t["annotate_s"]), "timing annotate_s")


# ── PhenoJury ────────────────────────────────────────────────────────────────────────────────
def _parse_log(path: Path):
    refuse_patient_data(path)
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = LOG_TS.match(line.rstrip("\n"))
            if m:
                yield datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S").timestamp(), m[4]


def phenojury(root: Path, docs: list, over: list, ver: list, notes: list) -> None:
    """Per-document seconds of every juror's generation, from the shared log."""
    d = root / EXP["phenojury"] / "gsc"
    # Pass boundaries per juror: any non-sentence line naming the juror (resume, load, unload,
    # Extract done). Other jurors' lines interleave -- they ran as an array into one log.
    blocks = {j: [] for j in JURORS}          # juror -> list of passes -> list of (t, report)
    last_marker = {j: None for j in JURORS}
    last_load = {j: None for j in JURORS}     # "Loading <juror> from ..." -- the pass's true start
    extract_done, extract_grounding = {}, {}
    grounding_done = {}
    for t, msg in _parse_log(d / "run.log"):
        m = SENT.match(msg)
        if m and m["model"] in blocks:
            j = m["model"]
            if last_marker[j] is not None or not blocks[j]:
                # From the model load when this pass loaded one, so the load is in the
                # pass's first abstract (i.e. its overhead) and not lost between markers.
                start = last_load[j] if last_load[j] is not None else (last_marker[j] or t)
                blocks[j].append({"start": start, "lines": []})
                last_marker[j] = last_load[j] = None
            blocks[j][-1]["lines"].append((t, m["report"]))
            continue
        for j in JURORS:
            # Not "\bllama\b": model paths (meta-llama/...) would split llama's pass.
            if re.search(rf"(?<![-/\w]){j}(?![-\w])", msg):
                last_marker[j] = t
                if re.search(rf"Loading {j} from", msg):
                    last_load[j] = t
                done = re.search(rf"Extract done \| model={j} .* in ([\d.]+)s \| peak GPU ([\d.]+)",
                                 msg)
                g = re.search(rf"PhenoBERT {j}: done in ([\d.]+)s", msg)
                if g:
                    grounding_done[j] = float(g[1])
                if done and float(done[2]) > 0.05:
                    # The generating pass's clock also covers its own grounding, logged just
                    # before it. That pass's grounding found 0 detections and was redone later.
                    extract_done[j] = float(done[1])
                    extract_grounding[j] = grounding_done.get(j, 0.0)
    for j in JURORS:
        # The last generation of every abstract wins. An earlier one was either cut off by a
        # wall-clock kill (a report's records land only when it finishes) or superseded by a
        # regeneration. Either way it is not what the predictions came from.
        final, wasted, first_of_pass, n_passes = {}, 0.0, 0, 0
        for p in blocks[j]:
            n_passes += 1
            prev_t, prev_r = p["start"], None
            runs = []                          # (report, start, end) per contiguous block
            for t, rid in p["lines"]:
                if rid != prev_r:
                    if runs:
                        prev_t = runs[-1][2]
                    runs.append([rid, prev_t, t])
                    prev_r = rid
                else:
                    runs[-1][2] = t
            for k, (rid, start, end) in enumerate(runs):
                if rid in final:
                    wasted += final[rid][1]
                final[rid] = (k == 0, end - start)
        gen_over = sum(s for first, s in final.values() if first)
        first_of_pass = sum(1 for first, _ in final.values() if first)
        for rid, (first, s) in final.items():
            if not first:
                docs.append(dict(system=f"juror:{j}", component="generation", report_id=rid,
                                 seconds=s, exact=True))
        over.append(dict(system=f"juror:{j}", component="generation", seconds=gen_over,
                         what=f"pass start to first abstract's end, {first_of_pass} pass(es) "
                              "(model load + that abstract)"))
        if len(final) != N_DOCS:
            notes.append(f"{j}: {N_DOCS - len(final)} abstract(s) have no sentence line (lost "
                         "log lines); their seconds sit in the next abstract's")
        if wasted:
            notes.append(f"{j}: {wasted:.0f} s of earlier generation (killed or superseded) "
                         "not counted")
        per, first, last = mtime_trace(d / f"phenobert_output_{j}")
        for stem, s in per:
            docs.append(dict(system=f"juror:{j}", component="grounding", report_id=stem,
                             seconds=round(s, 3), exact=True))
        if j in grounding_done:
            over.append(dict(system=f"juror:{j}", component="grounding",
                             seconds=round(grounding_done[j] - (last - first), 3),
                             what="PhenoBERT 'done in' before the first output file"))
            check(ver, f"juror:{j}", "grounding", last - first, grounding_done[j],
                  "run.log 'PhenoBERT <juror>: done in'")
        traced_gen = sum(s for _, s in final.values())
        if j in extract_done:
            check(ver, f"juror:{j}", "generation", traced_gen,
                  extract_done[j] - extract_grounding[j],
                  "generating pass 'Extract done' minus its 'PhenoBERT done in'")
        else:
            ver.append(dict(system=f"juror:{j}", component="generation", traced_s=round(traced_gen, 1),
                            reference_s="", gap_s="", ok=True,
                            reference=f"no single clock ({n_passes} passes); trace is the only total"))


# ── Assembly ─────────────────────────────────────────────────────────────────────────────────
def subset_rows(docs: list, over: list, repo: Path) -> list:
    """Seconds per document for each system on each GSC+ frame (114, 206, all 228)."""
    ids = {c: {l.strip() for l in (repo / p).read_text().splitlines() if l.strip()}
           for c, p in SUBSETS.items()}
    ids["gsc_all_228"] = None
    by_sys = defaultdict(lambda: defaultdict(float))
    exact = defaultdict(lambda: True)
    per_doc = defaultdict(lambda: defaultdict(float))
    for r in docs:
        per_doc[r["system"]][r["report_id"]] += r["seconds"]
        exact[r["system"]] &= bool(r["exact"])
    overhead = defaultdict(float)
    for r in over:
        overhead[r["system"]] += r["seconds"]
    systems = {s: [s] for s in ("raghpo_8b", "raghpo_70b", "autopcr_8b", "autopcr_70b",
                                "phenobert", "autopcr_parse")}
    systems.update({arm: [f"juror:{j}" for j in js] for arm, js in CONDITIONS.items()})
    out = []
    for method, parts in systems.items():
        for cohort, keep in ids.items():
            docs_s = sum(s for p in parts for rid, s in per_doc[p].items()
                         if keep is None or rid in keep)
            n = N_DOCS if keep is None else len(keep)
            ovh = sum(overhead[p] for p in parts)
            out.append(dict(method=method, cohort=cohort, n_documents=n,
                            document_seconds=round(docs_s, 1), overhead_seconds=round(ovh, 1),
                            seconds_per_report=round((docs_s + ovh) / n, 2),
                            exact=all(exact[p] for p in parts)))
    return out


def write(path: Path, rows: list) -> None:
    """Write *rows* (dicts) to a CSV at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def main() -> None:
    """Recover per-subset GSC+ timings from the run logs and write ``subset_seconds.csv``."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--log-root", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=REPO,
                    help="checkout holding the subset id lists (default: this script's)")
    args = ap.parse_args()
    refuse_patient_data(args.output_root, args.log_root)
    docs, over, ver, notes = [], [], [], []
    autopcr(args.output_root, args.log_root, docs, over, ver)
    raghpo(args.output_root, args.log_root, docs, over, ver, notes)
    phenobert(args.output_root, docs, over, ver)
    phenojury(args.output_root, docs, over, ver, notes)
    out = args.output_root / "gsc_subset_timing" / "gsc"
    write(out / "per_document.csv", docs)
    write(out / "overhead.csv", over)
    write(out / "verification.csv", ver)
    rows = subset_rows(docs, over, args.repo)
    write(out / "subset_seconds.csv", rows)
    (out / "notes.txt").write_text("\n".join(notes) + "\n")
    for n in notes:
        print("  note:", n)
    for r in rows:
        print(f"  {r['method']:18s} {r['cohort']:18s} {r['seconds_per_report']:>9.2f} s/report"
              + ("" if r["exact"] else "  (approx)"))


if __name__ == "__main__":
    sys.exit(main())
