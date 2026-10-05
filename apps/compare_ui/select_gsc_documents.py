"""Draw the 20-abstract GSC+ comparison sample, and pre-register the draw.

`app/compare_ui` answers one question: standing on one document, *why* did PhenoJury find
``HP:0001250`` and RAG-HPO not. On HCY the twenty reports it reads come from
``select_hcy_deepdive.py``, whose cells are about PhenoBERT's per-report score and the curator's
difficulty labels. That is the right frame for patient data and the wrong one for a corpus whose
whole value is that it is **published**: GSC+ abstracts may be read by a model, quoted in the
thesis and shown to somebody who does not have cluster access, so the sample worth drawing there
is about *disagreement between the three linking strategies*.

    cell         n   a document qualifies when it carries at least one annotated term where...
    ─────────────────────────────────────────────────────────────────────────────────────
    all_right    5   PhenoJury, RAG-HPO and AutoPCR all found it
    pj_only      5   PhenoJury found it and the other two did not
    pj_wrong     5   PhenoJury missed it and at least one of the other two found it
    all_wrong    5   none of the three found it

The three columns are ``roster.FRAME_METHODS`` and nothing else. TreePhenoRAG is not
in the rule: on GSC+ it is a transfer condition with nothing fitted on the cohort, so a cell defined on
its verdict would be a cell about a different estimator. It still draws as a column in the app,
where its predictions exist.

Four properties make this a frame rather than a shortlist:

1. **"Right" and "wrong" are about an annotated term, not about precision.** A term is *right* for a
   method when the method predicted it and the ground truth carries it. A false positive is a different
   failure and is not what any of these four cells is about -- ``all_wrong`` means "a phenotype
   the annotators recorded that none of the three produced", which is the case worth reading. The
   per-document false-positive counts are written to ``pools.csv`` anyway, because a reader
   looking at one of these abstracts will want them.

2. **Membership is ontological, as in the builder.** A ground truth code and a predicted code can
   be two spellings of one phenotype, so both are resolved through the fixed ontology before they
   are compared (``apps.compare_ui.sources.get_view``). Without this a cell could claim PhenoJury
   got a term right while the app, which does resolve, draws it red. Running without an ontology
   is possible and is recorded in the frame as ``resolved: false`` -- a frame that cannot say
   which it did is worse than either.

3. **The strict reading of a cell is preferred, and the relaxed one is recorded.** "Only PhenoJury
   got it right, while the other two (or at least one) was wrong" is two rules. The strict one
   (*neither* of the others found it) is what fills the cell. The relaxed one (*at least one* did
   not) is the fallback when the strict pool runs out, and each pick says which rule it came in
   under. Silently using the relaxed rule would make ``pj_only`` mean something weaker than its
   name, and only the file would know.

4. **It is written before anything is read.** The seed, the input digests, the full pools, the
   witness terms and the draw all land in one directory. The deep dive then happens against that
   file. A document that turns out to be interesting cannot be added afterwards without the diff
   showing it. The draw is random within each pool, not extremal, and the twenty are
   distinct -- a document satisfying two cells is spent on one of them, cells filled
   most-constrained-first.

Patient-data boundary
---------------------
GSC+ is published journal abstracts and may leave the cluster. That is the point of this frame.
Even so, nothing here writes document *text* into its outputs -- the frame carries ids, codes and
counts. ``select_hcy_deepdive.assert_no_text`` is reused to enforce it, so the two frames have one
leak check between them, not two.

Usage
-----
::

    python apps/compare_ui/select_gsc_documents.py \\
        --results-dir  $REPO/output \\
        --raghpo-dir   $REPO/resources/data/GSC_RAGHPO \\
        --out          $REPO/output/gsc_compare_frame

    python apps/compare_ui/select_gsc_documents.py --selftest     # no data needed
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _path in (str(_REPO), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from apps.compare_ui import cohorts, gold as gold_mod, roster, sources  # noqa: E402
from apps.compare_ui.methods import base  # noqa: E402
from select_hcy_documents import assert_no_text, digest, fill  # noqa: E402

logger = logging.getLogger(__name__)

#: ``{cell: n}`` -- the shape of the sample. Changing this changes the pre-registration, so it is
#: a constant, not a flag.
CELL_SIZES: dict[str, int] = {
    "all_right": 5,
    "pj_only": 5,
    "pj_wrong": 5,
    "all_wrong": 5,
}

#: The order cells are described in, and the order the app's picker lists them.
CELL_ORDER: tuple[str, ...] = ("all_right", "pj_only", "pj_wrong", "all_wrong")

CELL_HELP: dict[str, str] = {
    "all_right": "an annotated term all three of PhenoJury, RAG-HPO and AutoPCR found",
    "pj_only": "an annotated term PhenoJury found and the other two did not",
    "pj_wrong": "an annotated term PhenoJury missed and at least one of the other two found",
    "all_wrong": "an annotated term none of the three found",
}

#: The three columns the cells are defined on, as ``(frame name, roster key)``. The frame names
#: are short because they end up in a CSV header and in prose. The roster keys are what the app
#: draws. Keeping both here is what lets a roster rename fail loudly instead of silently changing
#: what a cell means.
METHODS: tuple[tuple[str, str], ...] = (
    ("pj", "phenojury"),
    ("rag", "raghpo_70b"),
    ("apc", "autopcr_70b"),
)

# Checked at import, not trusted. A cell is a claim about three specific columns of
# ``app/compare_ui``. If one of them is renamed or dropped, a frame drawn here would keep its cell
# names and quietly mean something else. Failing here costs a traceback, which is the cheap outcome.
assert tuple(key for _short, key in METHODS) == roster.FRAME_METHODS, (
    "the frame's methods {} no longer match apps.compare_ui.roster.FRAME_METHODS {} -- the cells "
    "are defined on those columns, so fix one or the other before drawing".format(
        tuple(key for _short, key in METHODS), roster.FRAME_METHODS))


# ──────────────────────────────────────────────────────────────────────────────
# The cell rules
# ──────────────────────────────────────────────────────────────────────────────
def cell_of_term(found: dict) -> dict:
    """``{cell: "strict"|"relaxed"}`` for one annotated term, given ``{short name: bool}``.

    One term can qualify a document for at most one cell -- the four rules partition the eight
    possible verdict triples -- but the *rule* it qualifies under is not always the strict one, so
    the value is the tier, not ``True``. See property 3 in the module docstring.
    """
    pj, rag, apc = found["pj"], found["rag"], found["apc"]
    others = (rag, apc)
    if pj and all(others):
        return {"all_right": "strict"}
    if pj:
        return {"pj_only": "strict" if not any(others) else "relaxed"}
    if any(others):
        return {"pj_wrong": "strict" if all(others) else "relaxed"}
    return {"all_wrong": "strict"}


def witnesses(gold: dict, predicted: dict) -> list:
    """Every ``(doc_id, hpo_id, cell, tier, found)`` the cell rules recognise.

    *ground truth* and *predicted* are both in **resolved** ids. See property 2. The list is the frame's
    own evidence: a reader who opens one of these abstracts in the app should be able to find out
    which term put it on screen without rerunning anything.
    """
    out = []
    for doc_id in sorted(gold):
        for hpo_id in sorted(gold[doc_id]):
            found = {short: hpo_id in (predicted.get(key) or {}).get(doc_id, set())
                     for short, key in METHODS}
            for cell, tier in cell_of_term(found).items():
                out.append({"doc_id": doc_id, "hpo_id": hpo_id, "cell": cell, "tier": tier,
                            **{"found_" + short: int(found[short]) for short, _ in METHODS}})
    return out


def pools_from(witness_rows: list, eligible: set) -> tuple[dict, dict]:
    """``(pools, tiers)`` -- the documents each cell may draw from, strict tier first.

    ``pools[cell]`` is ordered: every document with a strict witness, then every document that
    only has a relaxed one. :func:`fill` draws *randomly* from a pool, so the ordering alone would
    not prefer the strict ones -- :func:`draw` slices the pool instead, and falls back to the full
    one only for a cell the strict tier cannot fill.
    """
    strict: dict = {cell: [] for cell in CELL_ORDER}
    relaxed: dict = {cell: [] for cell in CELL_ORDER}
    for row in witness_rows:
        if row["doc_id"] not in eligible:
            continue
        bucket = strict if row["tier"] == "strict" else relaxed
        if row["doc_id"] not in bucket[row["cell"]]:
            bucket[row["cell"]].append(row["doc_id"])

    pools, tiers = {}, {}
    for cell in CELL_ORDER:
        only_relaxed = [d for d in relaxed[cell] if d not in strict[cell]]
        pools[cell] = sorted(strict[cell]) + sorted(only_relaxed)
        tiers[cell] = {"strict": sorted(strict[cell]), "relaxed": sorted(only_relaxed)}
    return pools, tiers


def draw(pools: dict, tiers: dict, seed: int) -> tuple[dict, list, dict]:
    """``(picks, trace, tier_of)`` -- the draw, strict tier first.

    Two passes over the same greedy fill. The first offers each cell only its strict pool, which
    is the rule the cell's *name* claims. The second re-runs the whole fill with the full pools to
    top up whatever the first left short. Re-running, not patching keeps one code path for
    "distinct documents, most-constrained cell first" -- the property that makes the twenty a
    sample, not four overlapping lists.
    """
    strict_pools = {cell: tiers[cell]["strict"] for cell in pools}
    picks, trace = fill(strict_pools, CELL_SIZES, seed)
    tier_of = {doc: "strict" for ids in picks.values() for doc in ids}

    short = {cell: CELL_SIZES[cell] - len(ids) for cell, ids in picks.items()}
    if any(short.values()):
        taken = set(tier_of)
        top_up_pools = {cell: [d for d in pools[cell] if d not in taken] for cell in pools}
        extra, extra_trace = fill(top_up_pools, short, seed + 1)
        for cell, ids in extra.items():
            picks[cell] = sorted(picks[cell] + ids)
            for doc in ids:
                tier_of[doc] = "relaxed"
        trace = trace + [dict(row, note=(row.get("note", "") + " [relaxed pass]").strip())
                         for row in extra_trace]
    return picks, trace, tier_of


# ──────────────────────────────────────────────────────────────────────────────
# scoring, for the pools table
# ──────────────────────────────────────────────────────────────────────────────
def score(predicted: set, gold: set) -> dict:
    """Per-document counts and rates. Rates are ``None`` where they are undefined, never zero."""
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    elif tp + fn == 0:
        f1 = None
    else:
        f1 = 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def scored_rows(gold: dict, predicted: dict, witness_rows: list) -> dict:
    """``{doc_id: row}`` for ``pools.csv``: ground truth size, per-method counts, per-cell witnesses."""
    per_cell: dict = {}
    for row in witness_rows:
        bucket = per_cell.setdefault(row["doc_id"], {cell: 0 for cell in CELL_ORDER})
        bucket[row["cell"]] += 1

    out = {}
    for doc_id in sorted(gold):
        row = {"n_gold": len(gold[doc_id])}
        for short, key in METHODS:
            got = (predicted.get(key) or {}).get(doc_id, set())
            row["n_pred_" + short] = len(got)
            for field, value in score(got, gold[doc_id]).items():
                row["{}_{}".format(short, field)] = value
        row.update({"w_" + cell: (per_cell.get(doc_id) or {}).get(cell, 0)
                    for cell in CELL_ORDER})
        out[doc_id] = row
    return out


# ──────────────────────────────────────────────────────────────────────────────
# loading
# ──────────────────────────────────────────────────────────────────────────────
def resolver(use_ontology: bool):
    """``(fn, resolved)`` -- the code normaliser the builder uses, or identity with a warning."""
    if not use_ontology:
        return (lambda code: str(code)), False
    try:
        view = sources.get_view()
    except Exception as exc:                                  # pragma: no cover - env-dependent
        logger.warning("no ontology (%s); falling back to raw string comparison. The cells may "
                       "then disagree with the app, which always resolves.", exc)
        return (lambda code: str(code)), False
    return (lambda code: str(view.resolve(str(code)) or code)), True


def load_gold(raghpo_dir: str, resolve) -> dict:
    """``{doc_id: {resolved hpo_id}}`` from RAG-HPO's own annotation of their 114 documents."""
    annotations = gold_mod.read_raghpo_annotations(
        os.path.join(raghpo_dir, gold_mod.RAGHPO_ANNOTATIONS))
    return {doc: {resolve(code) for code in codes} for doc, codes in annotations.items()}


def load_predictions(results_dir: str, resolve) -> tuple[dict, dict]:
    """``({roster key: {doc_id: {resolved}}}, {roster key: path})`` for the three frame methods.

    Paths come from ``apps.compare_ui.roster`` and ``sources.Paths``, not from flags, so the frame
    is drawn from the same files the app then explains. A method whose prediction set is missing
    is a hard stop: three cells out of four are defined by its absence from a set, and "not
    predicted because the file is not there" is not a finding.
    """
    paths = sources.Paths(output_base=results_dir, cohort=cohorts.GSC.key)
    predicted, used = {}, {}
    for _short, key in METHODS:
        method = roster.get(key)
        declared = method.predictions_for(cohorts.GSC)
        path = (os.path.join(results_dir, declared) if declared
                else paths.cohort_exp(method.exp_id, method.variant + "_predictions.jsonl"))
        sets = base.load_prediction_sets(path)
        if not sets:
            raise SystemExit("ERROR: no predictions for {} at {}".format(key, path))
        predicted[key] = {doc: {resolve(code) for code in codes} for doc, codes in sets.items()}
        used[key] = path
    return predicted, used


# ──────────────────────────────────────────────────────────────────────────────
# The frame
# ──────────────────────────────────────────────────────────────────────────────
def build(*, gold: dict, predicted: dict, eligible, seed: int, resolved: bool) -> dict:
    """Everything the frame records, computed once. Pure -- see :func:`selftest`."""
    eligible = set(eligible) & set(gold)
    witness_rows = witnesses(gold, predicted)
    pools, tiers = pools_from(witness_rows, eligible)
    picks, trace, tier_of = draw(pools, tiers, seed)
    selected = sorted({doc for ids in picks.values() for doc in ids})
    return {
        "params": {
            "seed": seed,
            "cell_sizes": dict(CELL_SIZES),
            "methods": {short: key for short, key in METHODS},
            "rules": dict(CELL_HELP),
            "resolved": resolved,
            "n_documents_considered": len(eligible),
        },
        "cell_sizes": dict(CELL_SIZES),
        "pools": pools,
        "tiers": tiers,
        "picks": {cell: picks.get(cell, []) for cell in CELL_ORDER},
        "selected": selected,
        "tier_of": tier_of,
        "trace": trace,
        "witnesses": witness_rows,
        "scored": scored_rows(gold, predicted, witness_rows),
    }


def render(result: dict, inputs: dict) -> str:
    """``FRAME.md`` -- the pre-registration a human reads, in the order it has to be read in."""
    picks = result["picks"]
    lines = [
        "# GSC+ comparison frame, 20 abstracts, four cells",
        "",
        "Drawn by `apps/compare_ui/select_gsc_documents.py` on {}, seed {}.".format(
            date.today().isoformat(), result["params"]["seed"]),
        "",
        "**This is a purposive sample.** Every cell is defined on whether a method got an annotated term "
        "right, which is the quantity anyone would want to measure over it. No rate computed here "
        "means anything; the quotable numbers are "
        "`comparison/tables/t1_overall.csv`, row `gsc_raghpo_ann`.",
        "",
        "## Cells",
        "",
        "| cell | n | drawn | pool (strict) | pool (+relaxed) | a document qualifies when… |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cell in CELL_ORDER:
        lines.append("| `{}` | {} | {} | {} | {} | {} |".format(
            cell, CELL_SIZES[cell], len(picks.get(cell) or ()),
            len(result["tiers"][cell]["strict"]),
            len(result["tiers"][cell]["strict"]) + len(result["tiers"][cell]["relaxed"]),
            CELL_HELP[cell]))

    lines += [
        "",
        "“Right” means the method predicted a term the ground truth carries. False positives are a "
        "different failure and define no cell here; they are in `pools.csv` all the same.",
        "",
        "## The draw",
        "",
        "| cell | document | rule it came in under | witness terms |",
        "| --- | --- | --- | --- |",
    ]
    by_doc: dict = {}
    for row in result["witnesses"]:
        by_doc.setdefault((row["cell"], row["doc_id"]), []).append(row["hpo_id"])
    for cell in CELL_ORDER:
        for doc_id in picks.get(cell) or ():
            terms = by_doc.get((cell, doc_id)) or []
            lines.append("| `{}` | {} | {} | {} |".format(
                cell, doc_id, result["tier_of"].get(doc_id, "strict"),
                ", ".join(terms[:6]) + (" …" if len(terms) > 6 else "")))

    lines += [
        "",
        "## Inputs",
        "",
        "| input | sha256 |",
        "| --- | --- |",
    ]
    for name, value in sorted(inputs.items()):
        lines.append("| `{}` | `{}` |".format(name, value))

    if not result["params"]["resolved"]:
        lines += [
            "",
            "> **Drawn without the ontology.** Ground truth and predicted codes were compared as raw "
            "strings, so a phenotype spelled with an alt id on one side counts as a miss here and "
            "as a hit in `apps/compare_ui`, which always resolves. Redraw with the ontology "
            "available before quoting a cell.",
        ]
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: str, result: dict, inputs: dict) -> list:
    """Write the GSC+ selection frame (JSON and CSV) to *out_dir* and return the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated": date.today().isoformat(),
        "cohort": cohorts.GSC.key,
        "scored_cohort": cohorts.GSC.scored_cohort,
        "inputs": inputs,
        "params": result["params"],
        "cell_sizes": result["cell_sizes"],
        "cell_help": dict(CELL_HELP),
        "pools": result["pools"],
        "tiers": result["tiers"],
        "picks": result["picks"],
        "selected": result["selected"],
        "tier_of": result["tier_of"],
        "trace": result["trace"],
    }
    # The frame carries ids, codes and counts and no document text. GSC+ may leave the cluster, so
    # this is not a confidentiality check -- it is the same check the HCY frame runs, kept shared
    # so neither frame grows its own idea of what a leak is.
    assert_no_text(payload, {})
    written = []

    (out / "frame.json").write_text(json.dumps(payload, indent=2, sort_keys=True),
                                    encoding="utf-8")
    written.append(str(out / "frame.json"))

    (out / "FRAME.md").write_text(render(result, inputs), encoding="utf-8")
    written.append(str(out / "FRAME.md"))

    cell_of = {doc: cell for cell, ids in result["picks"].items() for doc in ids}
    sample = next(iter(result["scored"].values()), {})
    fields = (["patient_id", "selected", "cell", "tier"]
              + [key for key in sample if key != "patient_id"])
    with open(out / "pools.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for doc_id, row in sorted(result["scored"].items()):
            writer.writerow({"patient_id": doc_id, "selected": int(doc_id in cell_of),
                             "cell": cell_of.get(doc_id, ""),
                             "tier": result["tier_of"].get(doc_id, ""), **row})
    written.append(str(out / "pools.csv"))

    wfields = ["doc_id", "hpo_id", "cell", "tier", "selected"] + [
        "found_" + short for short, _ in METHODS]
    with open(out / "witnesses.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=wfields)
        writer.writeheader()
        for row in result["witnesses"]:
            writer.writerow(dict(row, selected=int(cell_of.get(row["doc_id"]) == row["cell"])))
    written.append(str(out / "witnesses.csv"))
    return written


# ──────────────────────────────────────────────────────────────────────────────
# selftest
# ──────────────────────────────────────────────────────────────────────────────
def selftest() -> int:
    """Exercise the cell rules, the tier preference and the greedy fill on synthetic data."""
    failures = []

    def check(name, condition, detail=""):
        if not condition:
            failures.append("{}: {}".format(name, detail) if detail else name)

    # -- the four rules partition the eight verdict triples, and each names the right cell
    seen = {}
    for pj in (0, 1):
        for rag in (0, 1):
            for apc in (0, 1):
                got = cell_of_term({"pj": bool(pj), "rag": bool(rag), "apc": bool(apc)})
                check("one cell per triple ({}{}{})".format(pj, rag, apc), len(got) == 1, got)
                seen[(pj, rag, apc)] = next(iter(got))
    check("all three found it", seen[(1, 1, 1)] == "all_right", seen[(1, 1, 1)])
    check("only PhenoJury", seen[(1, 0, 0)] == "pj_only", seen[(1, 0, 0)])
    check("PhenoJury and one other", seen[(1, 1, 0)] == "pj_only", seen[(1, 1, 0)])
    check("PhenoJury missed it, both others found it",
          seen[(0, 1, 1)] == "pj_wrong", seen[(0, 1, 1)])
    check("PhenoJury missed it, one other found it",
          seen[(0, 0, 1)] == "pj_wrong", seen[(0, 0, 1)])
    check("nobody found it", seen[(0, 0, 0)] == "all_wrong", seen[(0, 0, 0)])
    check("the strict tier is the unanimous one",
          cell_of_term({"pj": True, "rag": False, "apc": False})["pj_only"] == "strict")
    check("and a partial disagreement is relaxed",
          cell_of_term({"pj": True, "rag": True, "apc": False})["pj_only"] == "relaxed")

    # -- a synthetic cohort: 8 documents per cell, all strict, so every cell fills
    gold, predicted = {}, {key: {} for _s, key in METHODS}
    plan = {"all_right": (1, 1, 1), "pj_only": (1, 0, 0),
            "pj_wrong": (0, 1, 1), "all_wrong": (0, 0, 0)}
    for cell, (pj, rag, apc) in plan.items():
        for i in range(8):
            doc = "{}_{}".format(cell, i)
            code = "HP:000000{}".format(i % 9)
            gold[doc] = {code}
            for short, key in METHODS:
                got = {"pj": pj, "rag": rag, "apc": apc}[short]
                predicted[key][doc] = {code} if got else set()

    result = build(gold=gold, predicted=predicted, eligible=list(gold), seed=7, resolved=True)
    check("twenty documents drawn", len(result["selected"]) == 20, len(result["selected"]))
    check("and they are distinct",
          len(result["selected"]) == len({d for ids in result["picks"].values() for d in ids}))
    for cell in CELL_ORDER:
        check("cell {} is full".format(cell), len(result["picks"][cell]) == CELL_SIZES[cell],
              len(result["picks"][cell]))
        check("cell {} drew only its own pool".format(cell),
              set(result["picks"][cell]) <= set(result["pools"][cell]))
    check("every pick is strict here",
          set(result["tier_of"].values()) == {"strict"}, set(result["tier_of"].values()))

    # -- the same seed draws the same twenty
    again = build(gold=gold, predicted=predicted, eligible=list(gold), seed=7, resolved=True)
    check("the draw is a function of the seed", again["selected"] == result["selected"])

    # -- a cell with no strict candidates falls back and says so
    gold2 = {"d0": {"HP:0000001"}}
    predicted2 = {key: {"d0": {"HP:0000001"} if short == "pj" or short == "rag" else set()}
                  for short, key in METHODS}
    small = build(gold=gold2, predicted=predicted2, eligible=["d0"], seed=1, resolved=True)
    check("a relaxed-only document still reaches pj_only",
          small["picks"]["pj_only"] == ["d0"], small["picks"])
    check("and is marked relaxed", small["tier_of"].get("d0") == "relaxed", small["tier_of"])
    check("an unfillable cell is left short, not raised",
          small["picks"]["all_right"] == [], small["picks"]["all_right"])

    # -- the rendered frame names every cell and never carries document text
    text = render(small, {"gold": "abc"})
    for cell in CELL_ORDER:
        check("FRAME.md names {}".format(cell), cell in text)
    assert_no_text(text, {"d0": "a synthetic abstract about phenotypes and inheritance"})

    if failures:
        print("\nselect_gsc_compare_cells: {} check(s) FAILED".format(len(failures)))
        for failure in failures:
            print("  - " + failure)
        return 1
    print("select_gsc_compare_cells: all checks passed")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    """Parse the command-line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=sources.DEFAULT_OUTPUT_BASE,
                        help="the directory holding the experiment output trees")
    parser.add_argument("--raghpo-dir", default=str(_REPO / sources.DEFAULT_RAGHPO_DIR),
                        help="RAG-HPO's vendored 114-document subset and its own annotation")
    parser.add_argument("--out", default="",
                        help="where to write the frame; default <results-dir>/gsc_compare_frame")
    parser.add_argument("--seed", type=int, default=20260919,
                        help="the draw's seed; recorded in the frame")
    parser.add_argument("--no-ontology", action="store_true",
                        help="compare codes as raw strings. Recorded in the frame, and the app "
                             "may then disagree with a cell -- see the module docstring.")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Select the GSC+ documents shown in the Compare UI. Returns the exit status."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s")

    if args.selftest:
        return selftest()

    resolve, resolved = resolver(not args.no_ontology)
    gold = load_gold(args.raghpo_dir, resolve)
    if not gold:
        print("ERROR: no annotations at " + os.path.join(args.raghpo_dir, "annotations.csv"),
              file=sys.stderr)
        return 2
    predicted, used = load_predictions(args.results_dir, resolve)

    source = gold_mod.RagHpoGold(args.raghpo_dir)
    eligible = [doc for doc in source.document_ids() if doc in gold]
    logger.info("%d documents with ground truth, %d eligible", len(gold), len(eligible))

    result = build(gold=gold, predicted=predicted, eligible=eligible, seed=args.seed,
                   resolved=resolved)

    inputs = {"raghpo_annotations": digest(os.path.join(args.raghpo_dir, "annotations.csv"))}
    inputs.update({key: digest(path) for key, path in used.items()})

    out_dir = args.out or os.path.join(args.results_dir, "gsc_compare_frame")
    written = write_outputs(out_dir, result, inputs)

    for cell in CELL_ORDER:
        picked = result["picks"][cell]
        logger.info("%-10s %d/%d drawn from a pool of %d (%d strict)", cell, len(picked),
                    CELL_SIZES[cell], len(result["pools"][cell]),
                    len(result["tiers"][cell]["strict"]))
        if len(picked) < CELL_SIZES[cell]:
            logger.warning("  %s is SHORT -- the pool ran out; the frame records it", cell)
    for path in written:
        logger.info("wrote %s", path)
    print("\nNext:\n  python apps/compare_ui/build.py --cohort gsc --output-base {} "
          "--frame {}".format(args.results_dir, out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
