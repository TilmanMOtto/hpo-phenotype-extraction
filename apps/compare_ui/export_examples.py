"""Hand-picked (document, segment) examples out of the bundles, as the rows of a thesis table.

The appendix's side-by-side tables (``tab_ch6_qualitative_{hcy,gsc}``) show, for a few sentences
chosen during manual error analysis, what each of the five main methods did against the ground truth.
Everything they need is already in a bundle, so this reads bundles and nothing else. Like the
builder, it runs where the bundles are, which for HCY is the cluster.

The selection is a tracked CSV of ids only (``figures/thesis_figures_scripts/qualitative_examples.csv``):
``block, cohort, doc_id, segment_idx, anchor``. An HCY row names its segment by index. A GSC+ row
may name it by ``anchor``, a verbatim piece of the published sentence, which must occur in
one segment. Given both, they must agree.

One row per term per example
----------------------------
The terms of an example are the annotated terms placed on its segment, plus every term a method
**predicted and localised to that segment**. A method's own location is
``reasons[code].segment_idx``, never ``marks[].segment_idx``: a mark for a tp or an fn is drawn on
the ground truth's segment (``methods/base.py``, ``marks_from``), so reading marks would place every found
term where the ground truth is and hide the disagreement this table exists to show. PhenoJury
names sentences per juror, and predicts a term when enough of them agree in one sentence, so a
term is localised to the segment only if the jurors naming it there reach that vote
(``reasons[code].by_sentence``. The threshold is :func:`jury_threshold`). One juror's mention is
outvoted noise, not the method's output.

A cell carries the method's **document-level** outcome, the one the comparison scores, refined by where
its own evidence sits:

``tp``            found, and the method's evidence is this segment
``tp_elsewhere``  found, but the method's evidence is another segment
``tp_unplaced``   found, and the method recorded no location at all
``fn``            an annotated term of this segment the method did not predict
``fp``            localised here, and not annotated on this segment (see below)
``na``            the method's column could not be built for this document
(empty)           the method has nothing to say about this term in this segment

A term that is ground truth in the document but not on this segment appears only if some method
localised it here, and which of two rows it is decides its mark:

``gold_elsewhere``  the ground truth evidence locations it on **another** segment. The curated ground truth evidence locations each
                    term on one sentence, so a term a method predicts here may well be mentioned
                    here too: the cell is ``tp`` and the row says where the ground truth put it. (A lone
                    juror's "Microcephaly" never reaches this row -- see :func:`jury_threshold`.)
``gold_unplaced``   the ground truth gives it **no** segment (GSC+ terms only RAG-HPO's annotators wrote
                    down, with a description found nowhere verbatim). Nothing says it is not here,
                    so the cell is ``tp``.

In both, the row's misses belong elsewhere, so they are left empty rather than printed as ``fn``.

Nothing about a term's text is printed to stdout -- only ids and counts. The HCY CSV carries the
segment verbatim because that is what the thesis prints. It is written, never echoed.

    sbatch slurm/compare_ui_bundles.sbatch           # on LeoMed: build the bundles, export
    python apps/compare_ui/export_examples.py --cohorts gsc \
        --bundle-root output/compare_ui_bundles/examples --out output/compare_ui_bundles/qualitative
    python apps/compare_ui/export_examples.py --selftest

The example bundles live in their own root (``output/compare_ui_bundles/examples/<cohort>``), not in
the app's: ``build.py --reports`` rewrites the ``index.json`` of the directory it writes to.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _path in (_REPO,):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from apps.compare_ui import bundles, roster, sources  # noqa: E402

DEFAULT_SPEC = os.path.join(_REPO, "figures", "thesis_figures_scripts", "qualitative_examples.csv")

#: The directory under the bundle root the exported CSVs go to, beside the per-cohort bundles.
EXPORT_SUBDIR = "qualitative"

SPEC_FIELDS = ("block", "cohort", "doc_id", "segment_idx", "anchor")

#: Where the comparison's PhenoJury inputs record the full pool's per-fold choices, under output_base.
JURY_MAIN_RESULT = "comparison_inputs/phenojury_protocol/{cohort}/tables/s7_headline.csv"
ROW_KINDS = ("gold", "gold_unplaced", "gold_elsewhere", "predicted")
CELLS = ("tp", "tp_elsewhere", "tp_unplaced", "fn", "fp", "na", "")
FIELDS = ["example_no", "block", "cohort", "doc_id", "segment_idx", "n_segments", "sentence",
          "hpo_id", "label", "row_kind", "trigger"] + list(roster.ORDER)


class SpecError(ValueError):
    """A selection row that cannot be resolved to one segment of its bundle."""


def export_path(out_dir: str, cohort: str) -> str:
    """Path of the exported examples CSV of *cohort*."""
    return os.path.join(out_dir, "examples_{}.csv".format(cohort))


def provenance_path(out_dir: str, cohort: str) -> str:
    """Path of the provenance JSON of the exported examples of *cohort*."""
    return os.path.join(out_dir, "examples_{}.json".format(cohort))


# -- the spec ----------------------------------------------------------------

def read_spec(path: str) -> list:
    """Rows of the examples specification CSV. Raises ``SpecError`` when a column is missing."""
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [f for f in SPEC_FIELDS if f not in (reader.fieldnames or ())]
        if missing:
            raise SpecError("{} lacks column(s) {}".format(path, missing))
        rows = []
        for n, row in enumerate(reader, start=2):
            row = {k: (row.get(k) or "").strip() for k in SPEC_FIELDS}
            if not row["doc_id"]:
                continue
            row["line"] = n
            rows.append(row)
    return rows


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def resolve_segment(bundle: dict, spec: dict) -> int:
    """The segment index a spec row names. Raises :class:`SpecError` unless it names one."""
    segments = bundle["segments"]
    where = "{} line {} ({})".format(spec.get("block") or "spec", spec.get("line", "?"),
                                     spec["doc_id"])
    given = None
    if spec.get("segment_idx"):
        try:
            given = int(spec["segment_idx"])
        except ValueError:
            raise SpecError("{}: segment_idx {!r} is not an integer".format(
                where, spec["segment_idx"])) from None
        if not 0 <= given < len(segments):
            raise SpecError("{}: segment_idx {} but the document has {} segments".format(
                where, given, len(segments)))

    anchor = _squash(spec.get("anchor"))
    if not anchor:
        if given is None:
            raise SpecError("{}: neither segment_idx nor anchor is set".format(where))
        return given

    hits = [seg["idx"] for seg in segments if anchor in _squash(seg["text"])]
    if not hits:
        raise SpecError("{}: the anchor occurs in no segment -- the segmentation may split it; "
                        "shorten it to a piece inside one sentence".format(where))
    if len(hits) > 1:
        raise SpecError("{}: the anchor occurs in segments {}; lengthen it".format(where, hits))
    if given is not None and given != hits[0]:
        raise SpecError("{}: segment_idx {} but the anchor is in segment {}".format(
            where, given, hits[0]))
    return hits[0]


# -- one example -------------------------------------------------------------

def main_result_k(path: str):
    """The largest vote threshold *k* any fold chose for the full pool, from the PhenoJury protocol's
    ``s7_headline.csv`` (its ``selection_modes`` column, ``[[[rule, unit, k], n_folds], ...]``),
    or None when the file or the row is absent.

    That maximum is the one threshold that settles placement under **every** configuration the
    folds chose: *k* jurors naming a term in the sentence itself passes a per-sentence vote at any
    *k'* <= *k*, and a +/-1-sentence window vote too, since the window contains the sentence.
    """
    import ast

    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("config") != "full_pool":
                continue
            try:
                modes = ast.literal_eval(row.get("selection_modes") or "[]")
                return max(int(mode[0][2]) for mode in modes) if modes else None
            except (ValueError, SyntaxError, TypeError, IndexError):
                return None
    return None


def jury_threshold(bundle: dict, cohort_k=None) -> dict:
    """How many jurors must name a term in one sentence for it to be PhenoJury's prediction there.

    ``{"t", "source", "k_lo", "k_hi"}``. PhenoJury predicts a term when at least *k* jurors name it
    in one sentence (``unit="segment"``), so a term one juror named is not PhenoJury's -- a single
    juror answering "Microcephaly" to every sentence is outvoted, and drawing it as the method's
    output would put the failure the vote removes back into the table.

    *k* comes from the fold's configuration when the bundle knows it. Otherwise from *cohort_k*,
    :func:`headline_k`'s threshold that holds under every configuration the folds chose. Only
    when neither is available is it **inferred from this document's own prediction set** (which
    assumes a per-sentence vote): every predicted term reached *k* in some
    sentence, so ``k <= k_hi`` (the smallest such peak), and no unpredicted term did, so
    ``k >= k_lo`` (one more than the largest unpredicted peak). Any *k* in between reproduces the
    prediction set; ``t`` is ``k_lo``. If no *k* reproduces it -- the detections and the
    prediction set come from different runs -- ``t`` is None and no PhenoJury term is placed.
    """
    pj = (bundle.get("methods") or {}).get("phenojury") or {}
    reasons = pj.get("reasons") or {}
    predicted = set(pj.get("predicted") or ())
    known = {(r.get("k"), r.get("unit")) for r in reasons.values() if r.get("config_known")}
    if len(known) == 1:
        k, unit = known.pop()
        if k and unit in ("segment", "window"):
            # A window vote can pass on fewer jurors in this sentence, so for a window only
            # The certain case -- k here -- is placed. The rest is found but not located.
            return {"t": int(k), "source": "fold configuration ({})".format(unit),
                    "k_lo": int(k), "k_hi": int(k)}
        if k:
            return {"t": None, "source": "votes over a {}, not a sentence".format(unit),
                    "k_lo": None, "k_hi": None}
    if cohort_k:
        return {"t": int(cohort_k), "source": "largest k the full pool's folds chose",
                "k_lo": None, "k_hi": None}

    def peak(reason):
        return max((len(m) for m in (reason.get("by_sentence") or {}).values()), default=0)

    k_lo = 1 + max((peak(r) for c, r in reasons.items() if c not in predicted), default=0)
    peaks = [peak(r) for c, r in reasons.items() if c in predicted]
    k_hi = min(peaks) if peaks else None
    if k_hi is not None and k_lo > k_hi:
        return {"t": None, "source": "inconsistent: no threshold reproduces the prediction set",
                "k_lo": k_lo, "k_hi": k_hi}
    return {"t": k_lo, "source": "inferred from the prediction set", "k_lo": k_lo, "k_hi": k_hi}


def _jurors(reason: dict, seg) -> int:
    return len((reason.get("by_sentence") or {}).get(str(seg)) or ())


def _localised_here(key: str, reason: dict, seg: int, t=None) -> bool:
    if not reason:
        return False
    if key == "phenojury":
        return t is not None and _jurors(reason, seg) >= t
    return reason.get("segment_idx") == seg


def _own_location(key: str, reason: dict, t=None):
    """The segments the method's own evidence names -- empty when it recorded none."""
    if not reason:
        return set()
    if key == "phenojury":
        if t is None:
            return set()
        return {int(s) for s, m in (reason.get("by_sentence") or {}).items() if len(m) >= t}
    idx = reason.get("segment_idx")
    return set() if idx is None else {int(idx)}


def example_rows(bundle: dict, seg: int, jury=None) -> list:
    """The term rows of one (document, segment) example, ground truth first, then by HPO id.

    *jury* is :func:`jury_threshold`'s result, computed here when not given.
    """
    t = (jury or jury_threshold(bundle))["t"]
    gold_here = {}
    gold_any = {}
    for g in bundle["gold"]:
        # A placed row wins over an unplaced one: "annotated on another segment" is the stronger
        # statement, and the bundle lists placed rows first anyway.
        if g["hpo_id"] not in gold_any or gold_any[g["hpo_id"]].get("segment_idx") is None:
            gold_any[g["hpo_id"]] = g
        if g.get("segment_idx") == seg:
            gold_here.setdefault(g["hpo_id"], g)

    methods = bundle["methods"]
    predicted = {k: set((methods.get(k) or {}).get("predicted") or ()) for k in roster.ORDER}
    reasons = {k: (methods.get(k) or {}).get("reasons") or {} for k in roster.ORDER}
    available = {k: (methods.get(k) or {}).get("status") == "ok" for k in roster.ORDER}

    here = {k: {c for c in predicted[k] if _localised_here(k, reasons[k].get(c), seg, t)}
            for k in roster.ORDER}

    codes = set(gold_here)
    for k in roster.ORDER:
        codes |= here[k]

    rows = []
    for code in codes:
        if code in gold_here:
            kind = "gold"
        elif code in gold_any:
            placed = gold_any[code].get("segment_idx") is not None
            kind = "gold_elsewhere" if placed else "gold_unplaced"
        else:
            kind = "predicted"
        cells = {}
        for k in roster.ORDER:
            if not available[k]:
                cells[k] = "na"
                continue
            found = code in predicted[k]
            if kind == "gold":
                if not found:
                    cells[k] = "fn"
                elif code in here[k]:
                    cells[k] = "tp"
                elif _own_location(k, reasons[k].get(code), t):
                    cells[k] = "tp_elsewhere"
                else:
                    cells[k] = "tp_unplaced"
            elif kind in ("gold_elsewhere", "gold_unplaced"):
                cells[k] = "tp" if code in here[k] else ""
            else:
                cells[k] = "fp" if code in here[k] else ""
        term = (bundle.get("terms") or {}).get(code) or {}
        gold_row = gold_here.get(code) or {}
        span = gold_row.get("span")
        rows.append({"hpo_id": code, "label": term.get("label") or code, "row_kind": kind,
                     "trigger": gold_row.get("trigger") or "", **cells,
                     "_at": span[0] if span else float("inf")})

    # Ground truth in reading order (where its trigger sits in the sentence), the rest alphabetically.
    order = {k: i for i, k in enumerate(ROW_KINDS)}
    rows.sort(key=lambda r: (order[r["row_kind"]], r.pop("_at"), r["label"].lower(), r["hpo_id"]))
    return rows


def _provenance(bundle: dict) -> dict:
    return {
        "schema": bundle.get("schema"),
        "built_at": bundle.get("built_at"),
        "gold_dataset": bundle.get("gold_dataset"),
        "cell": bundle.get("cell"),
        "methods": {k: {"status": (bundle["methods"].get(k) or {}).get("status"),
                        "note": (bundle["methods"].get(k) or {}).get("note"),
                        "evidence_note": (bundle["methods"].get(k) or {}).get("evidence_note")}
                    for k in roster.ORDER},
    }


# -- one cohort --------------------------------------------------------------

def unlocated(spec: dict) -> bool:
    """A spec row that names no segment yet -- neither an index nor an evidence location."""
    return not spec.get("segment_idx") and not (spec.get("anchor") or "").strip()


def export_cohort(spec_rows: list, bundle_dir: str, cohort: str, skip_unlocated: bool = False,
                  cohort_k=None):
    """``(rows, provenance, summary)`` for the spec rows of *cohort*. Raises on any bad row.

    *skip_unlocated* leaves out rows that name no segment yet instead of raising, so the located
    ones can be tabled while the rest are looked up. Only that case is skipped: a row that names a
    segment wrongly still raises.
    """
    rows, prov, summary = [], {"cohort": cohort, "documents": {}, "skipped_lines": []}, []
    wanted = [s for s in spec_rows if s["cohort"] == cohort]
    if skip_unlocated:
        prov["skipped_lines"] = [s.get("line") for s in wanted if unlocated(s)]
        wanted = [s for s in wanted if not unlocated(s)]
    missing = sorted({s["doc_id"] for s in wanted
                      if bundles.read(bundle_dir, s["doc_id"]) is None})
    if missing:
        raise SpecError(
            "no readable bundle in {} for {}. Build them with:\n  python apps/compare_ui/build.py "
            "--cohort {} --reports {}".format(bundle_dir, ", ".join(missing), cohort,
                                              ",".join(missing)))
    for n, spec in enumerate(wanted, start=1):
        bundle = bundles.read(bundle_dir, spec["doc_id"])
        seg = resolve_segment(bundle, spec)
        jury = jury_threshold(bundle, cohort_k)
        term_rows = example_rows(bundle, seg, jury)
        n_gold = sum(r["row_kind"] == "gold" for r in term_rows)
        # Juror counts in this segment the threshold does not settle: k_lo <= count < k_hi.
        pj = (bundle["methods"].get("phenojury") or {}).get("reasons") or {}
        unsettled = sum(1 for r in pj.values()
                        if jury["k_hi"] and jury["k_lo"] and
                        jury["k_lo"] <= _jurors(r, seg) < jury["k_hi"])
        summary.append((n, spec["doc_id"], seg, n_gold, len(term_rows), jury, unsettled))
        prov["documents"].setdefault(spec["doc_id"], _provenance(bundle))["jury_threshold"] = jury
        base = {"example_no": n, "block": spec["block"], "cohort": cohort,
                "doc_id": spec["doc_id"], "segment_idx": seg,
                "n_segments": len(bundle["segments"]),
                "sentence": bundle["segments"][seg]["text"]}
        if not term_rows:
            # An example with nothing on it still gets its sentence printed -- an empty row set
            # is a finding about the segment, not a reason to drop it silently.
            rows.append(dict(base, hpo_id="", label="", row_kind="", trigger="",
                             **{k: "" for k in roster.ORDER}))
        for r in term_rows:
            rows.append(dict(base, **r))
    return rows, prov, summary


def write_csv(path: str, rows: list) -> None:
    """Write *rows* (dicts) to a CSV at *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


# -- CLI ---------------------------------------------------------------------

def parse_args(argv=None):
    """Parse the command-line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", default=DEFAULT_SPEC)
    parser.add_argument("--bundle-root", default=sources.DEFAULT_BUNDLE_ROOT,
                        help="the parent of the per-cohort bundle directories")
    parser.add_argument("--out", default="",
                        help="where the CSVs go; default <bundle-root>/" + EXPORT_SUBDIR)
    parser.add_argument("--cohorts", default="",
                        help="comma-separated cohorts to export; default every cohort in the spec")
    parser.add_argument("--output-base", default=sources.DEFAULT_OUTPUT_BASE,
                        help="where comparison_inputs/ lives, for PhenoJury's s7_headline.csv")
    parser.add_argument("--skip-unlocated", action="store_true",
                        help="leave out spec rows with neither segment_idx nor anchor, and say "
                             "which, instead of failing the cohort")
    parser.add_argument("--selftest", action="store_true",
                        help="export from the synthetic fixture and check the result")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Export the specified examples from the bundles. Returns the exit status."""
    args = parse_args(argv)
    if args.selftest:
        return _selftest()

    spec_rows = read_spec(args.spec)
    cohorts = ([c.strip() for c in args.cohorts.split(",") if c.strip()]
               or sorted({s["cohort"] for s in spec_rows}))
    out_dir = args.out or os.path.join(args.bundle_root, EXPORT_SUBDIR)
    status = 0
    for cohort in cohorts:
        try:
            headline = os.path.join(args.output_base, JURY_MAIN_RESULT.format(cohort=cohort))
            rows, prov, summary = export_cohort(
                spec_rows, sources.bundles_dir(args.bundle_root, cohort), cohort,
                skip_unlocated=args.skip_unlocated, cohort_k=main_result_k(headline))
        except SpecError as exc:
            print("ERROR [{}]: {}".format(cohort, exc), file=sys.stderr)
            status = 2
            continue
        write_csv(export_path(out_dir, cohort), rows)
        with open(provenance_path(out_dir, cohort), "w", encoding="utf-8") as handle:
            json.dump(prov, handle, indent=2, sort_keys=True)
        print("[{}] {} example(s) -> {}".format(cohort, len(summary), export_path(out_dir, cohort)))
        for line in prov["skipped_lines"]:
            print("  SKIPPED spec line {}: no segment_idx or anchor yet".format(line))
        for n, doc, seg, n_gold, n_rows, jury, unsettled in summary:
            print("  #{:<2} {:<10} segment {:<3} {} annotated terms, {} term row(s); PhenoJury >= {} "
                  "jurors ({}){}{}".format(
                      n, doc, seg, n_gold, n_rows, jury["t"], jury["source"],
                      "  WARNING: {} juror count(s) here fall in the inferred range [{}, {})"
                      .format(unsettled, jury["k_lo"], jury["k_hi"]) if unsettled else "",
                      "" if n_rows else "  WARNING: nothing on it"))
    return status


def _selftest() -> int:
    """Export the fixture's GSC+ cohort and check the rules in the module docstring."""
    import tempfile

    from apps.compare_ui import fixture

    with tempfile.TemporaryDirectory() as root:
        written = fixture.write_gsc(root)
        _ctx, built = fixture.build_gsc(written)
        built.pop("_skipped", None)
        bundle_dir = os.path.join(root, "bundles", "gsc")
        for bundle in built.values():
            bundles.write(bundle_dir, bundle)
        spec = [{"block": "gsc", "cohort": "gsc", "doc_id": "GSC001", "segment_idx": "",
                 "anchor": "seizures and a fine tremor", "line": 2}]
        rows, _prov, summary = export_cohort(spec, bundle_dir, "gsc")
        assert summary and summary[0][2] == 0, summary
        assert all(r["segment_idx"] == 0 for r in rows)
        assert {r["row_kind"] for r in rows} <= set(ROW_KINDS)
        assert all(r[k] in CELLS for r in rows for k in roster.ORDER)
    print("export_examples selftest: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
