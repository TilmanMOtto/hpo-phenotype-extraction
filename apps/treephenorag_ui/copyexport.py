"""Markdown serialisers, the numbers behind every panel, ready for ``experiments/findings/``.

Every panel carries a Copy Markdown button whose content comes from one of these functions. The
point is that a figure can go into a slide and its numbers into a findings file without either
being retyped, and that the numbers carry their provenance: which cell, which configuration,
which ground truth file. A pasted table with no provenance line is how two τ values end up in one
paragraph.
"""

from __future__ import annotations

import os


def _pct(value, digits: int = 1) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.{digits}f}%"


def _num(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _plural(count, noun: str) -> str:
    return noun if count == 1 else f"{noun}s"


def table(headers, rows) -> str:
    """A Markdown table."""
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join([head, rule, body]) if rows else "\n".join([head, rule])


def provenance(report: dict, extra: str = "") -> str:
    """The one line that makes a pasted table re-findable six weeks later."""
    op = report.get("operating_point") or "—"
    curated = report.get("curated") or {}
    # The dataset directory, not the file: every curated ground truth ever built is called
    # ``hcy_ground_truth_curated.csv``, so the basename alone names no particular ground truth and a
    # pasted number would be unreproducible. The date is in the directory name.
    gold = (curated.get("name")
            or os.path.basename(report.get("gold_source") or "") or "?")
    bits = [
        f"`{report.get('experiment')}` · {report.get('cohort', '').upper()}",
        f"{report.get('sweep_axis') or 'configuration'}={op}",
        f"gold={gold}",
        f"n={(report.get('counts') or {}).get('n_reports_scored')} reports",
    ]
    if report.get("restricted_n") is not None:
        # Bold and first-class, not a trailing aside: without it a pasted table claims to describe
        # The cohort when it describes a purposive draw from it.
        bits.append(f"**RESTRICTED to {report['restricted_n']} report(s), purposive sample, "
                    "not a cohort rate**")
    if curated.get("n_reports_outside"):
        # Same reasoning as the restriction note above: a rate over the curated cohort is not a
        # rate over the run, and the paste must say so or it will be read as one.
        bits.append(f"**{curated['n_reports_outside']} report(s) outside the curated cohort, "
                    f"not scored**")
    if curated.get("curation_log"):
        bits.append(f"curation log {curated['curation_log']}")
    if report.get("status") and report["status"] != "ok":
        bits.append(f"**status: {report['status']}**")
    if extra:
        bits.append(extra)
    return "_" + " · ".join(bits) + "_"


def curated_md(report: dict) -> str:
    """The curated-ground truth panel as Markdown, what the ground truth covers and what it charged this cell."""
    block = report.get("curated") or {}
    if not block:
        return ""
    flat = report.get("flat") or {}
    rows = [
        ["dataset", block.get("name")],
        ["built", block.get("date") or "n/a"],
        ["curation log (sha256:16)", block.get("curation_log") or "n/a"],
        ["reports in cohort", block.get("n_reports_in_cohort")],
        ["annotated terms", block.get("n_gold_pairs")],
        ["reports in this run outside the cohort", block.get("n_reports_outside")],
        ["annotations excluded by the policy", block.get("n_annotations_excluded")],
        ["excluded annotations predicted here",
         f"{block.get('n_excluded_fp')} of {flat.get('fp')} FP"],
    ]
    reasons = [[reason, count]
               for reason, count in (block.get("excluded_fp_by_reason") or {}).items()]
    policy = [[key, ", ".join(str(v) for v in value) if isinstance(value, (list, tuple))
               else str(value)]
              for key, value in (block.get("policy") or {}).items()]
    parts = [f"### {report.get('label')}, the curated ground truth",
             table(["", "value"], rows)]
    if reasons:
        parts.append("**False positives that are excluded annotations, by reason**\n\n"
                     + table(["dropped for", "FP"], reasons))
    if policy:
        parts.append("**Inclusion policy**\n\n" + table(["key", "value"], policy))
    parts.append(provenance(report))
    return "\n\n".join(parts)


def scorecard_md(report: dict) -> str:
    """The scorecard of one run as Markdown."""
    flat = report.get("flat") or {}
    hier = report.get("hierarchy") or {}
    ledger = report.get("ledger") or {}
    rows = [
        ["micro P / R / F1", f"{_num(flat.get('micro_precision'))} / "
                            f"{_num(flat.get('micro_recall'))} / {_num(flat.get('micro_f1'))}"],
        ["macro P / R / F1", f"{_num(flat.get('macro_precision'))} / "
                            f"{_num(flat.get('macro_recall'))} / {_num(flat.get('macro_f1'))}"],
        ["hP / hR / hF", f"{_num(hier.get('micro_hp'))} / {_num(hier.get('micro_hr'))} / "
                        f"{_num(hier.get('micro_hf'))}"],
        ["TP / FP / FN", f"{flat.get('tp')} / {flat.get('fp')} / {flat.get('fn')}"],
        ["recall (actual)", _pct(ledger.get("recall_actual"))],
        ["recall ceiling (pruning)", _pct(ledger.get("recall_ceiling"))],
        ["recall lost to pruning", _pct(ledger.get("recall_lost_to_pruning"))],
    ]
    return "\n\n".join([
        f"### {report.get('label')}, scorecard",
        table(["metric", "value"], rows),
        _fate_table(ledger),
        provenance(report),
    ])


def _fate_table(ledger: dict) -> str:
    from . import theme

    counts = ledger.get("fate_counts") or {}
    total = sum(counts.values()) or 1
    rows = [[fate, counts.get(fate, 0), _pct(counts.get(fate, 0) / total),
             theme.FATE_HELP.get(fate, "")] for fate in theme.FATE_ORDER]
    return "**Where every annotated term went**\n\n" + table(
        ["fate", "n", "share", "meaning"], rows)


def pruning_md(report: dict) -> str:
    """The pruning summary of one run as Markdown."""
    blocking = report.get("blocking") or {}
    culprits = report.get("culprits") or []
    rows = [
        [c["hpo_id"], c.get("hpo_label", ""), c.get("depth"), c["n_gold_lost"],
         c["n_reports"], _num(c.get("prune_score_min")), _num(c.get("prune_score_max")),
         c.get("tau_would_expand_all") if c.get("tau_would_expand_all") is not None else "—"]
        for c in culprits[:20]
    ]
    n_blocked = blocking.get("n_blocked", 0)
    parts = [
        f"### {report.get('label')}, pruning autopsy",
        f"{n_blocked} gold {_plural(n_blocked, 'term')} never reached. "
        f"Median blocking depth {_num(blocking.get('median'), 1)}; "
        f"{_pct(blocking.get('fraction_at_depth_1'))} cut at depth 1.",
        "**Blocking depth**\n\n" + table(
            ["depth", "annotated terms blocked"],
            sorted(((k, v) for k, v in (blocking.get("histogram") or {}).items()),
                   key=lambda kv: int(kv[0]))),
        "**Culprits, the nodes that refused to expand**\n\n" + table(
            ["hpo_id", "label", "depth", "annotated terms lost", "reports",
             "prune min", "prune max", "τ that would open it"], rows),
        provenance(report),
    ]
    return "\n\n".join(parts)


def fp_md(report: dict) -> str:
    """The false-positive categories of one run as Markdown."""
    taxonomy = (report.get("errors") or {}).get("fp_taxonomy") or {}
    counts = taxonomy.get("counts") or {}
    fractions = taxonomy.get("fractions") or {}
    near = (report.get("errors") or {}).get("near_miss") or {}
    factory = report.get("fp_factory") or []
    return "\n\n".join([
        f"### {report.get('label')}, false-positive anatomy",
        f"{taxonomy.get('n_fp', 0)} false {_plural(taxonomy.get('n_fp', 0), 'positive')}. "
        f"Median distance to the nearest annotated term: {_num(near.get('median'), 1)} hops; "
        f"{_pct(near.get('fraction_within_2'))} are within 2.",
        table(["bucket", "n", "share"],
              [[b, counts.get(b, 0), _pct(fractions.get(b))] for b in counts]),
        "**Which expansions let them in**\n\n" + table(
            ["hpo_id", "label", "FPs below it", "reports"],
            [[f["hpo_id"], f.get("hpo_label", ""), f["n_fp"], f["n_reports"]]
             for f in factory[:20]]),
        provenance(report),
    ])


def fn_md(report: dict) -> str:
    """The false-negative categories of one run as Markdown."""
    taxonomy = (report.get("errors") or {}).get("fn_taxonomy") or {}
    counts = taxonomy.get("counts") or {}
    fractions = taxonomy.get("fractions") or {}
    ledger = report.get("ledger") or {}
    return "\n\n".join([
        f"### {report.get('label')}, false-negative anatomy",
        f"{taxonomy.get('n_fn', 0)} missed annotated {_plural(taxonomy.get('n_fn', 0), 'term')}.",
        table(["what was predicted nearby", "n", "share"],
              [[b, counts.get(b, 0), _pct(fractions.get(b))] for b in counts]),
        _fate_table(ledger),
        provenance(report),
    ])


def frontier_md(report: dict, rows: list[dict]) -> str:
    """Precision, recall and calls per report across the threshold sweep as Markdown."""
    return "\n\n".join([
        f"### {report.get('method_label')} · {report.get('cohort', '').upper()}, τ sweep",
        table(["τ", "micro P", "micro R", "micro F1", "recall ceiling",
               "annotated terms blocked", "SLM calls / report"],
              [[r["operating_point"], _num(r.get("micro_precision")), _num(r.get("micro_recall")),
                _num(r.get("micro_f1")), _pct(r.get("recall_ceiling")), r.get("n_blocked"),
                _num(r.get("mean_slm_calls"), 0)] for r in rows]),
        provenance(report, "swept on disk, not re-run"),
    ])


def calibration_md(report: dict) -> str:
    """The calibration summary of one run as Markdown."""
    parts = [f"### {report.get('label')}, calibration"]
    for name, summary in (report.get("calibration") or {}).items():
        if not summary:
            parts.append(f"**{name}_score**, not measurable on this cell "
                         f"(the sample has only one class).")
            continue
        parts.append(
            f"**{name}_score**, predicting {summary.get('meaning')}\n\n" + table(
                ["metric", "value"],
                [["n scored nodes", summary.get("n")],
                 ["prevalence", _pct(summary.get("prevalence"))],
                 ["ECE (equal-mass)", _num(summary.get("ece_equal_mass"))],
                 ["Brier", _num(summary.get("brier"))],
                 ["AUROC", _num(summary.get("auroc") or summary.get("auc"))],
                 ["AURC", _num(summary.get("aurc"))],
                 ["mean conf | positive", _num(summary.get("mean_confidence_positive"))],
                 ["mean conf | negative", _num(summary.get("mean_confidence_negative"))]]))
    parts.append(provenance(report))
    return "\n\n".join(parts)


def compare_md(a: dict, b: dict) -> str:
    """Two runs side by side as Markdown."""
    def row(name, key, group="flat", fmt=_num):
        return [name, fmt((a.get(group) or {}).get(key)), fmt((b.get(group) or {}).get(key))]

    return "\n\n".join([
        f"### {a.get('label')} @ {a.get('operating_point')} "
        f"vs {b.get('label')} @ {b.get('operating_point')}",
        table(["metric", "A", "B"], [
            row("micro precision", "micro_precision"),
            row("micro recall", "micro_recall"),
            row("micro F1", "micro_f1"),
            row("macro F1", "macro_f1"),
            row("hF", "micro_hf", "hierarchy"),
            row("recall ceiling", "recall_ceiling", "ledger", _pct),
            row("recall lost to pruning", "recall_lost_to_pruning", "ledger", _pct),
        ]),
        "A: " + provenance(a),
        "B: " + provenance(b),
    ])


def findings_block(report: dict, frontier: list[dict] | None = None) -> str:
    """The whole cell, shaped like a section of ``experiments/findings/exp13_findings.md``."""
    parts = [
        f"## {report.get('method_label')} · {report.get('cohort', '').upper()} "
        f"@ {report.get('operating_point')}",
        provenance(report),
        scorecard_md(report).split("\n\n", 1)[1],
    ]
    if report.get("is_tree"):
        parts.append(pruning_md(report).split("\n\n", 1)[1])
    parts.append(fp_md(report).split("\n\n", 1)[1])
    parts.append(fn_md(report).split("\n\n", 1)[1])
    if frontier:
        parts.append(frontier_md(report, frontier).split("\n\n", 1)[1])
    return "\n\n".join(parts)
