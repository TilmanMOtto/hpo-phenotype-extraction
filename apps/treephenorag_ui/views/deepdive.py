"""One report, read top to bottom: the document, the score, the terms, the walk, the evidence.

Every other view aggregates. This one does not, and it is laid out in the order the checking
happens, each panel answering the question the one above it raises:

1. **The report.** The document as it was written, continuous prose, with its own paragraphs and
   line breaks, laid over the segmentation the pipeline indexes by, so hovering any sentence shows
   its extent and names it. This is the ground truth as a human wrote it, before any method touched
   it. When a node is selected, the sentences that were actually *retrieved* for it are marked in
   place, which is the join that makes a wrong verdict legible: the model did not see the report, it
   saw five sentences.
2. **What this report scored.** Precision, recall and F1 for this report alone, with the TP/FP/FN
   split and where its annotated terms went.
3. **Terms in play.** Every TP, FP and FN, each with *how* it is wrong, an FP that is a parent of
   an annotated term and an FP in another organ system are the same row in a confusion matrix and
   completely different failures. The classes come from ``thesis_metrics.errors``, the same
   functions behind the thesis's taxonomy table, computed here for one report. The organ-system
   refinement is added on top, because "unrelated but in the same system" is the distinction a
   reader asks about next and the thesis bucket does not carry it.
4. **The walk.** Every node the traversal scored for this report, by depth and accept score.
5. **The ontology.** The same terms placed back in the hierarchy, with the ancestors that connect
   them, see :mod:`apps.treephenorag_ui.subgraph`. Click a term to open it below.
6. **The node.** For any node: its two scores against the two thresholds, why it was blocked if it
   was, and one row per SLM call, the retrieved sentence, its cosine similarity, the raw
   ``logit_yes``/``logit_no``, the margin, the verdict, and the reconstructed prompt.

The calls scan is the one unbounded read in this package, so it is triggered here and nowhere
else, behind a spinner, and cached afterwards.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from .. import active, discovery, loaders, phenobert, sample, state, subgraph, terms, theme

#: A walk with more nodes than this is drawn as a depth profile instead of node-by-node. A
#: full-ontology traversal visits thousands of nodes per report. Plotting them individually would
#: produce an unreadable smear and a megabyte of figure JSON.
MAX_NODES_DRAWN = 400

#: How many sentences of a report become individually hoverable spans. Clinical reports run to a
#: few dozen sentences. The HCY discharge summaries run longer. Past the cap the remaining text is
#: still emitted, as plain prose, the report is never silently truncated, it just stops being
#: sentence-addressable, which is the same rule ``hcy_curation_ui.views.reader`` uses.
MAX_SENTENCES_SHOWN = 400


def layout():
    """The Dash layout of this view."""
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Report", className="field-label"),
                # Stepping is the common motion on this page, a reader works through a cohort
                # report by report, and reopening a dropdown of a few hundred ids to advance by
                # one is the slowest way to do it.
                html.Div([
                    html.Button("◀", id="dd-report-prev", className="copy-btn step-btn",
                                title="Previous report"),
                    html.Div(dcc.Dropdown(id="dd-report", options=[], value=None, clearable=False),
                             style={"flex": "1 1 auto"}),
                    html.Button("▶", id="dd-report-next", className="copy-btn step-btn",
                                title="Next report"),
                ], className="stepper"),
            ], className="field"),
            html.Div([
                html.Label("Node", className="field-label"),
                dcc.Dropdown(id="dd-node", options=[], value=None, placeholder="pick a node",
                             clearable=True),
            ], className="field"),
            html.Div([
                html.Label("Show", className="field-label"),
                dcc.Dropdown(
                    id="dd-node-filter",
                    options=[{"label": "Terms in play (TP + FP + FN)", "value": "terms"},
                             {"label": "Errors only (FP + FN)", "value": "errors"},
                             {"label": "Accepted", "value": "accepted"},
                             {"label": "Annotated terms", "value": "gold"},
                             {"label": "Culprits", "value": "culprits"},
                             {"label": "Everything", "value": "all"}],
                    value="terms", clearable=False),
            ], className="field"),
        ], className="sidebar-section"),
        html.Div(id="dd-text"),
        html.Div(id="dd-summary"),
        html.Div(id="dd-terms"),
        html.Div(id="dd-walk"),
        # Static, unlike the panels above: ``dd-tree-graph.clickData`` is a callback Input, and an
        # Input naming a component that only exists once some other callback has drawn it is the
        # one Dash wiring mistake that fails silently.
        theme.panel(
            "Where the terms sit in the ontology",
            html.Div([
                dcc.Graph(id="dd-tree-graph", figure=go.Figure(), config=theme.GRAPH_CONFIG),
                html.Div(id="dd-tree-note"),
            ]),
            panel_id="dd-tree", graph_id="dd-tree-graph",
            subtitle="This report's true positives, false positives and false negatives, plus the "
                     "ancestors that connect them. Click a term to open its evidence below."),
        dcc.Loading(html.Div(id="dd-evidence"), type="dot"),
    ])


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "—" if value != value else f"{value:.{digits}f}"


def _present(value) -> bool:
    """Truthiness that survives pandas' ``NA``, whose ``__bool__`` raises rather than answering.

    Every taxonomy column is ``NA`` on a *light* bundle, the deep-dive does not classify the
    whole cohort just to open one report, so ``if row["fp_class"]`` is a ``TypeError`` on any
    false positive, which is the node a reader opens this page to look at.
    """
    return value is not None and not pd.isna(value) and bool(value)


def _int_cell(value):
    """An integer, or an em dash for the ``None``/``NA``/``NaN`` a never-reached node carries."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if number != number else int(number)


def _report_frame(cell_id, operating_point, report_id):
    """``(context, node rows)`` for one report, without building the cohort's table.

    This page has always been about one report at a time, but it used to get there by masking a
    cohort node table that had to exist first. That is what made the loosest swept tau_prune
    unopenable: 1 051 846 rows built and held so a few hundred could be drawn. ``report_bundle``
    reads that report's bytes through the nodes index instead, and runs the same ledger and the
    same table builder over them, so the fates shown here still agree with the scorecard's.

    ``context`` carries ``gold`` and ``predicted`` already narrowed to this report.
    """
    registry = state.get_registry()
    context = registry.report_bundle(cell_id, operating_point, report_id)
    if not context:
        return None, None
    frame = context["frame"]
    if frame is None or frame.empty:
        return context, None
    return context, frame


# ── 1. The report, with its annotations ──────────────────────────────────────

def _text_panel(registry, report: dict, report_id: str, gold, highlighted: set[int] | None,
                mode: str = "light", predicted=None):
    """The report as a document, with the annotated terms beneath it.

    Four layers, none of which can be read off the others:

    *The sentences*, drawn as continuous prose with the text between them restored, headings,
    blank lines, the paragraph breaks a segmentation drops on the floor. Each one is an inline span
    that tints on hover and names itself: ``sentence #4``, the ``sent_index`` the call records carry.
    That numbering is the join between "what the SLM answered" and "what it was shown", so it
    survives everything else on this panel. It lives in the tooltip, not in a gutter because
    a column of numbers down the margin is what stops a clinical report reading as one.

    *The retrieval*, marked in place: the sentences the selected node's SLM calls were actually
    made on. This is the only thing on the page that says what the model was looking at, it did
    not see the report, it saw five sentences.

    *The extraction*, from the PhenoBERT baseline PhenoBERT baseline: each phrase it matched, underlined
    where it sits, tagged with the term it normalised to. A phrase the grounder found and this
    method missed is the concrete form of a recall gap, and it is invisible in a list of ids. The
    layer is additive and degrades on its own, see :mod:`apps.treephenorag_ui.phenobert`, so a cohort
    with no baseline run renders as before, with a note saying why.

    *The ground truth*, on the words a curator read it at. The original HCY ground truth is a set of terms per
    report with no offsets, and against it this layer does not exist: the annotated terms can only
    be a chip row under the text, because there is nothing to point at. The **curated** ground truth carries
    a segment and a trigger word per term, so the term is boxed on its own trigger and coloured by
    whether this method found it, the difference between "recall lost `HP:0001250`" and "missed
    *Anfälle*, sentence 4, the model was never shown it". A term whose trigger cannot be located in
    the segment it was located to is not drawn on any words. It keeps a chip instead. See
    :meth:`apps.treephenorag_ui.curated.CuratedGold.anchors` for why the search is confined to that one
    segment, and :meth:`~apps.treephenorag_ui.curated.CuratedGold.placements` for why a placement is
    verified, not trusted.

    **One coordinate system.** The prose, the inter-sentence gaps, the PhenoBERT offsets and the
    ground truth triggers are all indices into a single string, the baseline's staged copy of the report
    where that layer is usable, the report folder's own text otherwise. Mixing two of them is the
    off-by-a-few-characters bug that puts a mark on the wrong word while every count stays right.
    """
    cohort = report.get("cohort", "")
    evidence = registry.evidence()
    view = registry.view

    text = registry.report_texts(cohort).get(str(report_id))
    sentences = evidence.segments(cohort).get(str(report_id))

    pb = registry.phenobert(cohort)
    placed, why_not = phenobert.annotate(pb, str(report_id), sentences or [])
    # What the copy button hands over: the report as it was written, with no marks in it. The
    # staged text when that is what was annotated, so the copy matches what is on screen, and it
    # is also the string every offset below indexes.
    raw = (pb["texts"].get(str(report_id)) if pb and not why_not else None) or text or ""

    report_view = evidence.report_view(cohort, str(report_id), raw) if sentences else None
    # What is actually drawn per sentence: the staged slice where the baseline annotated one (those
    # are the offsets that were verified), the report's own spelling otherwise.
    drawn = list((report_view or {}).get("display") or sentences or [])
    for index, (staged, _spans) in placed.items():
        if 0 <= index < len(drawn):
            drawn[index] = staged

    curated = registry.curated_for(cohort)
    marks = (curated.placements(str(report_id), sentences or [])
             if curated is not None and sentences else {})
    locations = (curated.locations(str(report_id), drawn, marks)
               if curated is not None and marks else {})

    notes = []
    if sentences and why_not:
        tone = "danger" if "gate G6" in why_not else "info" if "no PhenoBERT baseline" in why_not else "warn"
        notes.append(theme.note(
            f"The PhenoBERT baseline's matches are not shown here: {why_not}. Everything else on "
            f"this page is unaffected. Looked in: {registry.phenobert_source(cohort)}", tone=tone))

    drawn_codes: set[str] = set()

    if not text and not sentences:
        body = theme.empty(
            "Report text not reachable, set the HCY report folder in Data sources. Everything "
            "else on this page is unaffected.")
        subtitle = ""
    elif sentences:
        children = []
        for index in range(min(len(drawn), MAX_SENTENCES_SHOWN)):
            woven, used = _weave(drawn[index], placed.get(index, (None, []))[1],
                                 locations.get(index, ()), predicted, view, mode)
            drawn_codes |= used
            children.append(_segment_span(index, woven, highlighted))
        body = html.Div(
            notes
            + [html.Div(_prose(report_view, children), className="prose report-text")]
            + ([_legend(mode, curated is not None)] if placed or locations else []))
        subtitle = (f"{len(sentences)} sentences. Hover one to see the segment the pipeline works "
                    f"with, it is numbered as the `sent_index` in the call records "
                    f", {evidence.source(cohort)}.")
        if highlighted:
            subtitle += (f" {len(highlighted)} of them "
                         f"{'was' if len(highlighted) == 1 else 'were'} retrieved for the "
                         "selected node, and tinted below.")
        if placed:
            n = phenobert.counts(placed)
            subtitle += (f" Underlined: the raw PhenoBERT baseline's matches, {n['positive']} "
                         f"that became its predictions, {n['negated']} negated, "
                         f"{n['unresolved']} not in this HPO release.")
        if curated is not None:
            shown_terms, total_terms = curated.placement_coverage(
                str(report_id), sentences, marks)
            subtitle += (f" Boxed: the curated ground truth on the words it was annotated at "
                         f"({shown_terms} of this report's {total_terms} terms placed), green "
                         f"where this method found them and red where it did not.")
    else:
        body = html.Div(notes + [html.Pre(text, className="mono")])
        subtitle = (f"Raw text, the sentence numbering could not be recovered "
                    f"({evidence.status(cohort, report_id)}), so the call records' `sent_index` "
                    f"cannot be resolved.")

    return theme.panel(
        "The report, and what was annotated in it",
        html.Div([body, _gold_chips(registry, gold, curated, str(report_id), drawn_codes, marks)]),
        panel_id="dd-text-panel", subtitle=subtitle,
        # The report and not a table of the annotations: what a reader wants out of
        # this panel is the text itself, to paste into a finding or a prompt. The marks are a
        # reading aid and would be noise anywhere else.
        markdown=raw)


def _prose(report_view, sentence_spans: list) -> list:
    """The sentences as continuous text, with the report's own words put back between them.

    A port of ``apps.curation_ui.views.reader._prose``, and it keeps that function's two rules,
    both of which are about not lying to the reader. It must not **reorder**: a sentence whose range
    starts behind the cursor, which the alignment's one backward retry can produce, is emitted in
    sentence order with a plain space, because reading order is what is being checked against. And
    it must not **drop**: whatever follows the last drawn sentence is emitted too, so a trailing
    signature block, or the tail past :data:`MAX_SENTENCES_SHOWN`, is visible, not silently
    absent.

    With no alignment (no report text, or a segmentation that could not be located in it) the
    sentences are simply joined by a space. The prose is then approximate, and the panel's subtitle
    already names the route the segmentation came from.
    """
    text = (report_view or {}).get("text") or ""
    ranges = (report_view or {}).get("ranges") or []
    out: list = []
    cursor = 0

    for index, span_component in enumerate(sentence_spans):
        span = ranges[index] if index < len(ranges) else None
        if text and span is not None and span[0] >= cursor:
            if span[0] > cursor:
                out.append(text[cursor:span[0]])
            cursor = span[1]
        elif out:
            out.append(" ")
        out.append(span_component)

    if text and cursor < len(text):
        out.append(text[cursor:])
    return out


def _segment_span(index: int, children: list, highlighted: set[int] | None):
    """One sentence, inline. The number lives in the tooltip, not in a gutter.

    ``id`` is a pattern id so the calls panel can scroll to it. Dash renders one as
    ``JSON.stringify`` of the dict with its **keys sorted**, which is what makes
    ``{"idx": n, "type": "dd-seg"}`` the element's real DOM id, see the clientside callback in
    ``app.py``.
    """
    retrieved = bool(highlighted) and index in highlighted
    return html.Span(
        children,
        id={"type": "dd-seg", "idx": index},
        className="seg" + (" seg-retrieved" if retrieved else ""),
        title=f"sentence #{index}"
              + (", retrieved for the selected node" if retrieved else ""),
    )


#: Where each kind sits when marks compete for the same characters. The ground truth is somebody asserting
#: a phenotype about these exact words; PhenoBERT is a detection. So a detection never takes the
#: box from an annotation, which is what stops a wide baseline span from stretching a mark across
#: words no annotator claimed. Within a rank the **widest** span wins, because that is the phrase
#: being read.
_MARK_RANK = {"gold": 0, "pb": 1}


def _weave(text: str, pb_spans, gold_spans, predicted, view, mode: str):
    """One sentence's children, with the ground truth and the baseline drawn **in place**.

    Returns ``(children, codes_drawn)``. ``codes_drawn`` is what the chip row below the report must
    *not* repeat, a term already shown on its own words does not need saying twice.

    Marks cannot nest: this walks the sentence with one forward cursor, so a mark starting behind it
    would emit the same characters twice and the prose would stop being the report. Overlaps are
    therefore resolved by **dropping the loser**, not by drawing it approximately: an offset that
    cannot be honoured is one that should not be honoured at all. An annotated term that loses an
    overlap still gets a chip, so nothing recorded goes unshown.

    A port in spirit of ``apps/phenojury_ui/views/patient.py:_annotate`` for the PhenoBERT half, kept
    semantically identical there: the two apps read the same the PhenoBERT baseline run, and a reader comparing
    them must not find the same match drawn two different ways.
    """
    candidates = [(int(s), int(e), "pb", row) for s, e, row in (pb_spans or [])]
    candidates += [(int(s), int(e), "gold", row) for s, e, row in (gold_spans or [])]
    if not candidates:
        return [text], set()

    chosen: list[tuple] = []
    taken: list[tuple[int, int]] = []
    for mark in sorted(candidates, key=lambda m: (_MARK_RANK[m[2]], -(m[1] - m[0]), m[0])):
        start, end = mark[0], mark[1]
        if start < 0 or end > len(text) or end <= start:
            continue
        if any(start < t_end and end > t_start for t_start, t_end in taken):
            continue
        taken.append((start, end))
        chosen.append(mark)
    chosen.sort(key=lambda m: m[0])

    children: list = []
    codes: set[str] = set()
    cursor = 0
    for start, end, kind, row in chosen:
        children.append(text[cursor:start])
        if kind == "gold":
            children.append(_gold_mark(text[start:end], row, predicted, view, mode))
            codes.add(row.hpo_code)
        else:
            children.append(_pb_mark(text[start:end], row, view, mode))
        cursor = end
    children.append(text[cursor:])
    return children, codes


def _found_by_method(view, hpo_code: str, predicted) -> bool | None:
    """Whether this method predicted *hpo_code*. ``None`` when there is no prediction set.

    Resolved through the ontology, not compared as a string. A ground truth file predates the release
    it is scored against, so a ground truth code and a predicted code can be the same phenotype spelled two
    ways, and a raw ``in`` test draws such a term in red on the report while the scorecard above
    counts it as a true positive.
    """
    if predicted is None:
        return None
    return any(terms.same_term(view, hpo_code, code) for code in predicted)


def _gold_mark(phrase: str, row, predicted, view, mode: str):
    """The ground truth trigger, boxed on its own words and tagged with the term it asserts."""
    palette = theme.palette(mode)
    found = _found_by_method(view, row.hpo_code, predicted)
    colour = palette["neutral"] if found is None else (
        palette["good"] if found else palette["critical"])
    verdict = "" if found is None else (" ✓" if found else " ✗")
    return html.Span([
        html.Mark(phrase, className="gold-mark", style={
            "background": "transparent", "color": "inherit",
            "boxShadow": f"inset 0 0 0 1.5px {colour}"}),
        html.Span(f"{terms.label(view, row.hpo_code)}{verdict}", className="pb-tag",
                  style={"borderColor": colour, "color": colour}),
    ], className="pb-ann", title=_gold_title(view, row, found))


def _gold_title(view, row, found: bool | None) -> str:
    """The hover on a ground truth mark: what the term is, then what the curation pass recorded about it."""
    lines = [terms.tooltip(view, row.hpo_code, prefix="curated ground truth →"),
             f"annotated at {row.describe()}"]
    if row.source:
        lines.append(f"source: {row.source}")
    if row.anchor_how:
        lines.append(f"evidence located {row.anchor_how}")
    if row.qualifiers:
        lines.append("qualifiers: " + ", ".join(row.qualifiers))
    if row.note:
        lines.append(row.note)
    if found is not None:
        lines.append("found by this method" if found else "MISSED by this method")
    return "\n\n".join(lines)


def _pb_mark(phrase: str, row: dict, view, mode: str):
    """One PhenoBERT match: the phrase underlined, tagged with the term it normalised to.

    The underline says *where* the baseline found something. The tag says *what it became*, which is
    the half that decides whether the extraction was right. The name and not the id, because the tag
    sits inside flowing text and is read at a glance, the id is one hover away.
    """
    palette = theme.palette(mode)
    colours = {"positive": palette["good"], "negated": palette["neutral"],
               "unresolved": palette["warning"]}
    kind = phenobert.kind(row)
    colour = colours[kind]
    return html.Span([
        html.Mark(phrase, className=f"pb-mark pb-mark-{kind}",
                  style={"background": "transparent", "borderBottom": f"2px solid {colour}",
                         "color": "inherit",
                         "textDecoration": "line-through" if kind == "negated" else "none"}),
        html.Span(_tag_text(view, row, kind), className=f"pb-tag pb-tag-{kind}",
                  style={"borderColor": colour, "color": colour}),
    ], className="pb-ann", title=_mark_title(view, row, kind))


def _tag_text(view, row: dict, kind: str) -> str:
    """The inline label: the term this phrase normalised to, plus why it did not count.

    Resolved through the ontology, not read off the row, so a term is named the same way
    here as it is in the node table above.
    """
    label = terms.label(view, row["hpo_id"]) if view is not None else (
        row.get("hpo_label") or row["hpo_id"])
    suffix = {"negated": " · negated", "unresolved": " · not in this HPO release"}.get(kind, "")
    return f"{label or row['hpo_id']}{suffix}"


_MARK_WHY = {
    "positive": "counted as a PhenoBERT prediction",
    "negated": "NEGATED, PhenoBERT found this and ruled it out, so it is not a prediction",
    "unresolved": "not in this HPO release, so it was dropped from the predictions",
}


def _mark_title(view, row: dict, kind: str) -> str:
    """The hover on one match: what it is, why it counted or did not, and what the term means."""
    lines = [terms.tooltip(view, row["hpo_id"], prefix="PhenoBERT →"),
             f"matched {row['phrase']!r}"]
    score = row.get("score", row.get("confidence"))
    if score is not None:
        lines.append(f"match score {score:.2f} (a fixed filter threshold, not a confidence)")
    if row.get("raw_hpo_id") and row["raw_hpo_id"] != row["hpo_id"]:
        lines.append(f"remapped from the alt id {row['raw_hpo_id']}")
    lines.append(_MARK_WHY[kind])
    return "\n\n".join(lines)


def _legend(mode: str, curated: bool):
    """The annotation kinds, drawn the way they are drawn in the text."""
    palette = theme.palette(mode)
    entries = [("phrase", "HPO term", "became a PhenoBERT prediction", palette["good"], False,
                False),
               ("phrase", "HPO term · negated", "found, then ruled out", palette["neutral"], True,
                False),
               ("phrase", "HPO term · not in this HPO release", "dropped", palette["warning"],
                False, False)]
    if curated:
        entries += [
            ("trigger", "HPO term ✓", "curated ground truth, found by this method", palette["good"], False,
             True),
            ("trigger", "HPO term ✗", "curated ground truth, MISSED by this method", palette["critical"],
             False, True),
        ]
    return html.Div(
        [html.Span([
            html.Span([
                html.Mark(phrase, className="gold-mark" if boxed else "", style={
                    "background": "transparent", "color": "inherit",
                    **({"boxShadow": f"inset 0 0 0 1.5px {colour}"} if boxed
                       else {"borderBottom": f"2px solid {colour}"}),
                    "textDecoration": "line-through" if struck else "none"}),
                html.Span(tag, className="pb-tag", style={"borderColor": colour, "color": colour}),
            ], className="pb-ann"),
            html.Span(f" {meaning}", className="stat-sub"),
        ], style={"marginRight": "16px"})
            for phrase, tag, meaning, colour, struck, boxed in entries],
        className="legend-row")


def _gold_chips(registry, gold, curated=None, report_id: str = "", drawn_codes=(), marks=None):
    """The annotated terms, each carrying its definition and synonyms as a native tooltip.

    With a curated ground truth loaded the row becomes the **residue**: terms not already drawn on the words
    above, split by why. A term drawn in place is not repeated, it has been shown where it belongs.

    The two residues are different failures and are worth separating. *Placed but not drawn* is an
    annotation whose sentence is known and whose trigger word could not be located in it (or which
    lost an overlap to a wider mark), the term is in the report somewhere near a known sentence.
    *Not placed on any sentence* is an annotation nobody located, or one whose segment this app's
    segmentation does not agree with. Both count towards recall either way, and a reader who cannot
    account for a gap in a chip row will assume the app dropped something.
    """
    if not gold:
        return html.Div("No annotated terms for this report.", className="stat-sub")
    view = registry.view
    if curated is None:
        return html.Div([
            html.Div("annotated terms (ground truth)", className="stat-label",
                     style={"marginTop": "12px"}),
            html.Div([
                html.Span([html.Code(hpo_id), " ", terms.label(view, hpo_id)],
                          className="pill has-def", title=terms.tooltip(view, hpo_id))
                for hpo_id in sorted(gold)
            ], className="chip-row"),
        ])

    placed_codes = {row.hpo_code for rows in (marks or {}).values() for row in rows}
    undrawn = sorted(set(gold) - set(drawn_codes))
    if not undrawn:
        return html.Div(
            f"All {len(gold)} curated annotated terms are drawn on the sentences above.",
            className="stat-sub", style={"marginTop": "12px"})

    groups = [
        ("placed on a sentence, but not on any words",
         [c for c in undrawn if c in placed_codes],
         "The sentence is known; the trigger word the curator typed could not be located in it, or "
         "it lost an overlap to a wider mark. Hover a chip for the sentence it was placed on."),
        ("not placed on any sentence",
         [c for c in undrawn if c not in placed_codes],
         "Either no curator located evidence for these, or their segment is not one this "
         "app's segmentation recovered. They count towards recall either way."),
    ]
    blocks = []
    for title, codes, why in groups:
        if not codes:
            continue
        blocks += [
            html.Div(f"curated annotated terms {title}", className="stat-label",
                     style={"marginTop": "12px"}),
            html.Div([
                html.Span([html.Code(hpo_id), " ", terms.label(view, hpo_id)],
                          className="pill has-def",
                          title=_unplaced_title(curated, view, report_id, hpo_id))
                for hpo_id in codes
            ], className="chip-row"),
            html.Div(why, className="stat-sub"),
        ]
    return html.Div(blocks)


def _unplaced_title(curated, view, report_id: str, hpo_id: str) -> str:
    """What the term means, then everything the curation pass recorded about where it came from."""
    head = terms.tooltip(view, hpo_id)
    row = curated.find(report_id, hpo_id) if curated is not None else None
    if row is None:
        return head
    return head + "\n\n" + (
        f"{row.source or 'unknown source'} · {row.describe()}"
        f"{' · evidence located ' + row.anchor_how if row.anchor_how else ' · evidence never located'}")


#: The ontology readers live in :mod:`apps.treephenorag_ui.terms` so that every mention of a phenotype in
#: this app, a box on a sentence, a chip under the report, a row in the terms table, an option in
#: The node picker, a point in the diagram, hovers to the same words. These aliases keep the call
#: sites in this module short. They are the same functions.
_label = terms.label
_entry = terms.entry
_definition = terms.definition
_synonyms = terms.synonyms
_tooltip = terms.tooltip


def _term_meaning(view, hpo_id: str):
    """What the selected term *means*, above the evidence for it.

    A verdict on ``HP:0001250`` cannot be judged without knowing what HP:0001250 is, and until now
    that was only available as a native tooltip on a chip elsewhere on the page, invisible to
    anyone who did not know to hover. It sits above the SLM calls because it is the question a
    reader asks first: what was the model being asked about?
    """
    if _entry(view, hpo_id) is None:
        return theme.note(
            f"`{hpo_id}` is not in this HPO release, so it has no definition here. A predicted "
            "code the ontology does not know is either a hallucination or an id from another "
            "release.", tone="warn")

    definition = _definition(view, hpo_id)
    synonyms = _synonyms(view, hpo_id)
    return html.Div([
        html.Div("definition", className="stat-label"),
        html.Div(definition or "No definition is recorded for this term in this HPO release.",
                 className="report-text",
                 style={} if definition else {"fontStyle": "italic"}),
        html.Div([
            html.Div("also known as", className="stat-label", style={"marginTop": "10px"}),
            html.Div([html.Span(s, className="pill") for s in synonyms], className="chip-row"),
        ]) if synonyms else None,
    ], style={"marginBottom": "14px"})


def _cell_panel(registry, report_id: str):
    """Why this report is in the deep-dive sample, if it is, the cell and PhenoBERT's score on it.

    The cell is the whole reason the report is on screen, and without it the sample is 20
    indistinguishable ids. ``pools.csv``'s per-report row is shown beside it because the two
    outcome cells are *defined* by that score: seeing ``pb_worst`` next to F1 = 0.00 is what makes
    the cell an argument, not a label.
    """
    frame = registry.sample
    if frame is None:
        return None
    cell = frame.label(report_id)
    if not cell:
        return None

    row = frame.row(report_id) or {}
    tiles = [theme.stat("sample cell", cell, f"drawn {frame.generated}" if frame.generated else "")]
    if row.get("n_gold") is not None:
        tiles.append(theme.stat("annotated terms", str(row["n_gold"]), "in the curated ground truth"))
    if row.get("f1") is not None:
        tiles.append(theme.stat("PhenoBERT F1", f"{row['f1']:.2f}",
                                f"{row.get('tp', 0)} TP / {row.get('fp', 0)} FP / "
                                f"{row.get('fn', 0)} FN",
                                help_text="PhenoBERT's score on this report, from the frame's "
                                          "pools.csv. It is the quantity the pb_best and pb_worst "
                                          "cells are defined on, not a score for the run open "
                                          "here."))
    qualifiers = [f"{name} x{row[name]}" for name in sample.QUALIFIERS if row.get(name)]
    if qualifiers:
        tiles.append(theme.stat("qualifiers", ", ".join(qualifiers),
                                "annotation labels carried by this report"))

    return theme.panel(
        "Why this report",
        html.Div([
            theme.stat_row(tiles),
            html.Div(frame.note(report_id), className="stat-sub", style={"marginTop": "10px"}),
            theme.note(sample.WARNING, tone="info"),
        ]),
        panel_id="dd-cell-panel",
    )


# ── 2. what this report scored ───────────────────────────────────────────────

def _summary(subset, report: dict, report_id: str, gold, predicted):
    """This report's own P/R/F1 and error split, over the report itself, not the cohort."""
    if subset is None or subset.empty:
        return theme.empty(f"No nodes recorded for report {report_id}.")
    row = next((r for r in (report.get("per_report") or [])
                if str(r.get("report_id")) == str(report_id)), {})
    counts = subset["outcome"].value_counts()
    states = subset["state"].value_counts()

    tiles = [
        theme.stat("F1", _fmt(row.get("f1")), f"report {report_id}",
                   help_text="This report alone, at this configuration. The scorecard's micro F1 "
                             "is pooled over the cohort and will not equal it."),
        theme.stat("precision", _fmt(row.get("precision")),
                   f"{int(counts.get('TP', 0))} of {len(predicted)} predicted"),
        theme.stat("recall", _fmt(row.get("recall")),
                   f"{int(counts.get('TP', 0))} of {len(gold)} annotated"),
        theme.stat("TP / FP / FN", f"{int(counts.get('TP', 0))} / {int(counts.get('FP', 0))} / "
                                   f"{int(counts.get('FN', 0))}",
                   f"{int(states.get('visited', 0))} nodes scored"),
        theme.stat("blocked here", str(int(states.get("blocked", 0))),
                   "never reached, a pruning failure",
                   tone="critical" if states.get("blocked", 0) else None),
        theme.stat("expanded", str(int(subset["expanded"].fillna(False).astype(bool).sum())),
                   "nodes whose children were revealed"),
    ]
    return theme.stat_row(tiles)


# ── 3. terms in play ─────────────────────────────────────────────────────────

def _relations(subset, gold, predicted, view) -> dict:
    """``{hpo_id: (class, reference, distance)}`` for one report's errors.

    Computed here, not read off the node table because the table only carries the classes
    on a *heavy* bundle, which classifies every false positive in the cohort. One report's worth
    is a handful of bounded ontology walks, the deep-dive must not make a reader wait for the
    other hundred reports to be classified before it can say what one term is.
    """
    from hpo_extraction.evaluation.metrics import errors

    out: dict[str, tuple] = {}
    if subset is None or subset.empty:
        return out
    for record in subset.itertuples():
        hpo_id = str(record.hpo_id)
        if record.outcome == "FP":
            bucket, reference = errors.classify_false_positive(hpo_id, gold, view)
            distance = (view.undirected_distance(hpo_id, reference)
                        if reference is not None else None)
            out[hpo_id] = (bucket, reference, distance)
        elif record.outcome == "FN":
            out[hpo_id] = (errors.classify_false_negative(hpo_id, predicted, view), None, None)
    return out


def _same_system(view, hpo_id: str, others) -> bool:
    """Whether a term shares a layer-1 organ system with anything in *others*.

    The thesis FP taxonomy stops at ``unrelated``, which pools "a different phenotype of the same
    organ system" with "a different organ system entirely". Those are not the same error, so the
    refinement is shown beside the bucket, never in place of it, because the bucket is the number
    the thesis tables report.
    """
    system = view.layer1(hpo_id)
    return bool(system) and any(view.layer1(other) == system for other in others)


def _relation_cell(view, hpo_id: str, outcome: str, relation, gold, predicted):
    if outcome == "TP":
        return html.Span("exact match", className="stat-sub")
    if relation is None:
        return "—"
    bucket, reference, distance = relation
    parts = [html.Span(bucket, className="pill",
                       title=(theme.FP_HELP if outcome == "FP" else theme.FN_HELP).get(bucket, ""))]
    if reference:
        parts.append(html.Span([" nearest ground truth ", html.Code(reference), " ",
                                _label(view, reference)], className="stat-sub"))
    if distance is not None:
        parts.append(html.Span(f" · {int(distance)} hops", className="stat-sub"))
    if bucket in ("unrelated", "nothing_near") and _same_system(
            view, hpo_id, gold if outcome == "FP" else predicted):
        parts.append(html.Span(" · same organ system", className="pill",
                               title="shares a layer-1 organ system with the other side, which "
                                     "the thesis bucket does not distinguish"))
    return html.Span(parts)


def _evidence_cell(curated, report_id: str, hpo_id: str, outcome: str):
    """What a curator wrote down about this term, or why it is not in the ground truth.

    Three different claims share one column, and conflating them would make it useless:

    *For an annotated term* (TP or FN), the trigger word and the sentence a curator placed it on. On a
    miss this is the whole point: "recall lost `HP:0001250`" becomes "missed *Anfälle* in
    sentence 4", which is a thing a reader can go and look at.

    *For a false positive the policy dropped*, the reason it was dropped. A ``family`` finding a
    method predicted is not the same error as a term nobody ever wrote, and the raw HCY ground truth would
    have scored it as a hit.

    *For a false positive nobody annotated*, nothing,. An empty cell here means the
    curated dataset has no record of the term in this report, which is the claim.
    """
    if curated is None:
        return "—"
    if outcome in ("TP", "FN"):
        row = curated.find(report_id, hpo_id)
        if row is None or not row.in_gold:
            return html.Span("no curated evidence", className="stat-sub")
        bits = [row.describe()]
        if row.source:
            bits.append(row.source)
        if row.qualifiers:
            bits.append("/".join(row.qualifiers))
        return html.Span(" · ".join(bits),
                         title=f"evidence located: {row.anchor_how or 'no'}"
                               + (f" · note: {row.note}" if row.note else ""))
    row = curated.excluded(report_id, hpo_id)
    if row is None:
        return html.Span("—", title="No curated annotation for this term in this report.")
    return html.Span([
        html.Span(row.exclude_reason or "excluded", className="pill"),
        " ", row.describe(),
    ], title=row.why)


def _terms_panel(subset, gold, predicted, view, report_id: str, curated=None):
    """One row per term in play, worst first: FP, then FN, then the TPs that worked."""
    if subset is None or subset.empty:
        return theme.panel("Terms in play", theme.empty("Nothing to show for this report."),
                           panel_id="dd-terms-panel")

    frame = subset.loc[subset["outcome"].isin(("TP", "FP", "FN"))]
    if frame.empty:
        return theme.panel(
            "Terms in play",
            theme.empty("This report has no annotated terms and predicted nothing."),
            panel_id="dd-terms-panel")

    relations = _relations(frame, gold, predicted, view)
    # Errors first, deepest first inside each group: a deep term is the specific claim, and the
    # specific claims are what a reader checks.
    order = {"FP": 0, "FN": 1, "TP": 2}

    def _key(record):
        depth = _int_cell(record.depth)
        return (order.get(record.outcome, 3),
                -depth if isinstance(depth, int) else 0, str(record.hpo_id))

    records = sorted(frame.itertuples(), key=_key)

    rows = []
    for record in records:
        hpo_id = str(record.hpo_id)
        # A fate is a *ground truth* term's cause of death, so a false positive has none, the column
        # falls back to the traversal state, which is the equivalent fact about a node nothing
        # annotated. Both are named in the subtitle, not pooled into one word.
        fate = record.fate if isinstance(record.fate, str) else record.state
        rows.append([
            html.Span([html.Code(hpo_id), " ", str(record.hpo_label)],
                      className="has-def", title=_tooltip(view, hpo_id)),
            html.Span(record.outcome, className="pill",
                      style={"color": theme.outcome_colors()[record.outcome]}),
            _relation_cell(view, hpo_id, record.outcome, relations.get(hpo_id), gold, predicted),
            *([_evidence_cell(curated, report_id, hpo_id, record.outcome)]
              if curated is not None else []),
            html.Span(str(fate), className="pill", title=theme.FATE_HELP.get(str(fate), "")),
            html.Code(str(record.culprit)) if isinstance(record.culprit, str) else "—",
            _fmt(record.accept_score),
            _fmt(record.prune_score),
            _int_cell(record.depth),
        ])

    counts = frame["outcome"].value_counts()
    headers = (["term", "outcome", "how wrong"]
               + (["curated evidence"] if curated is not None else [])
               + ["fate", "blocked by", "accept", "prune", "depth"])
    # accept / prune / depth are always the last three, wherever the evidence column lands.
    numeric = {len(headers) - 3: "num", len(headers) - 2: "num", len(headers) - 1: "num"}
    subtitle = ("`how wrong` is the thesis error taxonomy (`thesis_metrics.errors`) computed for "
                "this report: for a false positive, where it sits relative to the nearest "
                "annotated term; for a miss, what was predicted around it. `fate` is how a ground truth "
                "term died, `blocked` means the traversal never reached it, and `blocked by` "
                "names the node that severed its last path; a false positive has no fate, so it "
                "shows the traversal state instead.")
    if curated is not None:
        subtitle += (" `curated evidence` is what the curation pass recorded: for an annotated term, "
                     "the trigger word and the sentence it was placed on; for a false positive, "
                     "the reason the policy dropped it, where it dropped one at all.")
    return theme.panel(
        f"Terms in play ({int(counts.get('TP', 0))} TP · {int(counts.get('FP', 0))} FP · "
        f"{int(counts.get('FN', 0))} FN)",
        theme.table(headers, rows, align=numeric),
        panel_id="dd-terms-panel",
        subtitle=subtitle,
        markdown=_terms_md(records, relations, report_id, curated))


def _terms_md(records, relations, report_id: str, curated=None) -> str:
    """The same table as Markdown. The evidence column travels with it, because a miss pasted
    into a findings file without the words it was missed at is a claim nobody can check."""
    extra = " evidence |" if curated is not None else ""
    lines = [f"**Terms in play, report {report_id}**", "",
             f"| term | outcome | how wrong |{extra} fate | accept | prune |",
             "|---|---|---|" + ("---|" if curated is not None else "") + "---|---|---|"]
    for record in records:
        hpo_id = str(record.hpo_id)
        relation = relations.get(hpo_id)
        fate = record.fate if isinstance(record.fate, str) else record.state
        cell = ""
        if curated is not None:
            cell = f" {_evidence_md(curated, report_id, hpo_id, record.outcome)} |"
        lines.append(
            f"| {record.hpo_id} {record.hpo_label} | {record.outcome} | "
            f"{relation[0] if relation else '—'} |{cell} {fate} | "
            f"{_fmt(record.accept_score)} | {_fmt(record.prune_score)} |")
    return "\n".join(lines)


def _evidence_md(curated, report_id: str, hpo_id: str, outcome: str) -> str:
    """:func:`_evidence_cell` as plain text, same three claims, no markup."""
    if outcome in ("TP", "FN"):
        row = curated.find(report_id, hpo_id)
        if row is None or not row.in_gold:
            return "no curated evidence"
        return " · ".join([row.describe()] + ([row.source] if row.source else []))
    row = curated.excluded(report_id, hpo_id)
    if row is None:
        return "—"
    return f"excluded ({row.exclude_reason or 'unspecified'}) · {row.describe()}"


# ── 4. The walk ──────────────────────────────────────────────────────────────

def _hover_def(view, hpo_id: str, width: int = 64) -> str:
    """The term's definition as a Plotly hover fragment, wrapped, or ``""``.

    A figure hover is the one term surface that cannot carry a native ``title``, so the definition
    has to travel in ``customdata``. Wrapped because Plotly does not wrap for you and a two-sentence
    HPO definition on one line runs off the side of the screen. Truncated because a hover box taller
    than the figure covers the thing being hovered.
    """
    if view is None or not hpo_id:
        return ""
    text = terms.definition(view, str(hpo_id))
    if not text:
        return ""
    if len(text) > 300:
        text = text[:297].rsplit(" ", 1)[0] + "…"
    return "<br><br>" + "<br>".join(textwrap.wrap(text, width))


def _walk_figure(subset, mode: str, view=None) -> go.Figure:
    """Nodes by depth against their two scores, the shape of one report's traversal."""
    palette = theme.palette(mode)
    outcome_colors = theme.outcome_colors(mode)
    visited = subset.loc[subset["state"] == "visited"]

    figure = go.Figure()
    if visited.empty:
        return go.Figure(layout=theme.plotly_layout(mode, height=340))

    if len(visited) > MAX_NODES_DRAWN:
        # Too many to draw individually: fall back to the depth profile, which is the part of the
        # picture that survives at that scale anyway.
        for outcome in ("TN", "FP", "FN", "TP"):
            group = visited.loc[visited["outcome"] == outcome]
            if group.empty:
                continue
            counts = group.groupby("depth").size()
            figure.add_bar(x=counts.index.tolist(), y=counts.to_numpy().tolist(), name=outcome,
                           marker_color=outcome_colors[outcome],
                           hovertemplate=f"{outcome} · depth %{{x}}<br>%{{y}} nodes<extra></extra>")
        figure.update_layout(**theme.plotly_layout(
            mode, height=340, barmode="stack",
            xaxis={"title": "ontology depth", "dtick": 1},
            yaxis={"title": "nodes scored", "type": "log"}))
        return figure

    jitter = np.random.default_rng(0).uniform(-0.22, 0.22, size=len(visited))
    for outcome in ("TN", "FN", "FP", "TP"):
        mask = (visited["outcome"] == outcome).to_numpy()
        if not mask.any():
            continue
        group = visited.loc[mask]
        figure.add_scatter(
            x=(group["depth"].to_numpy(dtype=float) + jitter[mask]),
            y=group["accept_score"].to_numpy(dtype=float),
            mode="markers", name=outcome,
            marker={
                "size": 11, "color": outcome_colors[outcome],
                "line": {"width": [2 if e else 0 for e in group["expanded"]],
                         "color": palette["text"]},
            },
            customdata=np.stack([
                group["hpo_id"].to_numpy(dtype=object),
                # astype(object) first: hpo_label is categorical (nodes._COMPACT_COLUMNS), and
                # fillna with a value outside the categories raises, not filling.
                group["hpo_label"].astype(object).fillna("").to_numpy(dtype=object),
                group["prune_score"].to_numpy(dtype=float),
                group["expanded"].to_numpy(dtype=bool),
                np.asarray([_hover_def(view, i) for i in group["hpo_id"]], dtype=object),
            ], axis=-1),
            hovertemplate="<b>%{customdata[1]}</b><br>%{customdata[0]}"
                          "<br>accept %{y:.3f} · prune %{customdata[2]:.3f}"
                          "<br>expanded: %{customdata[3]}"
                          "%{customdata[4]}<extra></extra>",
        )
    figure.update_layout(**theme.plotly_layout(
        mode, height=360,
        xaxis={"title": "ontology depth", "dtick": 1},
        yaxis={"title": "accept_score", "range": [-0.02, 1.02]}))
    return figure


# ── 5. The ontology subgraph ─────────────────────────────────────────────────

def _tree_figure(layout: dict, labels, selected, mode: str, view=None) -> go.Figure:
    """The induced subgraph as a clickable diagram: edges behind, one trace per outcome in front.

    ``customdata`` carries the HPO id on every marker, which is what the click callback reads, ``pointNumber`` alone would index into whichever trace was clicked and mean nothing.
    """
    palette = theme.palette(mode)
    colors = {**theme.outcome_colors(mode), "context": palette["neutral"]}
    figure = go.Figure()
    positions = {node["hpo_id"]: (node["x"], node["y"]) for node in layout["nodes"]}
    if not positions:
        return go.Figure(layout=theme.plotly_layout(mode, height=320))

    for edges, dash, width in ((layout["edges"], "solid", 1.2),
                               (layout["extra_edges"], "dot", 1.0)):
        xs: list = []
        ys: list = []
        for parent, child in edges:
            if parent not in positions or child not in positions:
                continue
            xs += [positions[parent][0], positions[child][0], None]
            ys += [positions[parent][1], positions[child][1], None]
        if xs:
            figure.add_scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                               line={"color": palette["border"], "width": width, "dash": dash})

    for group in subgraph.GROUPS:
        members = [n for n in layout["nodes"] if n["group"] == group]
        if not members:
            continue
        ids = [n["hpo_id"] for n in members]
        figure.add_scatter(
            x=[n["x"] for n in members], y=[n["y"] for n in members],
            mode="markers", name=group,
            marker={
                "size": [16 if i == selected else 11 for i in ids],
                "color": colors[group],
                "symbol": ["circle" if group != "context" else "circle-open" for _ in members],
                "line": {"width": [3 if i == selected else 1 for i in ids],
                         "color": palette["text"]},
            },
            customdata=np.stack([
                np.asarray(ids, dtype=object),
                np.asarray([labels.get(i, i) for i in ids], dtype=object),
                np.asarray([n["depth"] if n["depth"] is not None else -1 for n in members]),
                np.asarray([_hover_def(view, i) for i in ids], dtype=object),
            ], axis=-1),
            hovertemplate="<b>%{customdata[1]}</b><br>%{customdata[0]}"
                          f"<br>{group} · depth %{{customdata[2]}}"
                          "%{customdata[3]}<extra></extra>",
        )

    figure.update_layout(**theme.plotly_layout(
        mode, height=440, clickmode="event",
        margin={"l": 20, "r": 20, "t": 34, "b": 20},
        xaxis={"visible": False}, yaxis={"title": "ontology depth (root at the top)",
                                         "showticklabels": False}))
    return figure


def _tree_note(layout: dict, view, selected):
    notes = []
    if layout["orphans"]:
        notes.append(theme.note([
            f"{len(layout['orphans'])} code(s) have no place in the hierarchy and are not drawn: ",
            html.Span(", ".join(o["hpo_id"] for o in layout["orphans"][:8]), className="mono"),
            ". A code the ontology does not know (a hallucination) or one outside the "
            "phenotypic-abnormality subtree has no ancestors to connect it to.",
        ], tone="warn"))
    if layout["n_truncated"]:
        notes.append(theme.note(
            f"{layout['n_truncated']} connecting ancestor(s) were dropped to keep the diagram "
            f"under {subgraph.MAX_NODES} nodes. Every TP, FP and FN is drawn; only structure was "
            "thinned.", tone="warn"))
    if selected:
        notes.append(theme.note([
            "Selected: ", html.Code(selected), " ", _label(view, selected),
            ", its evidence is in the panel below.",
        ], tone="info"))
    return html.Div(notes)


# ── 6. The node ──────────────────────────────────────────────────────────────

def _row_for(subset, view, hpo_id: str):
    """The node table's row for *hpo_id*, matching the raw id **or** the resolved one.

    The two are not interchangeable here. The node table keys on the raw id as the artifacts wrote
    it, matching ``scoring.align``, while every id coming out of the ontology diagram has been
    through ``view.resolve`` (:mod:`apps.treephenorag_ui.subgraph`). A term recorded under an alt id is
    therefore in the table and unfindable by the id the click produced, and the page would report
    it as never scored, the one answer that is worse than no answer.
    """
    if subset is None or subset.empty or not hpo_id:
        return None
    ids = subset["hpo_id"].astype(str)
    row = subset.loc[ids == str(hpo_id)]
    if row.empty and view is not None:
        resolved = view.resolve(hpo_id)
        if resolved:
            row = subset.loc[ids.map(lambda i: view.resolve(i) or i) == resolved]
    return None if row.empty else row


#: How far above a node to look for the pruning decision that stopped the walk. Deeper than any
#: real HPO branch. The cap is there so a cycle or a pathological DAG cannot hang a callback.
MAX_ANCESTOR_HOPS = 40


def nearest_pruned_ancestor(view, subset, hpo_id: str) -> dict | None:
    """The closest ancestor of *hpo_id* that the traversal scored and refused to expand.

    This is the answer to the question a reader asks by clicking a node that has no evidence: the
    walk never got here, and *this* is where it stopped. Returns ``{"hpo_id", "hops",
    "prune_score", "label"}`` for the nearest such ancestor, or ``None`` when there is none within
    :data:`MAX_ANCESTOR_HOPS`, which means the node is simply outside the walk, not cut off
    from it.

    Breadth-first *upward*, level by level, because the hop count is half the answer: an ancestor
    one edge up is a near miss and one six edges up means the branch was abandoned near the root.
    ``view.ancestors`` cannot be used, it returns an unordered set and carries no distance.
    """
    if view is None or subset is None or subset.empty or not hpo_id:
        return None

    start = view.resolve(hpo_id) or hpo_id
    frontier = {start}
    seen = {start}
    for hops in range(1, MAX_ANCESTOR_HOPS + 1):
        frontier = {p for node in frontier for p in view.direct_parents(node)} - seen
        if not frontier:
            return None
        seen |= frontier
        for candidate in sorted(frontier):
            row = _row_for(subset, view, candidate)
            if row is None:
                continue
            record = row.iloc[0]
            if not bool(record["expanded"]):
                return {"hpo_id": str(record["hpo_id"]), "hops": hops,
                        "prune_score": record["prune_score"],
                        "label": _label(view, candidate)}
    return None


def _score_tiles(record, operating_point: str | None) -> list:
    """The two score tiles, tinted by what the run actually did with each score.

    Tinted from the recorded ``accepted``/``expanded`` booleans, not by comparing the score
    to a threshold. Those booleans are what the traversal did. A comparison would have to guess
    τ_accept on a 1-D ``tau_0.5`` configuration, where it is not in the directory name at all.
    The thresholds are named in the sub-label where they are known, and only there.
    """
    tau_prune = discovery.op_value(operating_point) if operating_point else None
    tau_accept = discovery.op_accept(operating_point) if operating_point else None
    # ``_present``, not ``bool``: both flags are NaN on a synthetic row for an annotated term the
    # traversal never reached, and ``bool(NA)`` raises, not answering.
    accepted = _present(record["accepted"])
    expanded = _present(record["expanded"])

    accept_sub = (f"vs τ_accept {tau_accept:g}, " if tau_accept is not None
                  else "vs τ_accept, ") + ("accepted, so reported" if accepted
                                            else "below it, so not reported")
    prune_sub = (f"vs τ_prune {tau_prune:g}, " if tau_prune is not None
                 else "vs τ_prune, ") + ("expanded, so its children were scored" if expanded
                                          else "pruned, so this branch stopped here")
    return [
        theme.stat("accept_score", _fmt(record["accept_score"]), accept_sub,
                   tint="good" if accepted else "critical"),
        theme.stat("prune_score", _fmt(record["prune_score"]), prune_sub,
                   tint="good" if expanded else "warning"),
    ]


def _unevaluated_panel(registry, bundle, subset, report: dict, report_id: str, hpo_id: str,
                       cell_id: str):
    """A node the traversal never scored, and where, above it, the walk stopped.

    The ontology diagram draws every ancestor connecting this report's terms, so a reader can click
    a node that has no row in the table. Answering "that node is not in this report's table" states
    the obvious and hides the interesting part: it is not there *because* something above it was
    pruned, and that pruning decision, its score, its distance, and the sentences behind it, is
    the evidence for why this term was never considered.
    """
    view = registry.view
    culprit = nearest_pruned_ancestor(view, subset, hpo_id)

    if culprit is None:
        context = theme.note(
            "This node was never scored, and no pruned ancestor of it was scored either, so it "
            "is outside the walk rather than cut off from it. It is drawn here only as structure "
            "connecting the terms that are in play.", tone="info")
        calls = html.Div()
    else:
        context = theme.note([
            "Never scored. The traversal stopped ",
            html.B(f"{culprit['hops']} hop{'s' if culprit['hops'] != 1 else ''} above"),
            ", at ", html.Code(culprit["hpo_id"]), " ", culprit["label"],
            f", whose prune score was {_fmt(culprit['prune_score'])}, below the τ_prune of this "
            "configuration. Its children were never scored, so nothing below it was ever asked "
            "about, and no τ_accept could have recovered this term.",
        ], tone="danger")
        calls = html.Div([
            html.Div(f"the calls behind {culprit['hpo_id']}, the evidence that severed this "
                     "branch", className="stat-label", style={"marginTop": "6px"}),
            _calls_panel(registry, bundle, cell_id, report, report_id, culprit["hpo_id"]),
        ])

    return theme.panel(
        f"{_label(view, hpo_id)} · {hpo_id}",
        html.Div([
            theme.stat_row([
                theme.stat("outcome", "not scored", "no row in this report's node table"),
                theme.stat("depth", _int_cell(view.depth(view.resolve(hpo_id) or hpo_id)),
                           "in the ontology"),
                theme.stat("severed", "yes" if culprit else "no",
                           f"{culprit['hops']} hop(s) above" if culprit else "outside the walk",
                           tint="critical" if culprit else None),
            ]),
            _term_meaning(view, hpo_id),
            context,
            calls,
        ]),
        panel_id="dd-node-panel",
        subtitle="A node with no verdict of its own. What follows is the decision that kept the "
                 "traversal from ever reaching it.")


def _node_panel(bundle, subset, report: dict, report_id: str, hpo_id: str, cell_id: str,
                gold, predicted, operating_point: str | None = None):
    """The selected node: its scores, its fate, and every SLM call behind it."""
    registry = state.get_registry()
    row = _row_for(subset, registry.view, str(hpo_id))
    if row is None:
        return _unevaluated_panel(registry, bundle, subset, report, report_id, str(hpo_id),
                                  cell_id)
    record = row.iloc[0]

    header = theme.stat_row([
        theme.stat("outcome", str(record["outcome"]),
                   str(record["state"]),
                   tone={"TP": "good", "FP": "critical", "FN": "warning"}.get(record["outcome"])),
        *_score_tiles(record, operating_point),
        theme.stat("expanded", "yes" if bool(record["expanded"]) else "no",
                   f"depth {record['depth']}"),
    ])

    # The table's own spelling of the id, which may differ from the clicked one when the term was
    # recorded under an alt id. Everything keyed on the artifacts, the relations map, the call
    # records, has to be looked up with this and not with what the diagram produced.
    raw_id = str(record["hpo_id"])

    # Classified here, not read off the table: the taxonomy columns are only filled on a
    # heavy bundle, and this page must not classify a hundred other reports to explain one node.
    relation = _relations(row, set(gold), set(predicted), registry.view).get(raw_id)
    context = []
    if record["state"] == "blocked":
        context.append(theme.note([
            "Never reached. Its last surviving path was severed by ",
            html.Code(str(record.get("culprit"))),
            f", whose prune score was {_fmt(record.get('culprit_prune_score'))}, below the "
            f"τ_prune of this configuration. No τ_accept could have recovered this term.",
        ], tone="danger"))
    if relation and record["outcome"] == "FP":
        bucket, reference, distance = relation
        context.append(theme.note(
            f"False positive, classified `{bucket}`"
            + (f" against {reference}" if reference else "")
            + (f", {int(distance)} hops away" if distance is not None else "")
            + ". " + theme.FP_HELP.get(bucket, ""), tone="warn"))
    elif relation and record["outcome"] == "FN":
        context.append(theme.note(
            f"Missed. Nearby predictions: `{relation[0]}`, "
            + theme.FN_HELP.get(relation[0], ""), tone="warn"))

    calls_panel = _calls_panel(registry, bundle, cell_id, report, report_id, raw_id)
    return theme.panel(
        f"{record['hpo_label']} · {raw_id}",
        html.Div([header, *context, _term_meaning(registry.view, raw_id), calls_panel]),
        panel_id="dd-node-panel",
        subtitle="What the SLM was shown, and what it answered. `margin` is "
                 "`logit_yes − logit_no`; the verdict is `Yes` when it is positive.",
    )


def retrieved_indices(registry, cell_id: str, report_id: str, hpo_id: str | None) -> set[int]:
    """The ``sent_index`` values one node's SLM calls were made on, for marking the report.

    Empty, never an exception, when the calls artifact is absent or the node was never asked
    about, both of which are ordinary states, not errors.
    """
    if not hpo_id:
        return set()
    scan = registry.calls_scan(cell_id)
    if scan is None:
        return set()
    calls = loaders.calls_for_report(scan, str(report_id))
    if calls.empty or "hpo_id" not in calls.columns or "sent_index" not in calls.columns:
        return set()
    calls = calls.loc[calls["hpo_id"].astype(str) == str(hpo_id)]
    return {int(i) for i in calls["sent_index"].dropna()}


def _sent_badge(sent_index, resolved: bool):
    """The ``sent #N`` badge, and, where the sentence resolved, a link back to the report.

    This is the one place the two halves of the page are the same object: the index the model was
    called on, and the sentence sitting in the prose above. Clicking scrolls the report to it and
    holds it lit, so "what was the model shown" stops being a number the reader has to count down
    the page to find.

    Not a link when the sentence could not be recovered: there is nothing to scroll to, and an
    inert click reads as a broken control, not as missing data.
    """
    try:
        index = int(sent_index)
    except (TypeError, ValueError):
        return html.Span("sent #?", className="pill")
    if not resolved:
        return html.Span(f"sent #{index}", className="pill",
                         title="this index does not resolve against the recovered segmentation")
    return html.Span(f"sent #{index}", className="pill pill-link",
                     id={"type": "dd-call-sent", "idx": index}, n_clicks=0,
                     title="show this sentence in the report above")


def _calls_panel(registry, bundle, cell_id: str, report: dict, report_id: str, hpo_id: str):
    scan = registry.calls_scan(cell_id)
    if scan is None:
        return theme.note(
            "No *_calls.jsonl for this cell, so the retrieved sentences and the prompt cannot be "
            "shown. Everything else on this page is unaffected.", tone="info")

    calls = loaders.calls_for_report(scan, str(report_id))
    if calls.empty or "hpo_id" not in calls.columns:
        return theme.note(f"No SLM calls recorded for report {report_id}.", tone="info")
    calls = calls.loc[calls["hpo_id"].astype(str) == hpo_id]
    if calls.empty:
        return theme.note(
            "This node has no call records. On a synthetic row for an annotated term that was never "
            "reached that is expected, the node was never asked about.", tone="info")

    evidence = registry.evidence()
    cohort = report.get("cohort", "")
    indices = (calls["sent_index"].dropna().tolist() if "sent_index" in calls.columns else [])
    coverage = evidence.coverage(cohort, report_id, indices)
    notes = []
    if coverage["n_resolved"] == 0:
        notes.append(theme.note(
            f"Sentence text unavailable ({evidence.status(cohort, report_id)}). Showing indices, "
            "margins and logits only, everything except the words. The score store does not keep the "
            "retrieved sentence, so the text has to be recovered; see the module docstring of "
            "`apps/treephenorag_ui/evidence.py` for the five routes it tries.", tone="warn"))
    elif not coverage["complete"]:
        notes.append(theme.note(
            f"Only {coverage['n_resolved']}/{coverage['n_referenced']} sentence indices resolve "
            f"against the {coverage['n_sentences']}-sentence segmentation of this report "
            f"({evidence.source(cohort)}). The segmentation used here may differ from the one "
            "inference used, treat the displayed text with suspicion.", tone="danger"))

    rows = []
    for record in calls.sort_values(["ctx_type", "rank"] if "ctx_type" in calls.columns
                                    else ["rank"]).itertuples():
        sentence = evidence.sentence(cohort, report_id, getattr(record, "sent_index", None))
        prompt = evidence.prompt(hpo_id, sentence) if sentence else None
        verdict = str(getattr(record, "verdict", ""))
        margin = getattr(record, "margin", None)
        rows.append(html.Div([
            html.Div([
                html.Span(f"rank {getattr(record, 'rank', '?')}", className="pill"),
                html.Span(getattr(record, "ctx_type", "") or "", className="pill")
                if getattr(record, "ctx_type", None) else None,
                html.Span(f"cos {_fmt(getattr(record, 'cosine_sim', None))}", className="pill"),
                html.Span(f"margin {_fmt(margin)}", className="pill",
                          title="logit_yes − logit_no. Positive is a Yes; the magnitude is how "
                                "far from the decision boundary the model was."),
                html.Span(verdict, className="pill",
                          style={"color": theme.palette()["good"] if verdict == "Yes"
                                 else theme.palette()["text_muted"]}),
                _sent_badge(getattr(record, "sent_index", None), bool(sentence)),
            ], className="panel-tools"),
            html.Div(sentence or html.I("sentence text unavailable"), className="report-text"),
            html.Details([
                html.Summary("prompt and logits", className="details-summary"),
                html.Pre(prompt or "(prompt needs the sentence text)", className="mono"),
                html.Div(
                    f"logit_yes {_fmt(getattr(record, 'logit_yes', None), 4)} · "
                    f"logit_no {_fmt(getattr(record, 'logit_no', None), 4)} · "
                    f"logsumexp {_fmt(getattr(record, 'logsumexp_all', None), 4)} · "
                    f"top1_token {getattr(record, 'top1_token_id', '?')}", className="kv"),
            ]),
        ], className="panel"))

    return html.Div(notes + rows)


# ── node picker ──────────────────────────────────────────────────────────────

def _node_options(subset, filter_mode: str, view=None) -> list[dict]:
    """The node picker's options, each hovering to what the term means.

    A dropdown of 600 ids is the one place on this page where a reader picks a term *before* seeing
    anything about it, so it is the place a definition is worth most. ``dcc.Dropdown`` renders a
    component label, so the option carries a ``title``; ``search`` is set explicitly because a
    component label is not a string and Dash has nothing to match typing against otherwise.
    """
    if subset is None or subset.empty:
        return []
    frame = subset
    if filter_mode == "terms":
        frame = subset.loc[subset["outcome"].isin(["TP", "FP", "FN"])]
    elif filter_mode == "errors":
        frame = subset.loc[subset["outcome"].isin(["FP", "FN"])]
    elif filter_mode == "accepted":
        frame = subset.loc[subset["accepted"].fillna(False).astype(bool)]
    elif filter_mode == "gold":
        frame = subset.loc[subset["in_gold"].fillna(False).astype(bool)]
    elif filter_mode == "culprits":
        culprits = set(subset["culprit"].dropna().astype(str))
        frame = subset.loc[subset["hpo_id"].astype(str).isin(culprits)]

    frame = frame.head(600)
    options = []
    for record in frame.itertuples():
        label = record.hpo_label if isinstance(record.hpo_label, str) else record.hpo_id
        badge = record.outcome if record.state == "visited" else record.state
        options.append(_node_option(badge, label, str(record.hpo_id), view))
    return options


def _node_option(badge: str, label: str, hpo_id: str, view=None) -> dict:
    """One picker option. Plain text without a *view*, so a caller with no ontology still works."""
    text = f"[{badge}] {label} · {hpo_id}"
    if view is None:
        return {"label": text, "value": hpo_id}
    return {"label": html.Span(text, title=terms.tooltip(view, hpo_id)),
            "value": hpo_id, "search": text}


def with_selected(options: list[dict], selected, subset, view) -> list[dict]:
    """*options* plus the current selection, if the filter excluded it but the table has it.

    Clicking the ontology diagram can select a node the Node dropdown's filter does not list, an
    accepted term while the filter reads "errors only", say. Dash renders a value with no matching
    option as an empty box, so the control would go blank while the panel below it kept describing
    the node. Adding the option back is the only way to keep the two telling the same story.
    """
    if not selected or any(o["value"] == str(selected) for o in options):
        return options
    row = _row_for(subset, view, str(selected))
    if row is None:
        return options
    record = row.iloc[0]
    label = record["hpo_label"] if isinstance(record["hpo_label"], str) else record["hpo_id"]
    badge = record["outcome"] if record["state"] == "visited" else record["state"]
    return [_node_option(badge, label, str(selected), view), *options]


def step_to(trigger, current, values: list):
    """The neighbour of *current* in *values*, in the direction *trigger* names.

    Clamped at both ends, not wrapping: a reader stepping through a cohort wants to know
    when they have reached the end of it, and silently landing back on the first report reads as a
    navigation they did not ask for. Raises ``PreventUpdate`` when the move would change nothing,
    so the arrow at an end is inert as well as greyed out. Modelled on
    ``apps/phenojury_ui/views/patient.py:step_to``.
    """
    step = {"dd-report-prev": -1, "dd-report-next": 1}.get(trigger)
    if step is None or not values:
        raise PreventUpdate
    if current not in values:
        return values[0]
    index = values.index(current)
    moved = min(max(index + step, 0), len(values) - 1)
    if moved == index:
        raise PreventUpdate
    return values[moved]


def _apply_subset(registry, cell_id, subset, report_ids):
    """Narrow *report_ids* to the chosen subset, if one applies to this cell.

    Never narrows to nothing: a subset whose reports are all absent from this run leaves the list
    alone, not emptying the page. That happens whenever the frame is opened against a run
    that processed a different cohort or an unmerged shard, and a blank dropdown does not say so.
    """
    chosen = registry.subset_ids(cell_id, (subset or {}).get("mode"), (subset or {}).get("cell"))
    if chosen is None:
        return report_ids
    keep = [r for r in report_ids if r in set(chosen)]
    return keep or report_ids


def _report_label(frame, report_id: str) -> str:
    """``1003450 · pb_worst`` when the report is in the sample, otherwise just the id."""
    cell = frame.label(report_id) if frame is not None else ""
    return f"{report_id} · {cell}" if cell else str(report_id)


def _option_values(options) -> list:
    """The values of a Dash dropdown's options, tolerating the plain-string shorthand."""
    return [o["value"] if isinstance(o, dict) else o for o in (options or [])]


def _trigger_id():
    """``callback_context.triggered_id``, or ``None`` outside a callback.

    The unit tests call every callback body directly, with no browser and therefore no callback
    context. Reading the trigger defensively is what lets the stepper be exercised there like
    everything else on this page, not being the one callback nothing checks.
    """
    try:
        return callback_context.triggered_id
    except Exception:
        return None


def clicked_hpo(click_data) -> str | None:
    """The HPO id behind a click on the ontology diagram, or ``None``.

    Lifted out of the callback so it is testable without a browser: the shape Plotly sends is
    nested three deep and a change in it would otherwise only show up as a diagram whose clicks
    stopped doing anything.
    """
    points = (click_data or {}).get("points") or []
    if not points:
        return None
    custom = points[0].get("customdata")
    if isinstance(custom, (list, tuple)) and custom:
        return str(custom[0])
    return None


def register(app) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("store-sentence", "data"),
        Input({"type": "dd-call-sent", "idx": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _focus_sentence(clicks=None):
        """A ``sent #N`` badge in the calls panel points back at the report above.

        The default is for the test harness, which supplies arguments only for string component
        ids and would otherwise call this one with none. No clicks is the no-op path anyway.

        The store is what the clientside scroll listens on. Writing the index alone would make a
        second click on the same badge a no-op, Dash does not fire when the value is unchanged, so the click count rides along and is stripped on the other side.
        """
        # The click guard comes before the context is read, not after. Dash fires this once on
        # every redraw of the calls panel, with every count at zero, and ``triggered_id`` is only
        # reachable from inside a real callback, so touching it first turns the ordinary
        # nothing-happened case into an exception.
        if not any(clicks or []):
            raise PreventUpdate
        triggered = callback_context.triggered_id
        if not triggered:
            raise PreventUpdate
        # The click total rides along because Dash does not fire a callback when the value is
        # unchanged: without it, clicking the same badge twice, having scrolled away in between,
        # which is when a reader clicks it again, would do nothing the second time.
        return {"idx": triggered["idx"], "n": sum(c or 0 for c in clicks)}

    @app.callback(
        Output("dd-report", "options"),
        Output("dd-report", "value"),
        Input("run-a", "value"), Input("op-a", "value"), Input("store-focus", "data"),
        Input("store-subset", "data"), active.TAB_INPUT, State("dd-report", "value"),
    )
    def _reports(cell_id, operating_point, focus, subset, tab, current):
        active.guard(tab, "deepdive")
        if not cell_id:
            return [], None
        registry = state.get_registry()
        # report_ids, not bundle: this callback fires on every run/op change and only needs a list
        # of ids. Going through bundle() built the whole cohort node table to fill a dropdown.
        report_ids = sorted(registry.report_ids(cell_id, operating_point))
        report_ids = _apply_subset(registry, cell_id, subset, report_ids)
        # The value stays the bare report id: _report_frame matches on it, store-focus carries it,
        # and the stepper compares against it. Only the label carries the cell.
        frame = registry.sample
        options = [{"label": _report_label(frame, r), "value": r} for r in report_ids]
        wanted = (focus or {}).get("report_id")
        if wanted in report_ids:
            return options, wanted
        if current in report_ids:
            return options, current
        return options, (report_ids[0] if report_ids else None)

    @app.callback(
        Output("dd-report", "value", allow_duplicate=True),
        Input("dd-report-prev", "n_clicks"), Input("dd-report-next", "n_clicks"),
        active.TAB_INPUT, State("dd-report", "value"), State("dd-report", "options"),
        prevent_initial_call=True,
    )
    def _step(_prev, _next, tab, current, options):
        active.guard(tab, "deepdive")
        return step_to(_trigger_id(), current, _option_values(options))

    @app.callback(
        Output("dd-report-prev", "disabled"),
        Output("dd-report-next", "disabled"),
        Input("dd-report", "value"), Input("dd-report", "options"), active.TAB_INPUT,
    )
    def _step_bounds(current, options, tab):
        active.guard(tab, "deepdive")
        values = _option_values(options)
        if not values or current not in values:
            return True, True
        index = values.index(current)
        return index == 0, index == len(values) - 1

    @app.callback(
        Output("dd-summary", "children"),
        Output("dd-terms", "children"),
        Output("dd-walk", "children"),
        Input("run-a", "value"), Input("op-a", "value"),
        Input("dd-report", "value"), Input("store-theme", "data"), active.TAB_INPUT,
    )
    def _overview(cell_id, operating_point, report_id, mode, tab):
        active.guard(tab, "deepdive")
        blank = html.Div()
        if not cell_id or not report_id:
            return theme.empty("Select a run and a report."), blank, blank
        registry = state.get_registry()
        report = registry.get_report(cell_id, operating_point) or {}
        bundle, subset = _report_frame(cell_id, operating_point, report_id)
        if subset is None or subset.empty:
            return theme.empty(f"No node records for {report_id}."), blank, blank

        gold = set((bundle or {}).get("gold") or ())
        predicted = set((bundle or {}).get("predicted") or ())
        cell_panel = _cell_panel(registry, str(report_id))
        drawn = len(subset.loc[subset["state"] == "visited"])
        walk = theme.panel(
            "The walk",
            dcc.Graph(id="dd-walk-graph",
                      figure=_walk_figure(subset, mode or "light", registry.view),
                      config=theme.GRAPH_CONFIG),
            panel_id="dd-walk-panel", graph_id="dd-walk-graph",
            subtitle=("Each scored node by depth and accept score; a dark ring means it was "
                      "expanded." if drawn <= MAX_NODES_DRAWN else
                      f"{drawn} nodes were scored, too many to draw individually, so this is "
                      "the depth profile by outcome."),
        )
        # The cell panel goes above the scorecard tiles: why this report is on screen is context
        # for its numbers, not a footnote to them. It is None for anything outside the sample.
        summary = _summary(subset, report, report_id, gold, predicted)
        return (html.Div([cell_panel, summary]) if cell_panel else summary,
                _terms_panel(subset, gold, predicted, registry.view, str(report_id),
                             registry.curated_for(report.get("cohort", ""))),
                walk)

    @app.callback(
        Output("dd-tree-graph", "figure"),
        Output("dd-tree-note", "children"),
        Input("run-a", "value"), Input("op-a", "value"),
        Input("dd-report", "value"), Input("dd-node", "value"),
        Input("store-theme", "data"), active.TAB_INPUT,
    )
    def _tree(cell_id, operating_point, report_id, hpo_id, mode, tab):
        active.guard(tab, "deepdive")
        if not cell_id or not report_id:
            return go.Figure(), html.Div()
        registry = state.get_registry()
        _, subset = _report_frame(cell_id, operating_point, report_id)
        if subset is None or subset.empty:
            return go.Figure(), html.Div()
        terms = subgraph.terms_of_report(subset)
        built = subgraph.build(terms, registry.view)
        labels = {str(r.hpo_id): str(r.hpo_label) for r in subset.itertuples()}
        selected = registry.view.resolve(hpo_id) if hpo_id else None
        return (_tree_figure(built, labels, selected, mode or "light", registry.view),
                _tree_note(built, registry.view, selected))

    @app.callback(
        Output("dd-node", "options"),
        Output("dd-node", "value"),
        Input("run-a", "value"), Input("op-a", "value"), Input("dd-report", "value"),
        Input("dd-node-filter", "value"), Input("store-focus", "data"),
        active.TAB_INPUT, State("dd-node", "value"),
    )
    def _nodes(cell_id, operating_point, report_id, filter_mode, focus, tab, current):
        active.guard(tab, "deepdive")
        if not cell_id or not report_id:
            return [], None
        registry = state.get_registry()
        _, subset = _report_frame(cell_id, operating_point, report_id)
        options = _node_options(subset, filter_mode or "terms", registry.view)
        values = {o["value"] for o in options}
        wanted = (focus or {}).get("hpo_id")
        if wanted in values:
            return options, wanted
        if current in values:
            return options, current
        options = with_selected(options, current, subset, registry.view)
        keep = current if any(o["value"] == str(current) for o in options) else None
        return options, keep

    @app.callback(
        Output("dd-node", "value", allow_duplicate=True),
        Input("dd-tree-graph", "clickData"), active.TAB_INPUT,
        prevent_initial_call=True,
    )
    def _click(click_data, tab):
        active.guard(tab, "deepdive")
        hpo_id = clicked_hpo(click_data)
        if hpo_id is None:
            raise PreventUpdate
        return hpo_id

    @app.callback(
        Output("dd-evidence", "children"),
        Output("dd-text", "children"),
        Input("run-a", "value"), Input("op-a", "value"),
        Input("dd-report", "value"), Input("dd-node", "value"),
        Input("store-theme", "data"), active.TAB_INPUT,
    )
    def _evidence(cell_id, operating_point, report_id, hpo_id, mode, tab):
        active.guard(tab, "deepdive")
        if not cell_id or not report_id:
            raise PreventUpdate
        registry = state.get_registry()
        report = registry.get_report(cell_id, operating_point) or {}
        bundle, subset = _report_frame(cell_id, operating_point, report_id)
        gold = set((bundle or {}).get("gold") or ())
        predicted = set((bundle or {}).get("predicted") or ())
        highlighted = retrieved_indices(registry, cell_id, str(report_id), hpo_id)
        text_panel = _text_panel(registry, report, report_id, gold, highlighted,
                                 mode or "light", predicted)
        if not hpo_id or subset is None:
            return (theme.empty("Pick a node, in the diagram above, or from the Node dropdown, "
                                "to see what the model was shown."), text_panel)
        return (_node_panel(bundle, subset, report, report_id, hpo_id, cell_id, gold, predicted,
                            operating_point),
                text_panel)
