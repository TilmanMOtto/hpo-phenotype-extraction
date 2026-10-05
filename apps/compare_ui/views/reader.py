"""The reader: the ground truth on the left, the report once in the middle, five verdicts on the right.

One copy of the report, not five. The layout the sketch settled on is a grid whose rows are the
report's own segments, so that reading across a row answers the question the app exists for --
*this sentence, these words: who got it and who did not* -- without reading the sentence five
times. The ground truth column carries the curated answer for that segment. The five gutters carry each
method's verdict on it.

Three choices inside that are worth stating, because each has a plausible alternative that is
worse:

**The ground truth is drawn in the text on a recorded span, or not at all.** On HCY that span comes from
``CuratedGold.anchors``, which searches only inside the segment the curator named. On GSC+ it comes
from the corpus's own mention offsets, or failing that from RAG-HPO's annotation wording found
verbatim -- the ladder in :mod:`apps.compare_ui.gold`, whose rungs the row's ``source`` names. Either
way an underline is something somebody wrote down, and an annotated term nobody placed gets a chip in the
ground truth column instead of an underline somewhere plausible.

**Every row of a gutter is a chip, including the misses.** A method's column shows ``✓`` for the
annotated terms it found, ``✗`` for what it predicted that the ground truth does not carry, and ``–`` for the
annotated terms it missed -- on the same row as the ground truth, so the eye compares. A column that showed only
predictions would make a method look better the less it said.

**Nothing is dropped for not fitting.** Ground truth with no located segment, and predictions no method
could place on one, go in a footer row under the grid rather than out of the bundle. Evidence that
exists and cannot be positioned is still evidence.
"""

from __future__ import annotations

from dash import dcc, html

from apps.compare_ui import theme
from apps.compare_ui.views import common

VIEW_ID = "reader"

#: Documents longer than this get a truncation note, not a page of DOM. HCY reports are a
#: dozen sentences and a GSC+ abstract fewer. This is a guard against a pathological one, not a
#: paging scheme.
MAX_SEGMENTS = 400


def layout():
    """The Dash layout of this view."""
    return html.Div([
        html.Div(id="reader-head"),
        html.Div(id="reader-grid"),
        html.Div(id="reader-detail"),
        dcc.Store(id="sig-reader", data=None),
    ])


# -- the header --------------------------------------------------------------

def render_head(registry, bundle, subset_note: str = ""):
    """Report identity, provenance, and every warning that must travel with this screen."""
    if bundle is None:
        return theme.empty("No bundle for this report.")

    row = registry.rows().get(bundle["report_id"]) or {}
    cell = bundle.get("cell") or row.get("cell") or ""

    bits = [html.Span(bundle["report_id"], className="cmp-report-id")]
    if cell:
        bits.append(html.Span(cell, className="pill pill-cell",
                              title="the deep-dive cell this report was drawn into -- the reason "
                                    "it is on screen at all"))
    bits.append(html.Span("{} annotated term(s) · {} segment(s)".format(
        len(bundle.get("gold") or ()), len(bundle.get("segments") or ())),
        className="cmp-report-sub"))
    if bundle.get("gold_dataset"):
        bits.append(html.Span(bundle["gold_dataset"], className="pill",
                              title="the curated ground truth dataset this report is scored against"))

    notes = []
    if subset_note:
        notes.append(theme.note(subset_note, "warn"))
    gold_note = registry.gold_note()
    if gold_note:
        # A property of the ground truth, so it belongs on every screen that draws it, not in the
        # Provenance tab a reader may never open. HCY's curated ground truth needs no such sentence: its
        # annotations carry their own position, which is what the underline already says.
        notes.append(theme.note(gold_note, "info"))
    drift = registry.drift(bundle["report_id"])
    if drift:
        notes.append(theme.note(
            "These artifacts have changed since the bundle was built: " + ", ".join(drift)
            + ". Rebuild with slurm/compare_ui_bundles.sbatch before quoting anything here.",
            "danger"))

    return html.Div([html.Div(bits, className="cmp-report-head")] + notes)


# -- the grid ----------------------------------------------------------------

def render_grid(registry, bundle, selection=None):
    """The whole reading surface: a header row, one row per segment, and a footer of the unplaced.

    **One grid, cells as direct children** -- not a grid of rows that are themselves grids. That
    was the first attempt and it is worth naming, because it renders as a pile, not as an
    error: a ``display: grid`` container places each *child* into one of its own columns, so seven
    consecutive rows were laid into the seven columns of a single grid row, on top of each other.
    Flat cells also get the alignment for free -- column widths are shared across the whole grid
    while each row auto-sizes to its own tallest cell, which is the behaviour a row-per-
    subgrid was trying to fake.
    """
    if bundle is None:
        return theme.empty("No bundle for this report.")

    methods = [m for m in registry.methods() if m["key"] in (bundle.get("methods") or {})]
    segments = bundle.get("segments") or []
    truncated = len(segments) > MAX_SEGMENTS

    gold_by_segment, gold_loose = _by_segment(bundle.get("gold") or ())
    # Only the by-segment half: a method's unplaced marks are reachable under the ``None`` key,
    # which ``_by_segment`` fills as well as the loose list.
    marks = {m["key"]: _by_segment(bundle["methods"][m["key"]].get("marks") or ())[0]
             for m in methods}

    cells = list(_header_cells(registry, methods))
    for segment in segments[:MAX_SEGMENTS]:
        cells.extend(_segment_cells(bundle, segment, methods, gold_by_segment, marks, selection))

    tail = (bundle.get("tail") or "").strip()
    if tail:
        cells.append(html.Div(html.Span(tail, className="cmp-gap"),
                              className="cmp-cell cmp-span-all"))

    cells.extend(_footer_cells(bundle, methods, gold_loose, marks, selection))

    grid = html.Div(cells, className="cmp-grid",
                    style={"gridTemplateColumns": _columns(len(methods))})

    extras = [common.outcome_legend()]
    if truncated:
        extras.append(theme.note(
            "Showing the first {} of {} segments.".format(MAX_SEGMENTS, len(segments)), "warn"))
    return html.Div([grid] + extras)


def _columns(n_methods: int) -> str:
    """The grid template: a fixed ground truth column, an elastic report, then one narrow gutter each.

    ``minmax`` floors, not fixed widths: the report column takes whatever is left, and when
    the window is narrower than the floors the grid overflows its own scrollport instead of
    squeezing five gutters into an unreadable smear.
    """
    return "minmax(10rem, 13rem) minmax(20rem, 1fr) " + " ".join(["6.5rem"] * n_methods)


def _header_cells(registry, methods):
    title = ("the terms a curator recorded for this report, with the words they read them at"
             if registry.cohort.gold_kind == "curated"
             else registry.gold_note() or "the terms annotated for this document")
    cells = [
        html.Div("Ground truth" if registry.cohort.gold_kind != "curated" else "Curated ground truth",
                 className="cmp-cell cmp-head", title=title),
        html.Div("Report", className="cmp-cell cmp-head"),
    ]
    for method in methods:
        title = method.get("label", "")
        if method.get("evidence"):
            title += "\n\nShows: " + method["evidence"]
        if method.get("evidence_note"):
            title += "\n\nCaveat: " + method["evidence_note"]
        cells.append(html.Div(
            method.get("short") or method["key"], className="cmp-cell cmp-head cmp-head-method",
            title=title, **{"data-family": method.get("family", "")}))
    return cells


def _segment_cells(bundle, segment, methods, gold_by_segment, marks, selection):
    idx = segment["idx"]
    gold_here = gold_by_segment.get(idx) or []

    cells = [
        html.Div(
            [_gold_entry(bundle, row, selection) for row in gold_here] or
            [html.Span("·", className="cmp-blank", title="no curated ground truth in this sentence")],
            className="cmp-cell cmp-cell-gold"),
        html.Div(_prose(segment, gold_here), className="cmp-cell cmp-cell-text",
                 id={"type": "cmp-seg", "idx": idx}, title="sentence #{}".format(idx)),
    ]
    for method in methods:
        here = (marks.get(method["key"]) or {}).get(idx) or []
        cells.append(html.Div(
            [_mark_chip(bundle, method["key"], mark, selection) for mark in here] or
            [html.Span("·", className="cmp-blank")],
            className="cmp-cell cmp-cell-method"))
    return cells


def _footer_cells(bundle, methods, gold_loose, marks, selection):
    """Everything that belongs to the report but to no sentence.

    Rendered as a row of the same grid, not as a separate list, so the columns still line
    up: a term nobody could place is still a term one method predicted and another did not.
    """
    loose_marks = {m["key"]: (marks.get(m["key"]) or {}).get(None) or [] for m in methods}
    if not gold_loose and not any(loose_marks.values()):
        return []

    cells = [
        html.Div([html.Div("not located", className="cmp-foot-label")]
                 + [_gold_entry(bundle, row, selection) for row in gold_loose],
                 className="cmp-cell cmp-cell-gold cmp-foot"),
        html.Div("Terms with no sentence to sit on, nobody recorded where the annotated term was "
                 "read, or the method recorded no position.",
                 className="cmp-cell cmp-cell-text cmp-foot cmp-foot-note"),
    ]
    for method in methods:
        cells.append(html.Div(
            [_mark_chip(bundle, method["key"], mark, selection)
             for mark in loose_marks[method["key"]]] or [html.Span("·", className="cmp-blank")],
            className="cmp-cell cmp-cell-method cmp-foot"))
    return cells


# -- cells -------------------------------------------------------------------

def _gold_entry(bundle, row, selection):
    hpo_id = row["hpo_id"]
    selected = bool(selection) and selection.get("hpo") == hpo_id
    return html.Div([
        common.term_chip(bundle, hpo_id, method="gold"),
        html.Div(hpo_id, className="cmp-code"),
        html.Div("“{}”".format(row["trigger"]), className="cmp-trigger") if row.get("trigger")
        else html.Div("no trigger word recorded", className="cmp-trigger cmp-blank"),
        html.Div(common.qualifier_chips(row.get("qualifiers")), className="cmp-quals"),
    ], className="cmp-gold-entry" + (" is-selected" if selected else ""))


def _mark_chip(bundle, method_key: str, mark, selection):
    selected = (bool(selection) and selection.get("hpo") == mark["hpo_id"]
                and selection.get("method") == method_key)
    chip = common.term_chip(bundle, mark["hpo_id"], outcome=mark["outcome"], method=method_key)
    if selected:
        chip.className = chip.className + " is-selected"
    return chip


def _prose(segment, gold_here):
    """The segment, with the ground truth underlined on the words the curator read it at.

    Overlaps are resolved by dropping the loser, not by nesting: two annotations on
    overlapping spans would otherwise produce interleaved tags that render as garbage. The widest
    span wins, and the dropped term is still a chip in the ground truth column, so nothing disappears.
    """
    children = []
    gap = segment.get("gap_before") or ""
    if gap.strip():
        children.append(html.Span(gap.strip(), className="cmp-gap"))

    text = segment.get("text") or ""
    spans = _resolve_overlaps([
        (row["span"][0], row["span"][1], row) for row in gold_here
        if row.get("span") and 0 <= row["span"][0] < row["span"][1] <= len(text)])

    cursor = 0
    for start, end, row in spans:
        if start > cursor:
            children.append(text[cursor:start])
        children.append(html.Mark(
            text[start:end], className="cmp-mark",
            title="curated ground truth: {}, {}".format(row["hpo_id"], row.get("source") or "")))
        cursor = end
    children.append(text[cursor:])
    return children


def _resolve_overlaps(spans):
    """Non-overlapping spans in document order. The widest wins a clash."""
    kept = []
    for start, end, row in sorted(spans, key=lambda s: (s[0], -(s[1] - s[0]))):
        if kept and start < kept[-1][1]:
            if (end - start) > (kept[-1][1] - kept[-1][0]):
                kept[-1] = (start, end, row)
            continue
        kept.append((start, end, row))
    return kept


def _by_segment(rows):
    """``({segment_idx: [row]}, [rows with no segment])`` -- the grid's two destinations."""
    placed: dict = {}
    loose = []
    for row in rows:
        idx = row.get("segment_idx")
        if idx is None:
            loose.append(row)
            placed.setdefault(None, []).append(row)
        else:
            placed.setdefault(int(idx), []).append(row)
    return placed, loose


# -- callbacks ---------------------------------------------------------------

def register(app) -> None:
    """Register this view's callbacks on *app*."""
    from dash import ALL, Input, Output, State, ctx
    from dash.exceptions import PreventUpdate

    from apps.compare_ui import state
    from apps.compare_ui.views import reasons

    @app.callback(
        Output("reader-head", "children"),
        Output("reader-grid", "children"),
        Output("sig-reader", "data"),
        Input("tabs", "value"),
        Input("cohort-picker", "value"),
        Input("report-picker", "value"),
        Input("store-selection", "data"),
        Input("store-theme", "data"),
        State("sig-reader", "data"),
    )
    def render(tab, cohort, report_id, selection, mode, previous):
        key = common.render_key(VIEW_ID, cohort, report_id, selection, mode)
        if not common.should_render(VIEW_ID, tab, key, previous):
            raise PreventUpdate
        registry = state.registry(cohort)
        bundle = registry.bundle(report_id) if report_id else None
        head = common.guard(VIEW_ID, "header", render_head, registry, bundle,
                            _subset_note(registry, report_id))
        grid = common.guard(VIEW_ID, "grid", render_grid, registry, bundle, selection)
        return head, grid, key

    @app.callback(
        Output("reader-detail", "children"),
        Input("store-selection", "data"),
        Input("cohort-picker", "value"),
        Input("report-picker", "value"),
        Input("tabs", "value"),
    )
    def detail(selection, cohort, report_id, tab):
        if tab != VIEW_ID:
            raise PreventUpdate
        registry = state.registry(cohort)
        bundle = registry.bundle(report_id) if report_id else None
        return common.guard(VIEW_ID, "reasoning", reasons.render, bundle, selection)

    @app.callback(
        Output("store-selection", "data"),
        Input({"type": "cmp-chip", "method": ALL, "hpo": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def select(clicks):
        """One callback for every chip on the page, resolved by which id fired.

        ``ALL``, not ``MATCH`` because the target is a single shared store: ``MATCH`` would
        need one output per chip, and the selection is one thing, not three hundred.
        """
        triggered = ctx.triggered_id
        if not triggered or not any(clicks or ()):
            raise PreventUpdate
        return {"method": triggered.get("method"), "hpo": triggered.get("hpo")}


def _subset_note(registry, report_id) -> str:
    """The sentence the sampling frame requires on every screen it selected the contents of.

    It is not decoration. Both frames draw their documents **on the outcome being examined** --
    HCY's cells include PhenoBERT's own best and worst per-report scores, and the GSC+ cells are
    defined on which of PhenoJury, RAG-HPO and AutoPCR got a term right -- so any rate over them is
    a rate over the selection rule. This app is for mechanisms with a document id attached, and the
    banner is what keeps a number read here from being quoted as a result.
    """
    frame = registry.index.get("frame") or {}
    if not frame:
        return ""
    cell = registry.cell_of(report_id) if report_id else ""
    n = len(registry.report_ids())
    return ("Purposive sample: these {} {} were drawn into cells defined on the outcome "
            "(this one is “{}”). No aggregate over them may be quoted, the cohort-wide numbers "
            "are on the Scorecard tab.".format(n, registry.unit(n), cell or "unnamed"))
