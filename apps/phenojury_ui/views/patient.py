"""Report deep dive, one report, read top to bottom.

Every other view aggregates. This one does not. It is the screen that turns a number into a claim
you can check, and it is laid out in the order the checking happens:

1. **the score** for this report at the current configuration;
2. **the report as a document**, with every phrase the PhenoBERT baseline (the PhenoBERT baseline) matched
   underlined in place, and the annotated terms listed beneath it;
3. **the terms in play**, predicted, annotated, or both, with who voted, why each error is the
   error it is, and what the baseline said about the same term;
4. **the evidence**, sentence by sentence: pick a sentence in the report and read what each model
   wrote about it, annotated the same way the report is.

The report and the replies go through **one annotator** (:func:`_annotate`), so they are not merely
similar, they are the same rendering with different input. Each match is the matched phrase,
underlined, with an inline tag naming the HPO term it normalised to. The underline alone says
*where* PhenoBERT found something. The tag says *what it became*, which is the half that decides
whether the extraction was right, a model writing "Abnormality of the spinal cord" and the linker
resolving it to "Focal-onset seizure" is the failure this page exists to make visible, and it is
invisible if the term is only on a hover.

Three details make the highlighting trustworthy rather than approximate:

* offsets in the **replies** come from PhenoBERT's own output, translated through a byte-exact
  rebuild of the input file it read (``detections.build_replies`` / ``verify.gate_spans``), no
  fuzzy string search, so a phrase occurring twice highlights the occurrence actually detected;
* they index the **stripped** reply, because ``run_phenobert_per_model`` strips reasoning blocks
  before writing its input. The raw reply is available behind a toggle, and highlighting it with
  these offsets would be off by the length of the ``<think>`` block, so it is not attempted;
* offsets in the **report** come from the PhenoBERT baseline and index the verbatim staged report, verified at
  load by ``verify.gate_pb_standalone``. When that check fails the panel refuses to draw, not underlining the wrong words.

Clicking a term in "Terms in play" filters the sentence list to the sentences that term has
anything to do with, grounded by a model, or matched by the baseline. That is the one question
this page could not answer before: not "what happened in this report", but "what happened to *this
term* in this report".

The prompt shown is imported from ``hpo_extraction.phenojury.generation``, not transcribed, the pipeline
does not persist prompts, and a transcribed one drifts.
"""

from __future__ import annotations

import dash
from dash import ALL, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from .. import autopsy, relations, theme, votes
from hpo_extraction.curation import phenobert_output as pbstandalone
from ..detections import popcount
from . import common

VIEW_ID = "patient"

#: Reply blocks are the heaviest thing this app draws, one per (sentence x model), each carrying
#: The model's whole generation. Most sentences in a clinical report produce nothing from anybody,
#: so the default is to draw only the ones that did, and to say how many were left out.
SHOW_ALL = "all"


def layout():
    """The Dash layout of this view."""
    return html.Div([
        # Which sentence is open, and which term the sentence list is filtered to. View-local
        #, not app-level: nothing outside this page has an opinion about either.
        dcc.Store(id="pa-sent-focus", data=None),
        dcc.Store(id="pa-term-focus", data=None),
        html.Div([
            html.Label("Report", className="field-label"),
            html.Div([
                html.Button("◀", id="pa-prev", className="copy-btn step-btn",
                            title="previous report"),
                html.Div(dcc.Dropdown(id="pa-report", options=[], value=None, clearable=False,
                                      placeholder="pick a report"),
                         style={"flex": "1 1 auto"}),
                html.Button("▶", id="pa-next", className="copy-btn step-btn",
                            title="next report"),
            ], className="stepper"),
            dcc.Checklist(id="pa-sentences", options=[
                {"label": " show every sentence, including the ones no model wrote about",
                 "value": SHOW_ALL}], value=[]),
            dcc.Checklist(id="pa-raw", options=[{"label": " show raw replies (with reasoning "
                                                          "blocks; spans are not highlighted)",
                                                 "value": "raw"}], value=[]),
        ], className="field"),
        html.Div(id="pa-body"),
    ], className="view")


def register(app) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("pa-report", "options"),
        Output("pa-report", "value"),
        Input("tabs", "value"),
        Input("run-a", "value"),
        Input("store-config", "data"),
        Input("store-focus", "data"),
        State("pa-report", "value"),
    )
    def picker(tab, run_a, config, focus, current):
        """Reports in numeric order, each labelled with its F1.

        Numeric, not lexicographic: GSC ids are variable-length digit strings, so ``10051003``
        sorts before ``1003450`` under a plain string sort and the list reads as scrambled. The F1
        stays in the label, it is what makes a report worth opening, but it no longer decides the
        order, because the order is what the ◀ ▶ buttons step through and stepping through
        "worst first" is not a traversal of anything.

        Gated on the tab like the body: the option list is one entry per report in the cohort, and
        rebuilding it for a screen nobody is looking at is a payload for nothing.
        """
        if tab != VIEW_ID:
            raise PreventUpdate
        bundle = common.bundle_for(run_a)
        if bundle is None:
            return [], None
        cfg = common.resolve(config, bundle)
        ordered = ordered_reports(bundle, cfg)
        options = [{"label": f"{r}, F1 {common.num(f1)}" if f1 is not None else f"{r}, no ground truth",
                    "value": r} for r, f1 in ordered]
        known = {r for r, _ in ordered}
        wanted = focus if focus in known else None
        if wanted is None:
            wanted = current if current in known else (ordered[0][0] if ordered else None)
        # Writing the same value back would still count as a change downstream, and both focus
        # stores clear on `pa-report`, so leaving and returning to this tab would silently drop a
        # term filter the reader set. no_update leaves it alone.
        return options, (dash.no_update if wanted == current else wanted)

    @app.callback(
        Output("pa-report", "value", allow_duplicate=True),
        Input("pa-prev", "n_clicks"),
        Input("pa-next", "n_clicks"),
        State("pa-report", "value"),
        State("pa-report", "options"),
        prevent_initial_call=True,
    )
    def step(_prev, _next, current, options):
        """◀ ▶ walk the option list. Clamped at both ends.

        The current value is a ``State``, not an ``Input``: a component prop that is both the
        input and the output of one callback is a cycle Dash refuses to register. Which end of the
        list we are at is answered separately, by :func:`step_bounds`.
        """
        values = [o["value"] for o in (options or [])]
        if not values:
            raise PreventUpdate
        # At an end this writes back the value that is already there. That is deliberate: the body
        # callback keys off `render_key`, so an unchanged value produces an unchanged key and it
        # redraws nothing, and a callback that always runs is one the selftest can always check.
        return step_to(dash.ctx.triggered_id, current, values) or values[
            values.index(current) if current in values else 0]

    @app.callback(
        Output("pa-prev", "disabled"),
        Output("pa-next", "disabled"),
        Input("pa-report", "value"),
        Input("pa-report", "options"),
    )
    def step_bounds(current, options):
        """Grey out the arrow that would do nothing."""
        values = [o["value"] for o in (options or [])]
        if not values or current not in values:
            return True, True
        index = values.index(current)
        return index == 0, index == len(values) - 1

    @app.callback(
        Output("pa-term-focus", "data"),
        Input({"type": "pa-term", "hpo": ALL}, "n_clicks"),
        Input("pa-term-clear", "n_clicks"),
        Input("pa-report", "value"),
        State("pa-term-focus", "data"),
        prevent_initial_call=True,
    )
    def focus_term(_clicks, _clear, _report, current):
        """Clicking a term in the table filters the sentences below it. Clicking it again clears."""
        return toggle_focus(dash.ctx.triggered_id, current, "hpo",
                            ("pa-term-clear", "pa-report"))

    @app.callback(
        Output("pa-sent-focus", "data"),
        Input({"type": "pa-sent", "n": ALL}, "n_clicks"),
        Input("pa-sent-clear", "n_clicks"),
        Input("pa-report", "value"),
        State("pa-sent-focus", "data"),
        prevent_initial_call=True,
    )
    def focus_sentence(_clicks, _clear, _report, current):
        """Clicking a sentence in the report opens it below. Clicking it again shows them all."""
        return toggle_focus(dash.ctx.triggered_id, current, "n",
                            ("pa-sent-clear", "pa-report"))

    @app.callback(
        Output("pa-body", "children"),
        Output(f"sig-{VIEW_ID}", "data"),
        Input("tabs", "value"),
        Input("run-a", "value"),
        Input("pa-report", "value"),
        Input("store-config", "data"),
        Input("store-theme", "data"),
        Input("pa-raw", "value"),
        Input("pa-sentences", "value"),
        Input("pa-sent-focus", "data"),
        Input("pa-term-focus", "data"),
        State(f"sig-{VIEW_ID}", "data"),
    )
    def body(tab, run_a, report_id, config, mode, raw_toggle, sentence_toggle, sent_focus,
             term_focus, previous):
        key = common.render_key(VIEW_ID, run_a, report_id, config, mode, raw_toggle,
                                sentence_toggle, sent_focus, term_focus)
        if not common.should_render(VIEW_ID, tab, key, previous):
            raise PreventUpdate
        bundle = common.bundle_for(run_a)
        if bundle is None:
            return common.no_data(run_a), key
        if not report_id:
            return theme.empty("Pick a report above."), key
        cfg = common.resolve(config, bundle)
        return render_report(bundle, cfg, report_id, mode or "light",
                             "raw" in (raw_toggle or []),
                             SHOW_ALL in (sentence_toggle or []),
                             sent_focus, term_focus), key


# ──────────────────────────────────────────────────────────────────────────────
# what the controls do
#
# The three below are the bodies of three callbacks, lifted out as pure functions. Both focus
# callbacks are driven by pattern-matching inputs, which the selftest's callback smoke cannot
# dispatch, it has no browser to match a pattern against, so lifting them out is what keeps them
# checkable at all. The callbacks themselves are then one line each and hold no logic to get wrong.
# ──────────────────────────────────────────────────────────────────────────────
def toggle_focus(trigger, current, field: str, clear_ids: tuple[str, ...]):
    """The new focus after a click: the clicked thing, or ``None`` to clear.

    Clicking the thing already in focus clears it, so one control both sets and unsets the filter.
    Any of *clear_ids* clears outright, including the report picker, because a filter left
    pointing at a term the new report has never heard of shows an empty list that reads as a data
    problem, not as a filter.
    """
    if trigger in clear_ids:
        return None
    if not isinstance(trigger, dict):
        raise PreventUpdate
    value = trigger.get(field)
    return None if value == current else value


def step_to(trigger, current, values: list):
    """The report ◀ or ▶ moves to, or ``None`` when that is already where we are.

    Clamped, not wrapping: a list of reports has two ends, and silently jumping from the
    last to the first is a navigation the reader did not ask for.
    """
    if not values:
        return None
    index = values.index(current) if current in values else 0
    if trigger == "pa-prev":
        index = max(0, index - 1)
    elif trigger == "pa-next":
        index = min(len(values) - 1, index + 1)
    return None if values[index] == current else values[index]


# ──────────────────────────────────────────────────────────────────────────────
# ordering
# ──────────────────────────────────────────────────────────────────────────────
def report_key(report_id: str):
    """Numeric where the id is a number, alphabetical after it, a total order either way."""
    return (0, int(report_id), "") if report_id.isdigit() else (1, 0, report_id)


def ordered_reports(bundle: dict, cfg: dict) -> list[tuple[str, float | None]]:
    """``[(report_id, F1)]`` in numeric report order. F1 is ``None`` where nothing is annotated."""
    scores = dict(rank_reports(bundle, cfg))
    return [(r, scores.get(r))
            for r in sorted(common.report_ids(bundle, cfg), key=report_key)]


def rank_reports(bundle: dict, cfg: dict) -> list[tuple[str, float | None]]:
    """``[(report_id, F1)]`` worst first. Reports with no annotations sort last.

    No longer the picker's order, it supplies the F1 each option is labelled with, and the
    "which report should I look at" question it answers is now the comparison page's job.
    """
    from hpo_extraction.evaluation.set_metrics import calc_metric

    predicted = votes.predicted_sets(bundle, cfg)
    ranked = []
    for report_id in common.report_ids(bundle, cfg):
        gold = set(bundle["gold"].get(report_id, ()))
        f1 = calc_metric(gold, predicted.get(report_id, set()))[2] if gold else None
        ranked.append((report_id, f1))
    return sorted(ranked, key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0))


# ──────────────────────────────────────────────────────────────────────────────
# The page
# ──────────────────────────────────────────────────────────────────────────────
def render_report(bundle: dict, cfg: dict, report_id: str, mode: str, show_raw: bool,
                  show_all_sentences: bool = False, sent_focus=None, term_focus=None):
    """The report text of *report_id* with every juror's terms marked."""
    return html.Div([
        render_stats(bundle, cfg, report_id),
        render_annotated_report(bundle, cfg, report_id, mode),
        render_terms(bundle, cfg, report_id, mode, term_focus),
        render_sentences(bundle, cfg, report_id, mode, show_raw, show_all_sentences,
                         sent_focus, term_focus),
        render_prompt(),
    ])


def render_stats(bundle: dict, cfg: dict, report_id: str):
    """Precision, recall and F1 of *report_id*."""
    from hpo_extraction.evaluation.set_metrics import calc_metric

    gold = set(bundle["gold"].get(report_id, ()))
    predicted = votes.predicted_sets(bundle, cfg).get(report_id, set())
    precision, recall, f1 = calc_metric(gold, predicted) if gold else (0.0, 0.0, 0.0)
    sentences = bundle["sentences"].get(report_id, {})
    tiles = [
        theme.stat("F1", common.num(f1) if gold else "—", common.describe(cfg)),
        theme.stat("precision", common.num(precision) if gold else "—",
                   f"{len(gold & predicted)} of {len(predicted)} predicted"),
        theme.stat("recall", common.num(recall) if gold else "—",
                   f"{len(gold & predicted)} of {len(gold)} annotated"),
        theme.stat("sentences", str(len(sentences)),
                   f"x {len(cfg['models'])} models = {len(sentences) * len(cfg['models'])} "
                   "generations"),
    ]
    pb = bundle.get("pb_standalone")
    if pb is not None:
        pb_pred = set(pb["predicted"].get(report_id, ()))
        pb_f1 = calc_metric(gold, pb_pred)[2] if gold else None
        tiles.append(theme.stat(
            "PhenoBERT F1", common.num(pb_f1) if gold else "—",
            f"{len(gold & pb_pred)} of {len(pb_pred)} predicted",
            help_text="PhenoBERT baseline: the same grounder run over the raw report instead of over what "
                      "the models wrote. The baseline this report's ensemble score has to beat."))
    return theme.stat_row(tiles)


# ── the report as a document ─────────────────────────────────────────────────

def render_annotated_report(bundle: dict, cfg: dict, report_id: str, mode: str):
    """The full report with the baseline's matches underlined, and the annotated terms beneath it.

    This is the only place in the app that shows the report as a document, not as a list of
    sentences, and the only place the *raw* PhenoBERT run is visible. It is also where negation and
    unresolved codes become legible: both are dropped from the predicted set, so without this panel
    "PhenoBERT missed it" and "PhenoBERT found it and said it was absent" look identical.
    """
    pb = bundle.get("pb_standalone")
    gold = set(bundle["gold"].get(report_id, ()))

    if pb is None:
        return theme.panel(
            "The report",
            html.Div([
                theme.note("The full report text comes from the PhenoBERT baseline run "
                           "(phenobert_input/), which is not loaded for this cohort. The sentences "
                           "the models saw are further down; only the document view and the "
                           "baseline's annotations are missing.", "warn"),
                _gold_chips(bundle, gold),
            ]),
            panel_id="pa-report-text",
            subtitle=f"{len(gold)} annotated term(s)")

    text = pb["texts"].get(report_id)
    rows = pb["by_report"].get(report_id, [])
    if text is None:
        return theme.panel(
            "The report",
            html.Div([
                theme.note("This report has no staged text in phenobert_input/, so the document "
                           "cannot be shown, the run kept its detections but not its inputs.",
                           "warn"),
                _gold_chips(bundle, gold),
            ]),
            panel_id="pa-report-text",
            subtitle=f"{len(gold)} annotated term(s), {len(rows)} baseline detection(s)")

    if not pbstandalone.report_ok(pb, report_id):
        return theme.panel(
            "The report",
            html.Div([
                theme.note("The baseline's character offsets do not land on their own matched "
                           "phrases in this report's staged text (gate G6 failed), so the "
                           "annotations would underline the wrong words. The text is shown "
                           "without them.", "danger"),
                html.Div(text, className="report-body"),
                _gold_chips(bundle, gold),
            ]),
            panel_id="pa-report-text",
            subtitle=f"{len(gold)} annotated term(s)")

    body, counts = _annotate(text, [(r["start"], r["end"], r) for r in rows], bundle, mode)
    return theme.panel(
        "The report",
        html.Div([
            html.Div(body, className="report-body"),
            _legend(mode),
            _gold_chips(bundle, gold),
        ]),
        panel_id="pa-report-text",
        subtitle=(f"annotated by the raw PhenoBERT baseline: {counts['positive']} "
                  f"match(es) that became predictions, {counts['negated']} negated, "
                  f"{counts['unresolved']} not in this HPO release. "
                  f"{len(gold)} annotated term(s) below."),
        markdown=_annotations_md(bundle, rows))


def _annotate(text: str, spans: list[tuple[int, int, dict]], bundle: dict, mode: str):
    """Text with each matched phrase marked **in place** and tagged with the term it normalised to.

    The one annotator on this page. The report and every model reply go through it, so the two
    surfaces are not merely similar, they are the same rendering with different input, and a change
    to one cannot drift from the other. *spans* is ``[(start, end, row)]`` into *text*, because the
    two callers get their offsets from different places: the report's index the staged report
    directly, a reply's index the stripped reply through ``offset_in_reply``.

    Each match becomes the phrase, underlined, immediately followed by an inline tag naming the HPO
    term. The underline alone says *where* PhenoBERT found something. The tag says *what it became*,
    which is the half that decides whether the extraction was right. Three kinds, by what happened
    to the match afterwards:

    ``positive``    in the ontology and not negated, this is what became a prediction
    ``negated``     PhenoBERT found the phenotype and ruled it out, so it never counted
    ``unresolved``  the code is not in this HPO release, so it was dropped

    Overlapping or out-of-range spans are skipped, not drawn approximately: an offset that
    cannot be honoured is one that should not be honoured at all.
    """
    p = theme.palette(mode)
    colours = {"positive": p["good"], "negated": p["neutral"], "unresolved": p["warning"]}
    counts = {"positive": 0, "negated": 0, "unresolved": 0}
    children: list = []
    cursor = 0

    for start, end, row in sorted(spans, key=lambda s: (s[0], s[1])):
        if start < cursor or end > len(text) or end <= start:
            continue
        children.append(text[cursor:start])
        kind = _kind(row)
        counts[kind] += 1
        colour = colours[kind]
        title = _mark_title(bundle, row, kind)
        children.append(html.Span([
            html.Mark(text[start:end], className=f"pb-mark pb-mark-{kind}",
                      style={"background": "transparent", "borderBottom": f"2px solid {colour}",
                             "color": "inherit",
                             "textDecoration": "line-through" if kind == "negated" else "none"}),
            html.Span(_tag_text(bundle, row, kind), className=f"pb-tag pb-tag-{kind}",
                      style={"borderColor": colour, "color": colour}),
        ], className="pb-ann", title=title))
        cursor = end

    children.append(text[cursor:])
    return children, counts


def _kind(row: dict) -> str:
    """What happened to a match. ``resolved`` is absent on reply rows, where it is always true."""
    if row.get("negated"):
        return "negated"
    if not row.get("resolved", True):
        return "unresolved"
    return "positive"


def _tag_text(bundle: dict, row: dict, kind: str) -> str:
    """The inline label: the term this phrase normalised to, plus why it did not count.

    The **name**, not the id, the tag sits inside flowing text and is read at a glance, where
    ``HP:0001250`` says nothing and ``Seizure`` says everything. The id is one hover away. Resolved
    through the ontology, not off the row, because only the report's rows carry an
    ``hpo_label`` and the two surfaces must not label the same term differently.
    """
    label = relations.label(bundle["tree"], row["hpo_id"]) if bundle.get("tree") is not None \
        else (row.get("hpo_label") or row["hpo_id"])
    suffix = {"negated": " · negated", "unresolved": " · not in this HPO release"}.get(kind, "")
    return f"{label or row['hpo_id']}{suffix}"


_MARK_WHY = {
    "positive": "counted as a PhenoBERT prediction",
    "negated": "NEGATED, PhenoBERT found this and ruled it out, so it is not a prediction",
    "unresolved": "not in this HPO release, so it was dropped from the predictions",
}


def _mark_title(bundle: dict, row: dict, kind: str) -> str:
    """The hover on one match: what it is, why it counted or did not, and what the term means.

    Tolerant of both row shapes. The report's rows come from ``phenobert_detections.jsonl``
    (``score``, ``hpo_label``, ``raw_hpo_id``). A reply's come from the per-model TSVs via
    ``detections.build_pb_table``, which carry ``confidence`` and nothing else, so every
    shape-specific field is read with ``get``.
    """
    head = f"{row['hpo_id']} · {row.get('hpo_label') or common.term(bundle, row['hpo_id'])}"
    lines = [head, f"matched {row['phrase']!r}"]
    score = row.get("score", row.get("confidence"))
    if score is not None:
        lines.append(f"match score {score:.2f} (a fixed filter threshold, not a confidence)")
    if row.get("raw_hpo_id") and row["raw_hpo_id"] != row["hpo_id"]:
        lines.append(f"remapped from the alt id {row['raw_hpo_id']}")
    lines.append(_MARK_WHY[kind])
    detail = relations.definition(bundle.get("tree"), row["hpo_id"])
    if detail:
        lines.append(detail)
    syns = relations.synonyms(bundle.get("tree"), row["hpo_id"])
    if syns:
        lines.append("Also known as: " + ", ".join(syns))
    return "\n\n".join(lines)


def _legend(mode: str):
    """The three annotation kinds, drawn the way they are drawn in the text."""
    p = theme.palette(mode)
    entries = [("phrase", "HPO term", "became a prediction", p["good"], False),
               ("phrase", "HPO term · negated", "found, then ruled out", p["neutral"], True),
               ("phrase", "HPO term · not in this HPO release", "dropped", p["warning"], False)]
    return html.Div(
        [html.Span([
            html.Span([
                html.Mark(phrase, style={
                    "background": "transparent", "borderBottom": f"2px solid {colour}",
                    "color": "inherit",
                    "textDecoration": "line-through" if struck else "none"}),
                html.Span(tag, className="pb-tag", style={"borderColor": colour, "color": colour}),
            ], className="pb-ann"),
            html.Span(f" {meaning}", className="stat-sub"),
        ], style={"marginRight": "16px"}) for phrase, tag, meaning, colour, struck in entries],
        className="legend-row")


def _gold_chips(bundle: dict, gold: set[str]):
    """The annotated terms, each with its definition and synonyms one hover away.

    Chips, not spans in the text: HCY's ground truth is a set of terms per report with no
    character offsets, so there is nothing to underline. GSC+ does carry mention-level spans, but
    they live in the cohort directory, not in any run, and a ground truth layer that appeared on one
    cohort and not the other would be read as the *method* differing.
    """
    if not gold:
        return html.Div("No annotated terms for this report.", className="stat-sub")
    return html.Div([
        html.Div("annotated terms", className="stat-label", style={"marginTop": "10px"}),
        html.Div([common.term_chip(bundle, h) for h in sorted(gold)], className="chip-row"),
    ])


def _annotations_md(bundle: dict, rows: list[dict]) -> str:
    return common.md_table(
        ["term", "phrase", "offset", "outcome"],
        [[common.term(bundle, r["hpo_id"]), r["phrase"], f"{r['start']}–{r['end']}",
          "negated" if r["negated"] else "unresolved" if not r["resolved"] else "predicted"]
         for r in rows])


# ── terms in play ────────────────────────────────────────────────────────────

def render_terms(bundle: dict, cfg: dict, report_id: str, mode: str = "light", term_focus=None):
    """Every term in play for this report: predicted, annotated, or both.

    Each term is a button. Clicking it filters the sentence list below to the sentences that term
    has anything to do with, which is the difference between reading a report and interrogating a
    decision.
    """
    gold = set(bundle["gold"].get(report_id, ()))
    all_predicted = votes.predicted_sets(bundle, cfg)
    predicted = all_predicted.get(report_id, set())
    masks = bundle["masks"](cfg["min_count"])
    subset_mask = bundle["mask_of"](cfg["models"])
    k = cfg["k"] if cfg["rule"] != "plurality" else 1
    colors = theme.model_colors(bundle["all_models"], mode)
    tree = bundle.get("tree")
    pb = bundle.get("pb_standalone")

    # Built over the whole cohort, not just this report: the autopsy walks report_ids, and handing
    # it a one-report dict would classify every other report's annotations as misses.
    fates = {(r["report_id"], r["hpo_id"]): r["fate"]
             for r in autopsy.build(bundle, cfg, bundle["gold"], all_predicted,
                                   common.report_ids(bundle, cfg))["rows"]}

    headers = ["term", "outcome", "votes", "who voted (ringed = decisive)", "class / fate"]
    if pb is not None:
        headers.insert(2, "PhenoBERT")

    rows = []
    for hpo_id in sorted(gold | predicted, key=lambda h: (h not in predicted, h)):
        mask = masks.get(report_id, {}).get(hpo_id, 0) & subset_mask
        outcome = ("TP" if hpo_id in gold and hpo_id in predicted else
                   "FP" if hpo_id in predicted else "FN")
        if outcome == "FP" and tree is not None:
            note = relations.classify(tree, hpo_id, gold)
        elif outcome == "FN":
            note = fates.get((report_id, hpo_id), "")
        else:
            note = ""
        decisive = votes.decisive_mask(mask, subset_mask, k) if outcome != "FN" else 0
        row = [_term_button(bundle, hpo_id, term_focus), outcome, popcount(mask),
               common.vote_chips(mask, decisive, cfg["models"], colors), note]
        if pb is not None:
            row.insert(2, _pb_cell(pb, report_id, hpo_id))
        rows.append(row)

    subtitle = f"{len(gold)} annotated, {len(predicted)} predicted at {common.describe(cfg)}"
    if pb is not None:
        subtitle += (". The PhenoBERT column is the PhenoBERT baseline on the same term, 'negated' "
                     "and 'unresolved' mean it matched the phrase and then dropped it")
    subtitle += ". Click a term to filter the sentences below to it."

    return theme.panel(
        "Terms in play",
        html.Div([
            common.table(headers, rows, numeric_from=2),
            _focus_bar(bundle, term_focus),
        ]),
        panel_id="pa-terms",
        subtitle=subtitle,
        markdown=common.md_table(
            ["term", "outcome", "votes"],
            [[h, ("TP" if h in gold and h in predicted else "FP" if h in predicted else "FN"),
              popcount(masks.get(report_id, {}).get(h, 0) & subset_mask)]
             for h in sorted(gold | predicted)]))


def _term_button(bundle: dict, hpo_id: str, term_focus):
    """A term as a filter control, with its definition on the ⓘ beside it."""
    active = " is-active" if hpo_id == term_focus else ""
    return html.Span([
        html.Button(common.term(bundle, hpo_id), id={"type": "pa-term", "hpo": hpo_id},
                    className=f"term-btn{active}", n_clicks=0,
                    title="show only the sentences this term appears in"),
        common.info_dot(bundle, hpo_id),
    ])


def _pb_cell(pb: dict, report_id: str, hpo_id: str) -> str:
    """What the baseline did with this term on this report."""
    if hpo_id in pb["predicted"].get(report_id, ()):
        return "✓ predicted"
    rows = [r for r in pb["by_report"].get(report_id, []) if r["hpo_id"] == hpo_id]
    if not rows:
        return "—"
    if all(r["negated"] for r in rows):
        return "negated"
    if all(not r["resolved"] for r in rows):
        return "unresolved"
    return "matched, dropped"


def _focus_bar(bundle: dict, term_focus):
    """The clear button. Always rendered, so the callback bound to it always has a target."""
    return html.Div([
        html.Button("clear term filter", id="pa-term-clear", className="copy-btn", n_clicks=0,
                    style={} if term_focus else {"display": "none"}),
        html.Span(f"  filtering the sentences to {common.term(bundle, term_focus)}"
                  if term_focus else "", className="stat-sub"),
    ], style={"marginTop": "8px"})


# ── the evidence, sentence by sentence ───────────────────────────────────────

def render_sentences(bundle: dict, cfg: dict, report_id: str, mode: str, show_raw: bool,
                     show_all: bool = False, sent_focus=None, term_focus=None):
    """The report again, now as a sentence picker, with the chosen sentence's evidence below it.

    By default every sentence at least one selected model wrote something about is drawn. A clinical
    report is mostly sentences every model answered "no phenotype" to, and a block for each of those
    is (sentences x models) reply components carrying nothing, the bulk of this page's weight, for
    none of its content. Picking one sentence narrows it to that sentence. Picking a term narrows it
    to the sentences that term appears in. What was left out is always counted on screen.
    """
    sentences = bundle["sentences"].get(report_id, {})
    if not sentences:
        return common.empty_panel(
            "No sentence text for this report, its llm_extractions_*.jsonl is missing.",
            "pa-sents", "Sentences")

    spans = _sentence_spans(bundle, report_id, sentences)
    related = (_related_sentences(bundle, cfg, report_id, sentences, spans, term_focus)
               if term_focus else None)

    notes = []
    picker = _sentence_picker(bundle, report_id, sentences, spans, sent_focus, related)
    if spans is None:
        notes.append(theme.note(
            "The sentences the models saw could not be located inside the staged report text, so "
            "they are listed rather than shown in place. Everything below is unaffected; only the "
            "in-document view is. This happens when the pipeline's sentence splitter rewrote the "
            "text it split.", "warn"))

    blocks, n_hidden, n_shown = _evidence_blocks(
        bundle, cfg, report_id, sentences, mode, show_raw, show_all, sent_focus, related, spans)

    controls = html.Div([
        html.Button("show every sentence again", id="pa-sent-clear", className="copy-btn",
                    n_clicks=0, style={} if sent_focus is not None else {"display": "none"}),
    ], style={"marginTop": "8px"})

    body = html.Div(notes + [picker, controls, html.Div(blocks or [theme.empty(
        "No model wrote anything about any sentence of this report. Tick the box above to read "
        "them anyway.")])])

    subtitle = ("Click a sentence above to read only its evidence. Inside each reply, the "
                "underlined text is the exact phrase PhenoBERT matched, positioned by its own "
                "character offsets, and the tag beside it is the HPO term that phrase was "
                "normalised to, so a phrase linked to something other than what it says is "
                "visible without hovering. Replies are shown after reasoning blocks are stripped, "
                "which is the text PhenoBERT actually read.")
    if term_focus:
        subtitle += (f" Filtered to {common.term(bundle, term_focus)}: "
                     f"{len(related or ())} of {len(sentences)} sentence(s) have it grounded by a "
                     "model or matched by the baseline.")
    if sent_focus is not None:
        subtitle += f" Showing sentence {sent_focus} only."
    elif n_hidden:
        subtitle += (f" {n_hidden} of {len(sentences)} sentences are hidden: no selected model "
                     "wrote anything about them and nothing was grounded there.")
    if not blocks and n_shown == 0 and term_focus:
        subtitle += " Nothing matches this term filter."

    return theme.panel("Sentences", body, panel_id="pa-sents", subtitle=subtitle)


def _sentence_spans(bundle: dict, report_id: str, sentences: dict):
    """Where each sentence sits in the staged report, or ``None`` if it cannot be established.

    Memoised on the bundle: the alignment is a scan of the whole report and three parts of this
    panel want the answer, but it depends only on the report, not on the configuration, so it
    survives every control the sidebar offers.
    """
    from .. import memo

    pb = bundle.get("pb_standalone")
    if not pb:
        return None
    text = pb["texts"].get(report_id)
    if text is None:
        return None
    return memo.cached(bundle, ("patient.spans", report_id),
                       lambda: pbstandalone.align_sentences(text, sentences))


def _related_sentences(bundle: dict, cfg: dict, report_id: str, sentences: dict, spans,
                       hpo_id: str) -> set[int]:
    """Sentences a term has anything to do with: grounded by a model, or matched by the baseline."""
    found = set()
    per_sentence = bundle["sentence_masks"](cfg["min_count"]).get(report_id, {})
    subset_mask = bundle["mask_of"](cfg["models"])
    for sent_num, hpos in per_sentence.items():
        if hpos.get(hpo_id, 0) & subset_mask:
            found.add(sent_num)

    pb = bundle.get("pb_standalone")
    if pb and spans:
        for row in pb["by_report"].get(report_id, []):
            if row["hpo_id"] != hpo_id:
                continue
            sent_num = pbstandalone.sentence_of(spans, row["start"])
            if sent_num is not None:
                found.add(sent_num)
    return found & set(sentences)


def _sentence_picker(bundle: dict, report_id: str, sentences: dict, spans, sent_focus, related):
    """The report as clickable sentences, in place when the alignment held, as a list when not."""
    classes = {}
    for sent_num in sentences:
        parts = ["report-sent"]
        if sent_num == sent_focus:
            parts.append("is-selected")
        if related is not None and sent_num in related:
            parts.append("is-related")
        classes[sent_num] = " ".join(parts)

    def clickable(sent_num: int, text: str):
        return html.Span(text, id={"type": "pa-sent", "n": sent_num}, n_clicks=0,
                         className=classes[sent_num],
                         title=f"sentence {sent_num}, click to read its evidence")

    if spans is None:
        return html.Div([
            html.Div([html.Span(f"{n}", className="pill sent-num"), clickable(n, sentences[n])],
                     className="sentence-line")
            for n in sorted(sentences)
        ], className="report-body")

    pb = bundle.get("pb_standalone")
    text = pb["texts"].get(report_id, "") if pb else ""
    children = []
    cursor = 0
    for sent_num in sorted(spans, key=lambda n: spans[n][0]):
        start, end = spans[sent_num]
        if start < cursor or end > len(text):
            continue
        children.append(text[cursor:start])
        children.append(clickable(sent_num, text[start:end]))
        cursor = end
    children.append(text[cursor:])
    return html.Div(children, className="report-body")


def _evidence_blocks(bundle: dict, cfg: dict, report_id: str, sentences: dict, mode: str,
                     show_raw: bool, show_all: bool, sent_focus, related, spans=None):
    """One block per shown sentence: the grounded terms, then every model's reply."""
    colors = theme.model_colors(bundle["all_models"], mode)
    by_sentence = bundle["pb_index"]["by_sentence"]
    sent_masks = bundle["sentence_masks"](cfg["min_count"])
    replies_by_key = bundle["built"]["replies"]
    subset_mask = bundle["mask_of"](cfg["models"])
    blocks = []
    n_hidden = 0
    n_shown = 0

    for sent_num in sorted(sentences):
        if sent_focus is not None and sent_num != sent_focus:
            continue
        if related is not None and sent_num not in related:
            continue
        n_shown += 1

        hpos = sent_masks.get(report_id, {}).get(sent_num, {})
        tally = sorted(((h, popcount(m & subset_mask)) for h, m in hpos.items()),
                       key=lambda x: -x[1])
        tally = [(h, v) for h, v in tally if v]

        replies = []
        wrote = False
        for model in cfg["models"]:
            reply = replies_by_key.get((model, report_id, sent_num))
            if reply is None:
                continue
            wrote = wrote or reply["wrote"]
            pb_rows = by_sentence.get((model, report_id, sent_num), [])
            replies.append(html.Div([
                html.Div(model, className="stat-label",
                         style={"color": colors.get(model, "#888")}),
                html.Div(_reply_body(bundle, reply, pb_rows, mode, show_raw),
                         className="reply-text"),
            ], className="reply-block"))

        # An explicitly picked sentence is always drawn, the reader asked for that one, and
        # "nothing here" is the answer they asked for.
        explicit = sent_focus is not None or related is not None
        if not show_all and not explicit and not wrote and not tally:
            n_hidden += 1
            n_shown -= 1
            continue

        blocks.append(html.Div([
            html.Div([
                html.Span(f"{sent_num}", className="pill",
                          style={"background": theme.palette(mode)["neutral"]}),
                html.Span(sentences[sent_num], className="sentence-text"),
            ], className="sentence-head"),
            html.Div(
                [common.term_chip(bundle, h, suffix=f" · {v}/{len(cfg['models'])}",
                                  style={"background": theme.palette(mode)["surface_2"],
                                         "color": theme.palette(mode)["text_secondary"]})
                 for h, v in tally] or [html.Span("no grounded detections", className="stat-sub")],
                className="chip-row"),
            _baseline_here(bundle, report_id, sent_num, mode, spans),
            html.Div(replies, className="reply-grid"),
        ], className="sentence-block"))

    return blocks, n_hidden, n_shown


def _baseline_here(bundle: dict, report_id: str, sent_num: int, mode: str, spans):
    """What the raw-report baseline matched inside this sentence, if anything.

    Only shown when the sentence could be placed in the staged report, an unaligned sentence has
    no offsets to compare against, and guessing would put the baseline's evidence on the wrong line.
    """
    pb = bundle.get("pb_standalone")
    if not pb or not spans or sent_num not in spans:
        return None
    start, end = spans[sent_num]
    rows = [r for r in pb["by_report"].get(report_id, []) if start <= r["start"] < end]
    if not rows:
        return None
    p = theme.palette(mode)
    return html.Div([
        html.Span("PhenoBERT on the raw report: ", className="stat-sub"),
        *[html.Span(
            f"{r['phrase']} → {common.term(bundle, r['hpo_id'])}"
            + (" (negated)" if r["negated"] else "" if r["resolved"] else " (unresolved)"),
            className="pill",
            title=_mark_title(bundle, r,
                              "negated" if r["negated"] else
                              "positive" if r["resolved"] else "unresolved"),
            style={"background": "transparent",
                   "border": f"1px solid {p['neutral'] if r['negated'] else p['good']}",
                   "color": p["text_secondary"]})
          for r in rows],
    ], className="chip-row baseline-row")


def _reply_body(bundle: dict, reply: dict, pb_rows: list[dict], mode: str, show_raw: bool):
    """One model's reply, annotated in place by the same renderer the report above uses.

    The offsets index the **stripped** reply, because ``run_phenobert_per_model`` strips reasoning
    blocks before writing its input. The raw reply is available behind a toggle and is
    left unannotated: these offsets would be off by the length of the ``<think>`` block, and a
    highlight that is confidently in the wrong place is worse than none.

    A row whose ``offset_in_reply`` could not be resolved is reported at the end, not
    dropped or searched for. Searching would put the mark on a plausible occurrence, not the
    detected one, the failure ``verify.gate_spans`` exists to make impossible, but saying nothing
    would read as "PhenoBERT found nothing here", which is a different and wrong claim.
    """
    p = theme.palette(mode)
    if show_raw:
        return html.Span(reply["raw"] or "(empty)")

    # The terminated form, because that is what the offsets index, and what PhenoBERT was actually
    # handed. It differs from `stripped` only by the line terminators `terminate_lines` supplies.
    text = reply["pb_text"]
    if not text.strip():
        note = ("empty after the reasoning block was stripped, this model contributed nothing "
                "here" if reply["raw"].strip() else "empty reply")
        return html.Span(f"({note})", style={"color": p["text_muted"], "fontStyle": "italic"})

    placed = [(r["offset_in_reply"], r["offset_in_reply"] + (r["end"] - r["start"]), r)
              for r in pb_rows if r.get("offset_in_reply") is not None]
    children, _counts = _annotate(text, placed, bundle, mode)

    unplaced = [r for r in pb_rows if r.get("offset_in_reply") is None]
    if unplaced:
        children.append(html.Span(
            "  (PhenoBERT also linked "
            + ", ".join(f"{r['phrase']} → {common.term(bundle, r['hpo_id'])}" for r in unplaced)
            + " here, but its offsets could not be placed in this text)",
            style={"color": p["text_muted"], "fontStyle": "italic"}))
    elif reply["wrote"] and not placed:
        children.append(html.Span("  (PhenoBERT normalised nothing in this reply)",
                                  style={"color": p["text_muted"], "fontStyle": "italic"}))

    if not reply["wrote"]:
        children.append(html.Span("  (counted as 'wrote nothing')",
                                  style={"color": p["text_muted"], "fontStyle": "italic"}))
    return html.Span(children)


def render_prompt():
    """The extraction prompt, imported from the driver so it cannot drift."""
    SYSTEM_PROMPT, USER_TEMPLATE, reason = votes.prompts()
    if SYSTEM_PROMPT is None:
        return theme.panel(
            "The prompt every model saw",
            theme.note("The prompt lives in hpo_extraction.phenojury.generation, which will not import "
                       f"here ({reason}). It is not transcribed anywhere, a copy would drift, "
                       "so it cannot be shown. Every other panel on this screen is unaffected.",
                       "warn"),
            panel_id="pa-prompt")

    return theme.panel(
        "The prompt every model saw",
        html.Div([
            html.Div("system", className="stat-label"),
            html.Pre(SYSTEM_PROMPT, className="prompt-box"),
            html.Div("user", className="stat-label"),
            html.Pre(USER_TEMPLATE, className="prompt-box"),
            theme.note("Imported from hpo_extraction.phenojury.generation, not transcribed. One prompt "
                       "per sentence per model: no retrieval, no ontology, no candidate list, "
                       "which is why the ensemble's errors are generation errors, not "
                       "retrieval ones. medpsy and medgemma have no system role and receive the "
                       "system text prepended to the user turn.", "info"),
        ]),
        panel_id="pa-prompt",
        markdown=f"**system**\n\n```\n{SYSTEM_PROMPT}\n```\n\n**user**\n\n```\n{USER_TEMPLATE}\n```")
