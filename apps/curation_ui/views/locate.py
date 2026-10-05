"""Placing the annotations nobody could place, the by-hand queue, on a screen of its own.

Approve mode asks *is this term right for this report?*. This tab asks the other question the
curation pass is left with: **where in the report did this term come from?** They are different jobs
and they want different screens. Adjudicating means reading every term the files carry. Placing the
last few means seeing only those, because a queue you have to find inside a full list is a queue
nobody finishes.

Three states put an annotation here, and they cost a curator the same work even though they are
different failures (:func:`reader.needs_anchor`):

* the file named words the report does not contain, a report re-exported after annotation;
* it named a sentence the trigger did not survive re-tokenization into;
* it named **no trigger at all**, so there was never anything to place.

The population is the union the curated ground truth is built from, ``prior_annotation ∪ suggestions ∪ daphne``,
which is what ``hcy_ground_truth`` already scores against, taken **one row per
annotation row**, not per code. That queue is now essential rather than tidy-up:
``curated_gold.GoldPolicy.require_evidence`` keeps a phenotype out of the exported ground truth until it
sits on a segment and a trigger word, using this module's own :func:`reader.needs_anchor` test, so
a row left here is a term no downstream table will score against. ``sources.adjudicated`` is per code, best source, so a prior_annotation row
``daphne`` never confirmed would be located implicitly, not in its own right, and the drop
that put it there is the judgement somebody has to see. ``prior_annotation_2`` is not in the union: it is
a code list, it never claimed a position, and its lexical candidate is offered elsewhere.

Suggestions this app made carry a segment by design, so they never reach the queue. They are
still counted in the denominator, which is what makes "3 of 11" a statement about the report, not about the queue.

**Locating is evidence, not a verdict.** There are no Keep/Remove buttons here. Recording which
sentence a term came from says nothing about whether the term belongs, and a screen that fused the
two would leave a log that cannot answer who kept it, the same rule the ``anchor`` and ``edit``
actions already follow in ``store``'s fold.

Two things are reused, not rebuilt:

* :func:`editor.panel`, mounted at ``AT``. ``editor.register`` is global and matches on
  ``{"type": "ann-*", "key": ALL, "at": ALL}``, so Save, *Jump to*, *Use selected*, *Use selection*,
  the per-row HPO search and the Tab-completion in ``assets/trigger_complete.js`` all work here
  without one new callback. The ``at`` discriminator is what keeps this panel's ids apart from
  Edit's and Approve's, all three of which are in the layout at once.
* ``approve``'s ``{"type": "anchor", …}`` buttons for the lexical candidates, and the sidebar's
  ``{"type": "pat", "pid": …}`` ids for the worklist, so this module adds no second writer of
  ``store-patient`` and no second way to record an evidence location.
"""

from __future__ import annotations

import dash
from dash import Input, Output, html
from dash.exceptions import PreventUpdate

from hpo_extraction.curation import sources as sources_mod
from .. import theme
from . import common, editor, labelling, reader

VIEW_ID = "anchor"

#: Where in this file's own layout the editor panels live. See ``editor.ident``. It must differ from
#: ``edit.AT`` and ``approve.AT``: all three panels are mounted at once, and Dash refuses to render
#: a layout carrying one component id twice.
AT = "anchor"

#: How many patients the worklist names before it stops. It is a queue to work through, not a census
#:, the count in the header is the census.
WORKLIST_LIMIT = 40


def layout() -> html.Div:
    """The Dash layout of this view."""
    return html.Div([
        html.Div(id="anchor-body"),
        html.Div(id="anchor-worklist"),
    ], id="anchor-panel")


def register(app: dash.Dash) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("anchor-body", "children"),
        Input("store-patient", "data"),
        Input("store-dirty", "data"),
        Input("store-theme", "data"),
        Input("tabs", "value"),
    )
    def draw(patient_id, _dirty, mode, tab):
        if tab != "anchor":
            raise PreventUpdate
        return common.guard("Evidence queue", render_queue, patient_id, mode or "light")

    @app.callback(
        Output("anchor-worklist", "children"),
        Input("store-dirty", "data"),
        Input("store-theme", "data"),
        Input("tabs", "value"),
    )
    def draw_worklist(_dirty, mode, tab):
        # Gated on the tab for the same reason Overview gates its own cohort work: this walks every
        # patient, and ``Registry`` caches sixteen of them. Doing it while somebody is curating
        # would evict the report they are reading to answer a question they did not ask.
        if tab != "anchor":
            raise PreventUpdate
        return common.guard("Evidence worklist", render_worklist, mode or "light")


# ──────────────────────────────────────────────────────────────────────────────
# rendering, pure, so selftest can drive it without a browser
# ──────────────────────────────────────────────────────────────────────────────
def queue_for(registry, patient_id: str) -> tuple[list[dict], int]:
    """``(rows still needing a segment and a trigger, how many annotations the report has)``.

    One entry per annotation **row**, carrying everything the panel draws: the source record, where
    it ended up if anywhere, and this app's own version of it from the log.
    """
    view = registry.patient(patient_id)
    if view is None:
        return [], 0

    placed = reader.placed_by_key(view)
    curated = {row["key"]: row for row in registry.rows_for(patient_id)}
    display = view["display"]

    # One row per *annotation*, not per file row: ``holistic`` and ``daphne`` overlap by
    # construction, so a term both carry is one thing to place, and asking for it twice is asking
    # for it once more than there is anything to do.
    #
    # This app's own suggestions are in the population too, not merely counted. A term the curator
    # has already placed by hand is the annotation, and a file row for the same term that could not
    # be placed is the same annotation recorded badly, so it folds into the located one and stops
    # being asked about. Nothing else can close that row: the file's claim will never start working.
    # See ``sources_mod.merge_identical``.
    records = sources_mod.merge_identical(
        [record for record in view["records"] if record.get("source") in reader.ANCHORABLE]
        + [_as_record(row) for row in curated.values() if _is_live_suggestion(row)],
        reader.realised_spans(view, curated),
        mergeable=reader.ANCHORABLE)
    # The denominator is "annotations in this report", not "annotations that could have gone wrong".
    n_total = len(records)

    queue = []
    for record in records:
        row = reader.merged_row(record, curated, display)
        placement = reader.merged_placement(record, placed)
        if not reader.needs_location(record, placement, row, display):
            continue
        # ``key`` is the **primary**, the best source under the adjudication order, and the row the
        # curated ground truth would carry. Locating writes there. The merged siblings are the same
        # annotation, so the group leaves the queue on the next draw without a second write.
        queue.append({"record": record, "row": row, "placement": placement,
                      "key": record["key"], "code": record["hpo_code"],
                      "source": record["source"],
                      "sources": record.get("sources") or [record["source"]],
                      "claims": record.get("claims") or [],
                      "candidates": view["triggers"].get(record["hpo_code"], [])})

    queue.sort(key=_order)
    return queue, n_total


#: A suggestion a verdict has already closed against is not an annotation this report has. Same
#: rule, and the same two statuses, that ``reader._curated_marks`` uses to decide what to draw.
CLOSED = ("rejected", "removed")


def _is_live_suggestion(row: dict) -> bool:
    return row.get("source") == "new" and row.get("status") not in CLOSED


def _as_record(row: dict) -> dict:
    """One of this app's own rows, in the annotation-record shape the merge reads.

    A translation, not a second source: ``store``'s rows and ``sources``' records carry the
    same facts under mostly the same names, and giving the merge two shapes to understand is how the
    second one ends up handled slightly differently.
    """
    return {"source": "new", "key": row.get("key", ""),
            "patient_id": row.get("patient_id", ""),
            "hpo_code": row.get("hpo_code", ""), "hpo_name": row.get("hpo_name", ""),
            "trigger_word": row.get("trigger_word", ""),
            "segment_idx": row.get("segment_idx"), "char_offset": None,
            "sentence_context": row.get("segment_text", ""),
            "confirmed": None, "provenance": row.get("author", ""), "slot": None}


def _order(entry: dict) -> tuple:
    """Confirmed pass first, then by code. The same order the panels beside it read in."""
    rank = sources_mod.precedence(entry["source"])
    return (rank is None, rank or 0, entry["code"])


def render_queue(patient_id, mode: str = "light"):
    """The annotations of *patient_id* that still lack a segment and a trigger word."""
    registry = common.registry_or_none()
    if registry is None or not patient_id:
        return theme.empty(common.NO_PATIENT)
    view = registry.patient(patient_id)
    if view is None:
        return theme.note(f"{patient_id}: not in any source.", "warn")

    queue, n_total = queue_for(registry, patient_id)
    if queue and not view["segments"]:
        # Every field on the form names a segment, and this report has none, ``segment_reports.py``
        # never reached it. Offering the form anyway would be offering a Save that
        # ``common.validate_annotation`` refuses, with no way to see why from this screen.
        return theme.card(
            f"{patient_id}: evidence cannot be located here",
            theme.note(f"{len(queue)} annotation(s) need a segment and a trigger word, but this "
                       "report has no segmentation, so there are no sentences to attach them to. "
                       "It needs a segmentation run before it can be worked on.", "warn"),
            subtitle="The annotation files carry terms for this report; "
                     f"{sources_mod.SEGMENTS_FILE} does not.")
    if not queue:
        return theme.card(
            f"{patient_id}: nothing to place",
            theme.empty(f"All {n_total} annotation(s) in this report have a segment and a trigger "
                        "word." if n_total else "No annotation file carries a term for this "
                                                "report."),
            subtitle="Prev/Next in the sidebar, or the worklist below, for the next report that "
                     "still has something.")

    return theme.card(
        f"Needs a segment and trigger ({len(queue)} of {n_total})",
        html.Div([_row(registry, view, entry, mode) for entry in queue], className="anchor-list"),
        subtitle="Click a sentence in the report to select it, then *Use selected*; or highlight "
                 "the words and press *Use selection*. This records where a term came from. It is "
                 "evidence, not a verdict, so the term's status stays as it is.",
    )


def _row(registry, view: dict, entry: dict, mode: str):
    """One annotation to place: what its file claimed, why nothing was drawn, and the form."""
    palette = theme.palette(mode)
    record, row = entry["record"], entry["row"]
    code = entry["code"]

    head = [common.chip(source, theme.source_color(source, palette),
                        title=theme.SOURCE_HELP.get(source, ""))
            for source in entry["sources"]]
    head.append(common.code_chip(code, registry.search.label(code), palette["text"],
                                 title=registry.search.describe(code)))
    if record.get("confirmed") is False:
        head.append(common.chip("unconfirmed", palette["warning"],
                                title="its file carries this row with confirmed = False"))

    body = [
        html.Div(head, className="row-head"),
        # The same sentence the Document-annotations panel draws in red, so the two screens agree
        # about why this row is here, not each phrasing it their own way.
        html.Div(reader.evidence_line(record, entry["placement"]), className="stat-sub"),
    ]

    for claim in entry["claims"]:
        # The folded rows' own accounts. Two files that both failed to place a term usually failed
        # differently, and on a row nobody could evidence location those two guesses are most of the evidence
        # there is, so the merge names them instead of keeping only the one that won.
        body.append(html.Div(f"{claim['source']} also recorded {reader.claim_line(claim)}",
                             className="stat-sub"))

    context = (record.get("sentence_context") or "").strip()
    if context:
        # What its file thought the sentence was. On a row whose segment index belongs to a
        # different segmentation run this is the only surviving evidence of where it came from.
        body.append(html.Div(f"its file's sentence: “{context}”", className="stat-sub"))

    if entry["candidates"]:
        body.append(html.Div(
            [html.Span("possible origin: ", className="stat-sub")] + [
                html.Button(
                    f"#{hit['segment_idx']} “{hit['trigger_word']}”"
                    + (" ~" if hit.get("tier") == "inflected" else ""),
                    id={"type": "anchor", "key": entry["key"], "seg": hit["segment_idx"],
                        "trigger": hit["trigger_word"]},
                    className="copy-btn", n_clicks=0,
                    title=f"matched the ontology phrase “{hit['phrase']}”"
                          + (" by inflection" if hit.get("tier") == "inflected" else "")
                          + ", click to record this as where the term came from",
                ) for hit in entry["candidates"]
            ], className="row-actions"))

    # Open, not collapsed. Approve keeps this shut because adjudicating is the job and repairing is
    # The exception. Here repairing *is* the job, and a form behind a disclosure triangle on every
    # row would cost one click per annotation to do the only thing this screen is for.
    panel = editor.panel(registry, entry["key"], AT,
                         segment_idx=_none_int((row or {}).get("segment_idx")),
                         trigger_word=(row or {}).get("trigger_word", ""),
                         hpo_code=code, note=(row or {}).get("note", ""),
                         labels=(row or {}).get("labels"), mode=mode)
    panel.open = True
    body.append(panel)
    body.append(labelling.qualifier_chips((row or {}).get("labels"), mode))
    return html.Div(body, className="anchor-row")


def render_worklist(mode: str = "light"):
    """Every report that still has something to place, so the queue can be worked through."""
    registry = common.registry_or_none()
    if registry is None:
        return theme.empty("No data loaded.")

    counts = [(pid, len(queue_for(registry, pid)[0])) for pid in registry.patient_ids]
    open_reports = [(pid, n) for pid, n in counts if n]
    n_left = sum(n for _, n in open_reports)
    if not open_reports:
        return theme.card("Worklist",
                          theme.empty(f"Every annotation in all {len(counts)} reports has a "
                                      "segment and a trigger word."),
                          subtitle="Every annotation has its evidence located.")

    shown = open_reports[:WORKLIST_LIMIT]
    rows = [
        html.Button(f"{pid} · {n}",
                    # The sidebar's own id, so a click here goes through the one callback that
                    # writes ``store-patient``. A second writer is how two screens end up
                    # disagreeing about which patient is current.
                    id={"type": "pat", "pid": pid}, className="copy-btn", n_clicks=0,
                    title=f"{n} annotation(s) in {pid} still need a segment and a trigger word")
        for pid, n in shown
    ]
    more = (f" · {len(open_reports) - len(shown)} more not listed"
            if len(open_reports) > len(shown) else "")
    return theme.card(
        f"Worklist ({len(open_reports)} reports)",
        html.Div(rows, className="row-actions anchor-worklist"),
        subtitle=f"{n_left} annotation(s) left across the cohort{more}. Computed over every report, "
                 "so it is the one slow thing on this screen, it is recomputed when you save, and "
                 "only while this tab is open.",
    )


def _none_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
