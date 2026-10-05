"""Annotate, read what one SLM wrote, record the phenotypes in it, compare against PhenoBERT.

The one screen in this app where a person adds information rather than reading it back. Every other
view re-runs artifacts. This one answers the question no artifact can, because nothing on disk
distinguishes *the model never wrote the finding* from *the model wrote it and PhenoBERT failed to
link it*. Both land in the autopsy as ``not_linked``, and they call for opposite fixes.

The unit is one ``(model, report, sentence)`` reply, which is the text PhenoBERT was handed.
So the SLM is held fixed and only the linker varies, and what comes out is a measurement of
PhenoBERT, not of the ensemble.

Three things are deliberate:

**The reply is drawn by ``patient._reply_body``**, the same renderer the deep dive uses, spans, tags
and all. Reusing it means the reader is annotating the surface they already know how to
read, and that the two screens cannot drift apart.

**The term picker never ships the ontology.** The dropdown starts empty and is filled by a callback
on ``search_value`` against the server-side index, capped at 50 hits. Putting all ~19 000 terms in a
browser store is the single reason ``app/annotation_ui`` is unpleasant over a tunnel.

**"No phenotype here" is a button, not the absence of one.** A skipped reply is a reading, and it is
the denominator of PhenoBERT's precision. Leaving it implicit would make the rate uninterpretable.
"""

from __future__ import annotations

import logging

import dash
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from .. import annotations as ann
from .. import state, theme
from . import common, patient

logger = logging.getLogger(__name__)

VIEW_ID = "annotate"

#: How many queue rows to draw. The queue is a navigation aid, not a table to read: past a screenful
#: The useful control is the model filter, not more rows.
QUEUE_LIMIT = 300


def layout():
    """The Dash layout of this view."""
    return html.Div([
        html.Div(id="an-status"),
        html.Div([
            html.Div([
                html.Label("Models", className="field-label"),
                dcc.Checklist(id="an-models", options=[], value=[], className="check-list"),
                html.Label("Show", className="field-label", style={"marginTop": "8px"}),
                dcc.RadioItems(
                    id="an-filter", inline=True, value="todo",
                    options=[{"label": " to do", "value": "todo"},
                             {"label": " done", "value": "done"},
                             {"label": " all", "value": "all"}],
                ),
                html.Div(id="an-progress", className="stat-sub", style={"marginTop": "8px"}),
                html.Div(id="an-queue", className="tbl-scroll", style={"marginTop": "8px"}),
            ], className="an-queue-col"),
            html.Div(id="an-reply-col", className="an-reply-col"),
        ], className="an-split"),
        html.Div(id="an-agreement"),
        # Which reply is open, and a counter bumped on every save so the panels rebuild.
        dcc.Store(id="an-current", data=None),
        dcc.Store(id="an-saved", data=0),
    ], className="view")


# ──────────────────────────────────────────────────────────────────────────────
# The queue
# ──────────────────────────────────────────────────────────────────────────────
def queue_rows(bundle: dict, cfg: dict, folded: dict, models: list[str],
               which: str = "todo") -> list[dict]:
    """Every reply in scope, with the state somebody has left it in.

    Scope is the current subset and model selection. Only replies where the model actually wrote
    something are offered: an empty generation has no phenotypes to read, and putting hundreds of
    them in the queue would bury the ones worth an opinion.
    """
    chosen = set(models or bundle["models"])
    rows = []
    for report_id in common.report_ids(bundle, cfg):
        for sent_num in sorted(bundle["sentences"].get(report_id, {})):
            for model in bundle["models"]:
                if model not in chosen:
                    continue
                reply = bundle["built"]["replies"].get((model, report_id, sent_num))
                if reply is None or not reply["wrote"]:
                    continue
                key = ann.key_of(bundle["run_id"], model, report_id, sent_num)
                saved = folded["replies"].get(key) or {}
                status = saved.get("status", "")
                if which == "todo" and status:
                    continue
                if which == "done" and not status:
                    continue
                rows.append({
                    "key": key, "model": model, "report_id": report_id,
                    "sentence_number": sent_num, "status": status,
                    "n_terms": len(saved.get("hpo_codes") or ()),
                    "n_pb": len(ann.phenobert_terms(bundle, model, report_id,
                                                    sent_num)["positive"]),
                })
    return rows


def render_queue(rows: list[dict], current: str | None):
    """The clickable queue. One button per reply, because a table cell is not a click target."""
    if not rows:
        return theme.empty("Nothing in the queue at this filter.")
    items = []
    for row in rows[:QUEUE_LIMIT]:
        mark = {"annotated": "✓", "skipped": "∅"}.get(row["status"], "·")
        items.append(html.Button(
            f"{mark} {row['report_id']} · s{row['sentence_number']} · {row['model']}"
            f"  (PB {row['n_pb']})",
            id={"type": "an-pick", "key": row["key"]},
            className="an-queue-item" + (" an-queue-current" if row["key"] == current else ""),
            title=f"{row['status'] or 'not yet read'}, "
                  f"{row['n_terms']} term(s) recorded, PhenoBERT found {row['n_pb']}",
        ))
    more = ([html.Div(f"… and {len(rows) - QUEUE_LIMIT} more; narrow with the model filter.",
                      className="stat-sub")] if len(rows) > QUEUE_LIMIT else [])
    return html.Div(items + more, className="an-queue-list")


def render_progress(bundle: dict, cfg: dict, folded: dict, models: list[str]) -> str:
    """One line of annotation progress: replies read, annotated, without phenotype, left."""
    todo = len(queue_rows(bundle, cfg, folded, models, "todo"))
    done_rows = queue_rows(bundle, cfg, folded, models, "done")
    skipped = sum(1 for r in done_rows if r["status"] == "skipped")
    total = todo + len(done_rows)
    return (f"{len(done_rows)} of {total} replies read "
            f"({len(done_rows) - skipped} annotated, {skipped} with no phenotype) · "
            f"{todo} left")


# ──────────────────────────────────────────────────────────────────────────────
# The reply being annotated
# ──────────────────────────────────────────────────────────────────────────────
def parse_key(key: str | None) -> tuple[str, str, int] | None:
    """``run|model|report|sentence`` back into its parts, or ``None`` if it does not parse."""
    if not key:
        return None
    parts = key.split("|")
    if len(parts) != 4:
        return None
    try:
        return parts[1], parts[2], int(parts[3])
    except ValueError:
        return None


def render_reply(bundle: dict, cfg: dict, key: str | None, folded: dict, mode: str):
    """The whole right-hand column: the sentence, the generation, the picker, the verdict."""
    parsed = parse_key(key)
    if parsed is None:
        return common.empty_panel(
            "Pick a reply from the queue on the left. Each one is what a single model wrote about a "
            "single sentence, the exact text PhenoBERT was given.", "an-reply", "Annotate a reply")
    model, report_id, sent_num = parsed

    reply = bundle["built"]["replies"].get((model, report_id, sent_num))
    if reply is None:
        return common.empty_panel(f"No reply on record for {model} on {report_id} sentence "
                                  f"{sent_num}.", "an-reply", "Annotate a reply")

    saved = folded["replies"].get(key) or {}
    pb_rows = bundle["pb_index"]["by_sentence"].get((model, report_id, sent_num), [])
    cell = common.cell_chip(report_id)

    header = html.Div([
        html.Span(f"{report_id} · sentence {sent_num} · ", className="an-head-id"),
        html.Span(model, className="pill"),
        cell if cell is not None else None,
    ], className="an-head")

    source = theme.panel(
        "The sentence the model was given", html.Div(
            bundle["sentences"].get(report_id, {}).get(sent_num, "(not recorded)"),
            className="prompt-box"),
        panel_id="an-source",
        subtitle="Stanza's segmentation, as persisted in llm_extractions_*.jsonl.")

    generation = theme.panel(
        "What the model wrote",
        patient._reply_body(bundle, reply, pb_rows, mode, False),
        panel_id="an-generation",
        subtitle="Underlines are PhenoBERT's matches on this text, tagged with the term each "
                 "normalised to. Struck through means it read the finding as absent.")

    picker = theme.panel(
        "The phenotypes you read in it",
        html.Div([
            dcc.Dropdown(id="an-hpo", options=[], value=saved.get("hpo_codes") or [],
                         multi=True, placeholder="type a phenotype, a synonym or an HPO id"),
            dcc.Input(id="an-note", value=saved.get("note", ""), type="text",
                      className="path-input", placeholder="note (optional)",
                      style={"marginTop": "8px"}),
            html.Div([
                html.Button("Save", id="an-save", className="primary-btn"),
                html.Button("No phenotype here", id="an-skip", className="copy-btn",
                            style={"marginLeft": "8px"}),
                html.Button("Next unread", id="an-next", className="copy-btn",
                            style={"marginLeft": "8px"}),
            ], style={"marginTop": "10px"}),
            html.Div(id="an-feedback", style={"marginTop": "8px"}),
        ]),
        panel_id="an-picker",
        subtitle="Record what the text asserts, not what the report is about, this is a reading "
                 "of the generation. Save with an empty list is not the same as No phenotype "
                 "here; the second is a positive statement that there is nothing to find.")

    return html.Div([header, source, generation, picker,
                     render_verdict(bundle, key, saved, mode)])


def render_verdict(bundle: dict, key: str, saved: dict, mode: str):
    """This reply's own agreement panel, what you saw, what PhenoBERT saw, and the difference."""
    if not saved.get("status"):
        return None
    row = ann.compare_reply(bundle, {**saved, "key": key})
    p = theme.palette(mode)

    def chips(codes, colour, relations=None):
        if not codes:
            return html.Span("none", className="stat-sub")
        return html.Div([
            common.term_chip(bundle, code,
                             suffix=(f" · {relations[code]}" if relations and code in relations
                                     else ""),
                             style={"background": colour})
            for code in codes], className="chip-row")

    return theme.panel(
        "You vs PhenoBERT, on this reply",
        html.Div([
            html.Div([html.Strong("agreed: "),
                      chips(sorted(set(row["manual"]) & set(row["phenobert"])), p["good"])]),
            html.Div([html.Strong("PhenoBERT missed: "),
                      chips(row["pb_missed"], p["warning"], row["missed_relations"])],
                     style={"marginTop": "6px"}),
            html.Div([html.Strong("PhenoBERT added: "),
                      chips(row["pb_spurious"], p["critical"], row["spurious_relations"])],
                     style={"marginTop": "6px"}),
            html.Div([html.Strong("PhenoBERT read as absent: "),
                      chips(row["pb_negated"], p["neutral"])],
                     style={"marginTop": "6px"}),
        ]),
        panel_id="an-verdict",
        subtitle="The tag after a disagreeing term is how far it sits from the other side in the "
                 "ontology, a parent of the right term is a different failure from an unrelated "
                 "one. Terms it read as absent are a decision, not a miss, and score neither way.")


# ──────────────────────────────────────────────────────────────────────────────
# The roll-up
# ──────────────────────────────────────────────────────────────────────────────
def render_agreement(bundle: dict, cfg: dict, folded: dict, mode: str):
    """PhenoBERT as a linker, over everything read so far in this run."""
    result = ann.compare(bundle, folded, common.report_ids(bundle, cfg))
    if not result["n_replies"]:
        return common.empty_panel(
            "Nothing annotated yet in this run. Read a few replies and this becomes PhenoBERT's "
            "precision and recall as a linker, with your reading as the reference.",
            "an-agreement", "PhenoBERT as a linker")

    totals = result["totals"]
    tiles = [
        theme.stat("replies read", str(result["n_replies"]),
                   f"{result['n_skipped']} with no phenotype"),
        theme.stat("PhenoBERT precision", common.num(totals["precision"]),
                   f"{totals['n_agreed']} of {totals['n_phenobert']} terms it emitted"),
        theme.stat("PhenoBERT recall", common.num(totals["recall"]),
                   f"{totals['n_agreed']} of {totals['n_manual']} terms you read"),
        theme.stat("F1", common.num(totals["f1"])),
    ]

    headers = ["model", "replies", "you", "PhenoBERT", "agreed", "it missed", "it added",
               "precision", "recall"]
    rows = [[model,
             s["n_replies"], s["n_manual"], s["n_phenobert"], s["n_agreed"],
             s["n_pb_missed"], s["n_pb_spurious"],
             common.num(s["precision"]), common.num(s["recall"])]
            for model, s in sorted(result["per_model"].items(),
                                   key=lambda kv: -kv[1]["n_replies"])]

    rel_rows = []
    for side, label in (("pb_missed", "it missed"), ("pb_spurious", "it added")):
        for relation, n in sorted(result["relations"][side].items(), key=lambda kv: -kv[1]):
            rel_rows.append([label, relation, n, theme.RELATION_HELP.get(relation, "")])

    body = [
        theme.stat_row(tiles),
        common.table(headers, rows),
    ]
    if rel_rows:
        body.append(html.Div([
            html.H4("How wrong each disagreement is", className="panel-title",
                    style={"marginTop": "14px", "fontSize": "0.95rem"}),
            common.table(["side", "relation", "n", "meaning"], rel_rows, numeric_from=2),
        ]))

    return theme.panel(
        "PhenoBERT as a linker", html.Div(body), panel_id="an-agreement",
        subtitle=f"Over the {result['n_replies']} reply/replies read so far, not a cohort "
                 f"statistic, and it moves as you annotate. Precision and recall are PhenoBERT's, "
                 f"with your reading as the reference. {common.describe(cfg)}",
        markdown=common.md_table(headers, rows))


# ──────────────────────────────────────────────────────────────────────────────
# callbacks
# ──────────────────────────────────────────────────────────────────────────────
def log_for(bundle: dict):
    """The annotation log for this bundle's run, or ``None`` when it cannot be opened."""
    if not state.has_registry():
        return None
    registry = state.get_registry()
    directory = ann.default_dir(bundle["run_dir"], registry.annotations_dir)
    try:
        return _log_cache(directory)
    except Exception as exc:  # noqa: BLE001 - reported on screen, never a 500
        logger.error("could not open the annotation log at %s: %s", directory, exc)
        return None


_LOGS: dict[str, ann.AnnotationLog] = {}


def _log_cache(directory: str) -> ann.AnnotationLog:
    """One log object per directory per process, it caches the file it is also appending to."""
    if directory not in _LOGS:
        import getpass

        try:
            author = getpass.getuser()
        except Exception:  # noqa: BLE001 - a nameless author beats a crash on save
            author = "unknown"
        _LOGS[directory] = ann.AnnotationLog(directory, author=author)
    return _LOGS[directory]


def register(app: dash.Dash) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("an-models", "options"),
        Output("an-models", "value"),
        Input("run-a", "value"),
    )
    def model_filter(run_a):
        bundle = common.bundle_for(run_a)
        if bundle is None:
            return [], []
        options = [{"label": f" {m}", "value": m} for m in bundle["models"]]
        # Open on one model, not eight: annotating is per reply, and eight models over
        # twenty reports is thousands of queue rows nobody asked for yet.
        return options, bundle["models"][:1]

    @app.callback(
        Output("an-status", "children"),
        Output("an-queue", "children"),
        Output("an-progress", "children"),
        Output("an-reply-col", "children"),
        Output("an-agreement", "children"),
        Output(f"sig-{VIEW_ID}", "data"),
        Input("tabs", "value"),
        Input("run-a", "value"),
        Input("store-config", "data"),
        Input("store-theme", "data"),
        Input("an-models", "value"),
        Input("an-filter", "value"),
        Input("an-current", "data"),
        Input("an-saved", "data"),
        State(f"sig-{VIEW_ID}", "data"),
    )
    def render(tab, run_a, config, mode, models, which, current, saved, previous):
        key = common.render_key(VIEW_ID, run_a, config, mode, models, which, current, saved)
        if not common.should_render(VIEW_ID, tab, key, previous):
            raise PreventUpdate
        bundle = common.bundle_for(run_a)
        if bundle is None:
            return common.no_data(run_a, "Annotate"), None, None, None, None, key
        cfg = common.resolve(config, bundle)
        mode = mode or "light"

        log = log_for(bundle)
        status = None
        if log is None:
            status = theme.note("No annotation log, nothing can be saved. Pass "
                                "--annotations-dir to put it somewhere writable.", "danger")
        else:
            problem = ann.writability_problem(log.dir)
            if problem:
                status = theme.note(problem, "danger")

        folded = log.fold() if log else {"replies": {}, "n_skipped": 0}
        rows = queue_rows(bundle, cfg, folded, models, which or "todo")
        return (status,
                render_queue(rows, current),
                render_progress(bundle, cfg, folded, models),
                render_reply(bundle, cfg, current, folded, mode),
                render_agreement(bundle, cfg, folded, mode),
                key)

    @app.callback(
        Output("an-hpo", "options"),
        Input("an-hpo", "search_value"),
        State("an-hpo", "value"),
    )
    def search_hpo(query, chosen):
        """Server-side search: the dropdown never holds more than one page of the ontology."""
        from apps.curation_ui.search import MAX_RESULTS

        if not state.has_registry():
            return []
        index = state.get_registry().search
        if index is None:
            return []
        options = index.search(query or "")
        # Keep the chosen values in the option list, or Dash renders them blank the moment the
        # query that found them stops matching.
        have = {o["value"] for o in options}
        options = [index.option(code) for code in (chosen or []) if code not in have] + options
        return options[:MAX_RESULTS]

    @app.callback(
        Output("an-current", "data"),
        Input({"type": "an-pick", "key": dash.ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def pick(clicks):
        if not any(clicks or ()):
            # The queue was redrawn, not clicked: every button is new and reports n_clicks=None.
            raise PreventUpdate
        return picked_key(dash.callback_context)

    @app.callback(
        Output("an-saved", "data"),
        Output("an-feedback", "children"),
        Input("an-save", "n_clicks"),
        Input("an-skip", "n_clicks"),
        State("an-current", "data"),
        State("an-hpo", "value"),
        State("an-note", "value"),
        State("run-a", "value"),
        State("an-saved", "data"),
        prevent_initial_call=True,
    )
    def save(n_save, n_skip, key, codes, note, run_a, saved):
        trigger = _trigger_id(dash.callback_context)
        if trigger not in ("an-save", "an-skip"):
            raise PreventUpdate
        if not (n_save if trigger == "an-save" else n_skip):
            # Saving bumps an-saved, which redraws this panel and recreates both buttons. Without
            # this the redraw would fire the callback again and record a second, phantom event.
            raise PreventUpdate
        bundle = common.bundle_for(run_a)
        parsed = parse_key(key)
        if bundle is None or parsed is None:
            return dash.no_update, theme.note("Pick a reply first.", "warn")
        log = log_for(bundle)
        if log is None:
            return dash.no_update, theme.note("No writable annotation log.", "danger")

        model, report_id, sent_num = parsed
        fields = {"key": key, "run_id": bundle["run_id"], "cohort": bundle["cohort"],
                  "prompt_key": bundle.get("prompt_key", ""), "model": model,
                  "report_id": report_id, "sentence_number": sent_num,
                  "note": note or ""}
        try:
            if trigger == "an-skip":
                log.append("skip", **fields)
            else:
                log.append("annotate", hpo_codes=list(codes or []), **fields)
            folded = log.fold()
            log.commit(ann.term_rows(folded),
                       ann.agreement_rows(ann.compare(bundle, folded)))
        except OSError as exc:
            return dash.no_update, theme.note(f"Could not save: {exc}", "danger")
        what = ("recorded: no phenotype here" if trigger == "an-skip"
                else f"saved {len(codes or [])} term(s)")
        return (saved or 0) + 1, theme.note(what, "info")


def picked_key(ctx) -> str:
    """The queue key whose button fired, or no update.

    Lifted out of the callback because its inputs are pattern-matching: ``callback_smoke`` cannot
    resolve ``ALL`` without a browser, so the selftest drives this function directly instead.
    """
    triggered = getattr(ctx, "triggered_id", None)
    if isinstance(triggered, dict) and triggered.get("type") == "an-pick":
        return triggered.get("key")
    return dash.no_update


def _trigger_id(ctx) -> str:
    triggered = getattr(ctx, "triggered_id", None)
    return triggered if isinstance(triggered, str) else ""
