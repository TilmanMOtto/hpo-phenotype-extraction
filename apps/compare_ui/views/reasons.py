"""One renderer per method, because the five methods do not have one kind of reason.

It would be easy to flatten all five into a table of (term, score, verdict). It would also destroy
the only thing this app has to say. PhenoBERT's reason is a span and a negation flag; AutoPCR's is
which of three routes fired; RAG-HPO's is a ranked menu and a Yes; PhenoJury's is eight models'
prose and a vote; TreePhenoRAG's is two scores against two thresholds and, for a miss, the ancestor
where the walk stopped. A common denominator would show all five as a number and explain none of
them.

So each ``kind`` gets its own panel, dispatched through :data:`RENDERERS`, and a payload whose
renderer is missing falls back to a labelled dump rather than to nothing -- a new adapter should
be visibly unfinished, not invisible.

Every panel that shows a reconstructed prompt says so, and every panel for a method whose model
output was never written says *that* too, in the method's own words from the roster. Showing our
reconstruction of AutoPCR's prompt without the sentence "the answer was not recorded" would be this
app claiming to show a model's reasoning while showing its own.
"""

from __future__ import annotations

from dash import html

from apps.compare_ui import theme
from apps.compare_ui.views import common


def render(bundle, selection):
    """The panel under the grid. ``None`` selection is the resting state, not an error."""
    if bundle is None:
        return html.Div()
    if not selection or not selection.get("hpo"):
        return theme.note(
            "Click any term, in the ground truth column or in a method's gutter, to see how that "
            "method decided it.", "info")

    hpo_id = selection["hpo"]
    method_key = selection.get("method") or "gold"
    if method_key == "gold":
        return _gold_panel(bundle, hpo_id)

    block = (bundle.get("methods") or {}).get(method_key)
    if block is None:
        return theme.note("No column named {!r} in this bundle.".format(method_key), "warn")

    reason = (block.get("reasons") or {}).get(hpo_id)
    outcome = next((m["outcome"] for m in block.get("marks") or ()
                    if m["hpo_id"] == hpo_id), "")

    body = [_main_result(bundle, block, hpo_id, outcome, reason)]
    if reason:
        renderer = RENDERERS.get(reason.get("kind"), _unknown)
        body.append(renderer(bundle, block, reason))
        body.append(_provenance(block, reason))
    else:
        body.append(theme.note(
            "This method recorded nothing about this term. It is neither in its prediction set "
            "nor in its evidence files.", "info"))

    return theme.panel(
        "{} · {}".format(block.get("label", method_key), _label(bundle, hpo_id)),
        html.Div(body),
        panel_id="reason-{}-{}".format(method_key, hpo_id.replace(":", "_")),
        subtitle=theme.KIND_TITLE.get((reason or {}).get("kind"), ""),
        markdown=_markdown(bundle, block, hpo_id, outcome, reason))


# -- shared furniture --------------------------------------------------------

def _main_result(bundle, block, hpo_id, outcome, reason):
    entry = (bundle.get("terms") or {}).get(hpo_id) or {}
    bits = [
        html.Div([
            html.Span(theme.OUTCOME_GLYPH.get(outcome, "?"), className="cmp-glyph"),
            html.Span(theme.OUTCOME_HELP.get(outcome, "not predicted and not in the ground truth")),
        ], className="cmp-chip cmp-headline", **{"data-outcome": outcome or "fn"}),
        html.Div(hpo_id, className="cmp-code"),
    ]
    head = html.Div(bits, className="cmp-reason-head")

    parts = [head]
    if entry.get("definition"):
        parts.append(html.Div(entry["definition"], className="cmp-def"))
    elif not entry.get("known", True):
        parts.append(theme.note(
            "The fixed HPO release does not carry this id, so it is either an obsolete spelling "
            "or a code no ontology defines.", "warn"))
    if (reason or {}).get("why"):
        parts.append(html.Div(reason["why"], className="cmp-why"))
    return html.Div(parts)


def _provenance(block, reason):
    bits = []
    if reason.get("source"):
        bits.append(html.Div(["read from ", html.Code(reason["source"])], className="cmp-prov"))
    if reason.get("answer_recorded") is False and reason.get("answer_note"):
        bits.append(theme.note(reason["answer_note"], "warn"))
    if block.get("evidence_note"):
        bits.append(theme.note(block["evidence_note"], "warn"))
    if block.get("note"):
        bits.append(theme.note(block["note"], "info"))
    return html.Div(bits)


def _gold_panel(bundle, hpo_id):
    """What the *curator* recorded -- the one column that is not a method."""
    row = next((g for g in bundle.get("gold") or () if g["hpo_id"] == hpo_id), None)
    if row is None:
        return theme.note("{} is not a curated annotated term for this report.".format(hpo_id), "info")

    entry = (bundle.get("terms") or {}).get(hpo_id) or {}
    rows = [
        ("HPO id", hpo_id),
        ("as the ground truth spells it", row.get("gold_code") or hpo_id),
        ("sentence", "—" if row.get("segment_idx") is None else "#{}".format(row["segment_idx"])),
        ("trigger word", row.get("trigger") or ",  none recorded"),
        ("source", row.get("source") or "—"),
        ("qualifiers", ", ".join(row.get("qualifiers") or ()) or "—"),
        ("curator note", row.get("note") or "—"),
    ]
    body = [html.Div(entry.get("definition") or "", className="cmp-def")]
    body.append(common.table(["", ""], [[html.Span(k, className="cmp-key"), v] for k, v in rows]))

    who = []
    for key, block in sorted((bundle.get("methods") or {}).items()):
        outcome = next((m["outcome"] for m in block.get("marks") or ()
                        if m["hpo_id"] == hpo_id), "")
        who.append([block.get("short") or key,
                    html.Span([html.Span(theme.OUTCOME_GLYPH.get(outcome, "?"),
                                         className="cmp-glyph"),
                               theme.OUTCOME_HELP.get(outcome, "—")],
                              className="cmp-chip", **{"data-outcome": outcome or "fn"})])
    body.append(common.table(["method", "on this term"], who))

    return theme.panel(
        "Curated ground truth · " + _label(bundle, hpo_id), html.Div(body),
        panel_id="reason-gold-" + hpo_id.replace(":", "_"),
        subtitle="what a curator recorded, and who found it",
        markdown=common.md_table(["field", "value"], [[k, v] for k, v in rows]))


# -- per-kind panels ---------------------------------------------------------

def _tagger(bundle, block, reason):
    """PhenoBERT: the spans it matched, and the two flags that silently drop one."""
    rows = []
    for det in reason.get("detections") or ():
        flags = []
        if det.get("negated"):
            flags.append(html.Span("negated", className="cmp-flag", **{"data-tone": "warning"}))
        if not det.get("resolved", True):
            flags.append(html.Span("unresolved id", className="cmp-flag",
                                   **{"data-tone": "critical"}))
        rows.append([
            html.Code(det.get("phrase") or ""),
            common.fmt(det.get("score"), 2),
            "{}–{}".format(det.get("start"), det.get("end")),
            html.Div(flags or "—"),
        ])
    if not rows:
        return theme.note("No span in this report was tagged with this term.", "info")
    return html.Div([
        common.table(["matched phrase", "score", "offsets", "flags"], rows),
        theme.note("A negated or unresolved detection never reaches the prediction set. That is a "
                   "different miss from never detecting the phrase at all, and only this file can "
                   "tell them apart.", "info"),
    ])


def _linker(bundle, block, reason):
    """AutoPCR: which of the three routes fired, and the menu the linker was choosing from."""
    parts = []
    for mention in reason.get("mentions") or ():
        parts.append(html.Div([
            html.Div([
                html.Code(mention.get("phrase") or ""),
                html.Span(mention.get("route") or "unknown", className="cmp-flag",
                          **{"data-tone": "accent"}, title=mention.get("route_help") or ""),
                html.Span(_score_text(mention), className="cmp-report-sub"),
            ], className="cmp-reason-head"),
            html.Div(mention.get("route_help") or "", className="cmp-why"),
            _menu(bundle, mention.get("menu")),
        ], className="cmp-mention"))
    if not parts:
        return theme.note("No mention of this term was recorded for this report.", "info")
    return html.Div(parts)


def _score_text(mention) -> str:
    score = mention.get("score")
    if score is None:
        return ""
    if score == -1.0:
        return "score −1.0 (the sentinel meaning the linker chose, not the retriever)"
    return "score " + common.fmt(score, 3)


def _rag(bundle, block, reason):
    """RAG-HPO: the finding it extracted and the candidates it ranked for it."""
    parts = []
    for selection in reason.get("selections") or ():
        parts.append(html.Div([
            html.Div([
                html.Span("from", className="cmp-key"),
                html.Span(selection.get("source_text") or "", className="cmp-quote"),
            ], className="cmp-reason-head"),
            html.Div("taken at rank {} (cosine {})".format(
                selection.get("rank"), common.fmt(selection.get("cosine_sim"), 3)),
                className="cmp-why"),
            _menu(bundle, selection.get("menu")),
        ], className="cmp-mention"))
    if not parts:
        return theme.note("No finding in this report retrieved this term.", "info")
    return html.Div(parts)


def _menu(bundle, menu):
    """A ranked candidate list, with the taken row marked."""
    if not menu:
        return theme.note("No candidate list was written for this phrase.", "info")
    rows = []
    for candidate in menu:
        rows.append([
            candidate.get("rank"),
            common.term_chip(bundle, candidate.get("hpo_id") or "", clickable=False)
            if (candidate.get("hpo_id") in (bundle.get("terms") or {}))
            else html.Span(candidate.get("hpo_label") or candidate.get("hpo_id") or ""),
            candidate.get("hpo_id") or "",
            common.fmt(candidate.get("cosine_sim"), 3),
            html.Span("chosen", className="cmp-flag", **{"data-tone": "good"})
            if candidate.get("chosen") else "",
        ])
    return common.table(["#", "candidate", "id", "cosine", ""], rows)


def _jury(bundle, block, reason):
    """PhenoJury: who voted, what the fold required, and the words each juror actually wrote."""
    parts = []

    config = [
        ("votes", "{} of the {} jurors in the selected subset".format(
            reason.get("votes"), len([j for j in reason.get("jurors") or () if j["in_subset"]]))),
        ("threshold", "k = {}".format(reason.get("k")) if reason.get("k") is not None
         else "not recorded"),
        ("unit", reason.get("unit") or "—"),
        ("rule", reason.get("rule") or "—"),
        ("outer fold", reason.get("outer_fold")),
    ]
    parts.append(common.table(["", ""], [[html.Span(k, className="cmp-key"), v]
                                         for k, v in config]))

    if not reason.get("config_known"):
        parts.append(theme.note(
            "The fold this report was evaluated in could not be resolved, so the threshold shown "
            "is a default rather than the one that produced the prediction.", "warn"))
    if reason.get("vote_recomputed") is False:
        parts.append(theme.note(
            "The PhenoJury protocol module could not be imported, so the vote count is a distinct-"
            "juror count over the whole report, right for the report unit and an upper bound "
            "otherwise. It is not this method's own number.", "warn"))

    rows = []
    for juror in reason.get("jurors") or ():
        rows.append([
            juror["model"],
            html.Span("in subset" if juror["in_subset"] else "not selected", className="cmp-flag",
                      **{"data-tone": "accent" if juror["in_subset"] else "neutral"}),
            html.Span("voted", className="cmp-flag", **{"data-tone": "good"})
            if juror["voted"] else "—",
            ", ".join("#{}".format(s) for s in juror["sentences"]) or "—",
        ])
    parts.append(common.table(["juror", "", "", "sentences"], rows))

    generations = block.get("generations") or {}
    shown = []
    for model, sent in reason.get("refs") or ():
        record = generations.get("{}|{}".format(model, sent))
        if record is None:
            continue
        shown.append(html.Div([
            html.Div(["{} · sentence #{}".format(model, sent)], className="cmp-key"),
            html.Div(record.get("sentence_text") or "", className="cmp-quote"),
            html.Pre(record.get("llm_output") or "",
                     className="cmp-generation" + (" is-truncated" if record.get("truncated")
                                                   else "")),
        ], className="cmp-mention"))
    if shown:
        parts.append(html.Div([html.Div("What the jurors wrote", className="cmp-subhead")]
                              + shown))
    else:
        parts.append(theme.note(
            "No juror generation is recorded for this term, none of them named it, so there is "
            "no sentence to show.", "info"))
    return html.Div(parts)


def _tree(bundle, block, reason):
    """TreePhenoRAG: the two gates, the calls behind them, and where a miss was blocked."""
    if not reason.get("available", True):
        return theme.note(reason.get("why") or "No re-run for this report.", "warn")

    config = reason.get("config") or {}
    tiles = theme.stat_row([
        theme.stat("pruning score", common.fmt(reason.get("prune_score"), 4),
                   "tau_prune {}".format(common.fmt(config.get("tau_prune"), 5)),
                   tone="good" if reason.get("expanded") else "warning",
                   help_text="the pooled verifier confidence at this node; it must clear "
                             "tau_prune for the walk to reveal the node's children"),
        theme.stat("reached at", common.fmt(reason.get("bottleneck_r"), 4), "bottleneck r",
                   tone="good" if reason.get("visited") else "critical",
                   help_text="the largest tau_prune at which a path from a root still reaches "
                             "this node, below it the node is never scored at all"),
        theme.stat("acceptance", common.fmt(reason.get("accept_score"), 4),
                   "tau_accept {}".format(common.fmt(config.get("tau_accept"), 3)),
                   tone="good" if reason.get("accepted") else "warning",
                   help_text="the second pooling; the node is emitted only if this clears "
                             "tau_accept"),
        theme.stat("depth", reason.get("depth"), "in the ontology"),
    ])

    parts = [tiles]

    if reason.get("lost_at") and reason.get("pruned"):
        parts.append(theme.note(
            "Blocked at {} ({}), depth {}, bucket “{}”, cause “{}”. That ancestor was scored at "
            "{} and never expanded, so the walk never reached this term.".format(
                reason.get("lost_at_label") or reason["lost_at"], reason["lost_at"],
                reason.get("blocking_depth"), reason.get("bucket"), reason.get("cause"),
                common.fmt(reason.get("lost_at_score"), 5)),
            "warn"))
    elif reason.get("bucket"):
        parts.append(theme.note(
            "Reached and scored, then lost at acceptance, bucket “{}”.".format(reason["bucket"]),
            "warn"))

    if reason.get("bucket") and not reason.get("evidence_available", True):
        # Stated, not left implicit: without the curated segment, "retrieval" can only mean
        # "no sentence was scored at all", so a term whose ten retrieved sentences all missed the
        # finding is indistinguishable here from one the verifier rejected on the right sentence.
        parts.append(theme.note(
            "The curated ground truth records no segment for this term, so the retrieval/judgement split "
            "is not decidable here, read the bucket as a lower bound on retrieval loss.", "info"))

    calls = reason.get("calls") or []
    if calls:
        rows = [[c.get("rank"),
                 "#{}".format(c["sent_index"]) if c.get("sent_index") is not None else "—",
                 html.Span(c.get("sentence") or "", className="cmp-quote"),
                 common.fmt(c.get("cosine_sim"), 3),
                 common.fmt(c.get("margin"), 3),
                 html.Span(c.get("verdict") or "", className="cmp-flag",
                           **{"data-tone": "good" if c.get("verdict") == "Yes" else "neutral"})]
                for c in calls]
        parts.append(html.Div("Verifier calls, best margin first", className="cmp-subhead"))
        parts.append(common.table(
            ["rank", "sentence", "what it was shown", "cosine", "margin", "verdict"], rows))
    else:
        parts.append(theme.note(
            "No verifier call is cached for this node, which means the walk never scored it.",
            "info"))

    parts.append(html.Div(
        "Re-run at this report's own fold: index {}, pooling {} / {}, S={}.".format(
            config.get("index") or "?", config.get("pool_pr"), config.get("pool_acc"),
            config.get("S")),
        className="cmp-prov"))
    return html.Div(parts)


def _unknown(bundle, block, reason):
    rows = [[html.Span(k, className="cmp-key"), str(v)[:300]]
            for k, v in sorted(reason.items()) if k not in ("kind", "why", "source")]
    return html.Div([
        theme.note("No renderer for reason kind {!r}, showing the payload as recorded.".format(
            reason.get("kind")), "warn"),
        common.table(["field", "value"], rows),
    ])


RENDERERS = {
    "tagger": _tagger,
    "linker": _linker,
    "rag": _rag,
    "jury": _jury,
    "tree": _tree,
}


# -- markdown ----------------------------------------------------------------

def _markdown(bundle, block, hpo_id, outcome, reason) -> str:
    """The panel as Markdown, so a worked example can go straight into the thesis.

    The 2026-09-16 meeting asked for "two or three worked examples with report text, annotated term and
    per-system prediction". This is that, one method at a time, without retyping it.
    """
    lines = [
        "**{}, {} ({})**".format(block.get("label", ""), _label(bundle, hpo_id), hpo_id),
        "",
        "- outcome: {} ({})".format(outcome or "—", theme.OUTCOME_HELP.get(outcome, "")),
    ]
    if reason and reason.get("why"):
        lines.append("- " + reason["why"])
    if reason and reason.get("source"):
        lines.append("- read from `{}`".format(reason["source"]))
    if block.get("evidence_note"):
        lines += ["", "> " + block["evidence_note"]]
    return "\n".join(lines)


def _label(bundle, hpo_id: str) -> str:
    return ((bundle.get("terms") or {}).get(hpo_id) or {}).get("label") or hpo_id
