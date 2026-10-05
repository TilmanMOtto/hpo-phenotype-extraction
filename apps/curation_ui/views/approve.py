"""Approve mode, adjudicating what exists and what was proposed.

Two blocks, in the order the work happens.

**Pending suggestions.** What Edit mode proposed, each with its segment and trigger word, and three
buttons: approve, reject, needs-work. Approving puts the term in the curated ground truth. Rejecting keeps
it in the log and out of the ground truth. Needs-work does neither and says so.

**Existing annotations.** The rows to *rule on*, which is not the same as every row the inputs
carry. ``annotations_confirmed.csv`` is a confirmed pass over the prior_annotation set, so a term it carries
is the annotation to judge and the prior_annotation row for the same code is that row's predecessor, not a
second opinion, asking about both would put the same question on the screen twice and let the two
answers disagree. Where the confirmed file **dropped** a term the prior_annotation file still carries, the
prior_annotation row takes its place and says so, because that drop is itself a judgement somebody made and
it has to be confirmable rather than invisible.

``prior_annotation_2`` and PhenoBERT are **reference**. They stay drawn on the report, listed in the document
panel, and named in the agreement chip on every row, knowing a second annotator also carries this
code is what makes a verdict easy, but neither takes a verdict, because nothing about either can
enter the curated ground truth. A PhenoBERT finding worth keeping is added as a suggestion from Edit mode,
which is the path that gives it a segment and a trigger word.

Each card names which files carry the code, ``prior_annotation only``, ``daphne + prior_annotation_2``, ``all sources``,
``PhenoBERT only``, which is the disagreement this app exists to resolve. That comparison is made
against the sources carrying something **for this patient**, not against every file that loaded: a
file with no row for this report is not a file that disagrees with the ones that have one, and
saying so put a disagreement chip on every term of every report the confirmed pass never reached.

**One card per term, one verdict per annotation.** A term annotated twice is two rows in the data
model and one phenotype in a curator's head, so the two share a card and a heading, but each keeps
its own evidence block, its own keep / remove / needs-work buttons and its own editor, because
keeping one occurrence must never silently settle the other.

**Ordered by segment**, within the bands below. Sorting by source and code meant this column and
the report beside it were two different orders, so checking a row against its sentence was a hunt.
Now a row's position is a hint about where its evidence is.

Rows the log has already decided that no current input nominates are shown too, at the end and
labelled: a source that was retired (every ``raw|…`` verdict from before ``prior_annotation`` replaced it),
or a detection kept back when detections were adjudicable. A decided row that vanished from the
screen would be a decision nobody could revisit, and one carrying an open deletion proposal would be
a question nobody could answer.

**Qualifiers ride on the annotation.** Whether a finding belongs to a relative, is negated, is only
a lab value, or is implied, not named, is shown as chips on the row and changed in the same
*Edit evidence* form as the trigger word, one place, because a field editable from two places is a
field whose two copies disagree. That replaced a collapsed block per row carrying nineteen
checkboxes, a grade and its own Save button.

**Every row can be repaired, not only judged.** :mod:`apps.curation_ui.views.editor` sits
collapsed under each one: the segment, the trigger word, the term and a note. That counts most for
the sources that carry their own evidence, a trigger word written against a different segmentation
lands on the wrong sentence, and before the editor existed the only repair was a lexical candidate
button, which for a rich annotation is not offered at all. An edit writes its own event and never
touches the verdict.

A row somebody has **proposed deleting** (in Edit mode) is banner-marked with its reason and its
proposer, and sorts to the top. It needs no controls of its own: *Remove* accepts the proposal and
*Keep* rejects it, which is the same judgement the three buttons already express. A parallel set of
approve/reject buttons doing the same thing to the same row would only invite the two to disagree.

A row whose trigger word could not be found in the report, the file named words the report does
not contain, carries its :mod:`locate` candidates as one-click **evidence location** buttons: "this came from
segment #7, trigger *seizures*". Locating is not a verdict and does not imply one. It records where
the term came from, which is what makes the verdict checkable later. A row already underlined on its
own words gets no candidates at all, because offering a guess beside a stated fact is how the two
get confused.

Nothing is inferred. An un-adjudicated term has no status and does not enter the curated export, silence is not consent, because a ground truth file assembled from terms nobody looked at is the file being
replaced.
"""

from __future__ import annotations

import dash
from dash import ALL, Input, Output, State, html
from dash.exceptions import PreventUpdate

from .. import theme
from hpo_extraction.curation import sources
from hpo_extraction.curation import store
from . import common, editor, labelling, reader

VIEW_ID = "approve"

#: What the buttons on an existing annotation record. ``approve``/``reject`` are for suggestions;
#: these are verdicts, and the distinction is kept in the log so the two can be told apart later.
VERDICTS = [("kept", "Keep"), ("removed", "Remove"), ("needs_work", "Needs work")]

#: Where in this file's own layout the editor panels live. See ``editor.ident``.
AT = "approve"

#: Said on the block and repeated on every empty state, because "why is that term not here?" is the
#: first question this screen's shape provokes.
ADJUDICATION_NOTE = (
    "One row per annotation to rule on: the confirmed set (annotations_confirmed.csv), plus the "
    "prior_annotation annotation for any term it no longer carries. prior_annotation_2 and PhenoBERT stay on the report "
    "and in the panel above as cross-checks, nothing about them enters the curated ground truth, so "
    "neither asks for a verdict."
)


def layout() -> html.Div:
    """The Dash layout of this view."""
    return html.Div([html.Div(id="approve-body")], id="approve-panel")


def register(app: dash.Dash) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("approve-body", "children"),
        Input("store-patient", "data"),
        Input("store-dirty", "data"),
        Input("store-theme", "data"),
    )
    def draw(patient_id, _dirty, mode):
        return common.guard("Approve", render, patient_id, mode or "light")

    @app.callback(
        Output("store-dirty", "data", allow_duplicate=True),
        Input({"type": "adjudicate", "key": ALL, "act": ALL}, "n_clicks"),
        State("store-patient", "data"),
        State("store-dirty", "data"),
        prevent_initial_call=True,
    )
    def adjudicate(_clicks, patient_id, dirty):
        """Approve/reject a suggestion, or pass a verdict on an existing annotation."""
        registry = common.registry_or_none()
        triggered = dash.ctx.triggered_id
        if registry is None or not triggered or not any(_clicks or []):
            raise PreventUpdate

        key, act = triggered["key"], triggered["act"]
        row = registry.state()["rows"].get(key)
        fields = _identity(registry, patient_id, key, row)

        if act in ("approve", "reject"):
            registry.record(act, patient_id, target_key=key, **fields)
        else:
            registry.record("verdict", patient_id, target_key=key, status=act, **fields)
        return (dirty or 0) + 1

    @app.callback(
        Output("store-dirty", "data", allow_duplicate=True),
        Output("store-segment", "data", allow_duplicate=True),
        Input({"type": "anchor", "key": ALL, "seg": ALL, "trigger": ALL}, "n_clicks"),
        State("store-patient", "data"),
        State("store-dirty", "data"),
        prevent_initial_call=True,
    )
    def anchor(_clicks, patient_id, dirty):
        """Record where an existing code-only term came from. Evidence, not a decision."""
        registry = common.registry_or_none()
        triggered = dash.ctx.triggered_id
        if registry is None or not triggered or not any(_clicks or []):
            raise PreventUpdate

        key, segment_idx = triggered["key"], int(triggered["seg"])
        view = registry.patient(patient_id)
        row = registry.state()["rows"].get(key)
        registry.record(
            "anchor", patient_id, target_key=key,
            segment_idx=segment_idx,
            segment_text=view["display"][segment_idx] if view else "",
            trigger_word=triggered["trigger"],
            **_identity(registry, patient_id, key, row),
        )
        return (dirty or 0) + 1, segment_idx


def _identity(registry, patient_id: str, key: str, row) -> dict:
    """The fields an event must carry so the fold can materialise a row it has never seen.

    A verdict on a file's annotation is the first time that annotation appears in the log, the
    annotation files are an input, not part of it, so the event has to name the code itself.
    Reconstructing it from the key alone would work today and break the moment a key format changes.
    """
    if row:
        return {"hpo_code": row.get("hpo_code", ""), "hpo_name": row.get("hpo_name", "")}
    parts = key.split("|")
    code = parts[2] if len(parts) > 2 else ""
    return {"hpo_code": code, "hpo_name": registry.search.label(code) if code else ""}


# ──────────────────────────────────────────────────────────────────────────────
# rendering, pure
# ──────────────────────────────────────────────────────────────────────────────
def render(patient_id, mode: str = "light"):
    """The Approve list of *patient_id*: every annotation with its accept and reject controls."""
    registry = common.registry_or_none()
    if registry is None or not patient_id:
        return theme.empty(common.NO_PATIENT)
    view = registry.patient(patient_id)
    if view is None:
        return theme.note(f"{patient_id}: not in any source.", "warn")

    rows = {row["key"]: row for row in registry.rows_for(patient_id)}
    return html.Div([
        _suggestions_block(registry, rows, mode),
        _existing_block(registry, view, rows, mode),
    ])


def _suggestions_block(registry, rows: dict, mode: str):
    pending = [r for r in rows.values()
               if r.get("source") == "new" and r.get("status") == "suggested"]
    decided = [r for r in rows.values()
               if r.get("source") == "new" and r.get("status") != "suggested"]

    if not pending and not decided:
        return theme.card("Suggestions", theme.empty("Nothing has been proposed for this patient."))

    body = [_suggestion_card(registry, row, mode)
            for row in sorted(pending, key=_suggestion_order)]
    if decided:
        body.append(html.Details([
            html.Summary(f"{len(decided)} already adjudicated", className="details-summary"),
            html.Div([_suggestion_card(registry, row, mode, closed=True)
                      for row in sorted(decided, key=_suggestion_order)]),
        ]))
    return theme.card(f"Suggestions ({len(pending)} open)", html.Div(body))


def _suggestion_order(row: dict) -> tuple:
    """Segment order, like everything else in this column. A suggestion always names its segment."""
    return _int(row.get("segment_idx")), row.get("hpo_code", "")


def _suggestion_card(registry, row: dict, mode: str, closed: bool = False):
    colors = theme.status_colors(mode)
    status = row.get("status") or "suggested"
    actions = [] if closed else [
        html.Button("Approve", id={"type": "adjudicate", "key": row["key"], "act": "approve"},
                    className="primary-btn", n_clicks=0),
        html.Button("Reject", id={"type": "adjudicate", "key": row["key"], "act": "reject"},
                    className="copy-btn", n_clicks=0),
        html.Button("Needs work", id={"type": "adjudicate", "key": row["key"],
                                      "act": "needs_work"},
                    className="copy-btn", n_clicks=0),
    ]
    return html.Div([
        html.Div([
            common.chip(status, colors.get(status, colors["suggested"]),
                        title=theme.STATUS_HELP.get(status, "")),
            common.code_chip(row.get("hpo_code", ""),
                             row.get("hpo_name") or row.get("hpo_code", ""),
                             theme.palette(mode)["text"],
                             title=registry.search.describe(row.get("hpo_code", ""))),
        ], className="row-head"),
        html.Div(common.segment_line(_int(row.get("segment_idx")),
                                     str(row.get("segment_text", ""))), className="stat-sub"),
        html.Div(f"trigger: “{row.get('trigger_word', '')}”", className="stat-sub"),
        html.Div(row["note"], className="stat-sub") if row.get("note") else None,
        html.Div(f"{row.get('author', '')} · {row.get('updated_at', '')}", className="stat-sub"),
        html.Div(actions, className="row-actions") if actions else None,
        editor.panel(registry, row["key"], AT, segment_idx=_none_int(row.get("segment_idx")),
                     trigger_word=row.get("trigger_word", ""),
                     hpo_code=row.get("hpo_code", ""), note=row.get("note", ""),
                     labels=row.get("labels"), mode=mode),
        labelling.qualifier_chips(row.get("labels"), mode),
    ], className="row-card")


def _existing_block(registry, view: dict, rows: dict, mode: str):
    entries = _existing_entries(registry, view, rows)
    if not entries:
        return theme.card("Existing annotations",
                          theme.empty("No annotation to rule on for this patient."),
                          subtitle=ADJUDICATION_NOTE)
    groups = group_entries(entries)
    n_open = sum(1 for e in entries if not e["status"])
    n_proposed = sum(1 for e in entries if e["status"] == store.DELETE_SUGGESTED)
    heading = f"Existing annotations ({len(entries)} on {len(groups)} terms, {n_open} undecided"
    heading += f", {n_proposed} deletion proposed)" if n_proposed else ")"
    return theme.card(
        heading,
        html.Div([_existing_group_card(registry, group, mode) for group in groups]),
        subtitle=ADJUDICATION_NOTE + " An undecided row does not enter the curated export: "
                                     "silence is not consent. Ordered by segment, so this column "
                                     "reads in the same order as the report beside it.",
    )


def _existing_entries(registry, view: dict, rows: dict) -> list[dict]:
    """One entry per annotation **to rule on**, with its verdict, its evidence and its repairs.

    The set comes from ``view["existing"]``, per code, the best source in
    ``sources.ADJUDICATION_ORDER`` that has one, not from every record the inputs carry. The rest
    stay on the report and in the document panel as reference, where they are worth having and cost
    nobody a decision.

    Ordered undecided-first: the point of the screen is the work that remains, and a patient
    half-done should open on the half that is not.
    """
    placed = reader.placed_by_key(view)
    pb_codes = {d.get("hpo_id") for group in view["detections"].values() for d in group}
    pb_codes |= {d.get("hpo_id") for d in view["unplaced"]}

    # Computed over *every* source, not only the adjudicated ones. Knowing that the second
    # annotator also carries this code is what makes the verdict easy, and it is the one
    # thing the reference sources are still here to say.
    sources_by_code: dict[str, set[str]] = {}
    for record in view["records"]:
        sources_by_code.setdefault(record["hpo_code"], set()).add(record["source"])

    # The sources carrying something *for this patient*, not every file that loaded. A file with
    # no row for this report is not a file that disagrees with the ones that have one, see
    # ``Registry.gold_sources_for``.
    loaded = registry.gold_sources_for(view["patient_id"])

    entries: list[dict] = []
    for record in view["existing"]:
        key = record["key"]
        row = rows.get(key, {})
        placement = placed.get(key)
        code = record["hpo_code"]
        label, why = common.agreement(sources_by_code.get(code, set()), loaded, code in pb_codes)

        # What the app knows now, which is the file's claim unless somebody has edited or located
        # it. The record's own values are the fallback, never the override.
        segment_idx = row.get("segment_idx")
        if segment_idx is None and placement is not None:
            segment_idx = placement["segment_idx"]
        trigger = row.get("trigger_word") or record.get("trigger_word", "")
        effective = row.get("hpo_code") or code

        detail = reader.evidence_line(record, placement)
        if record.get("provenance"):
            detail += f" · provenance: {record['provenance']}"
        if record.get("confirmed") is False:
            detail += " · not confirmed in its file"

        entries.append({
            "key": key, "source": record["source"], "code": code, "effective_code": effective,
            "name": row.get("hpo_name") or record.get("hpo_name")
                    or registry.search.label(effective),
            "known": registry.search.known(effective),
            "status": row.get("status", ""),
            "segment_idx": segment_idx,
            "trigger_word": trigger,
            "note": row.get("note", ""),
            "agreement": label, "agreement_why": why,
            # Candidates exist only for a code nothing could be drawn for, see ``Registry``.
            "candidates": view["triggers"].get(code, []),
            "labels": row.get("labels") or [],
            "difficulty": row.get("difficulty", ""),
            "delete_reason": row.get("delete_reason", ""),
            "delete_by": row.get("delete_by", ""),
            "edited": editor.edited_fields(row, record),
            "stands_in": _stands_in(record, sources_by_code.get(code, set())),
            "carried": "",
            "detail": detail,
        })

    entries.extend(_carried_over(registry, view, rows, sources_by_code, pb_codes, loaded))

    entries.sort(key=order_key)
    return entries


#: Sorts last: a row whose segment nothing could establish. Larger than any real segment index and
#: not ``inf``, so the key stays sortable against plain ints.
NO_SEGMENT = 10 ** 6


def band(status: str) -> int:
    """Deletion-proposed, then undecided, then decided. The work that remains, first.

    A proposed deletion is somebody waiting on an answer, so it outranks even an undecided row.
    """
    if status == store.DELETE_SUGGESTED:
        return 0
    return 1 if not status else 2


def order_key(entry: dict) -> tuple:
    """``(band, segment, code)``, the right column read in the order the report reads.

    Sorting by source then code, as this used to, meant the panel and the document beside it were
    two different orders: checking a row against its sentence was a hunt, and reading the report
    top to bottom while working down the panel was impossible. Now the two agree, and a row's
    position is a hint about where its evidence is.

    The bands survive that change because they answer a different question. Segment order alone
    would bury an open deletion proposal, somebody's blocked question, three quarters of the way
    down a long report, where nothing marks it as the thing waiting on you.
    """
    segment_idx = entry.get("segment_idx")
    try:
        segment = int(segment_idx)
    except (TypeError, ValueError):
        segment = NO_SEGMENT
    return band(entry.get("status", "")), segment, entry.get("code", "")


def group_entries(entries: list[dict]) -> list[dict]:
    """Fold per-annotation entries into one group per HPO term, ordered like the report.

    A term annotated twice is two rows in the data model and one phenotype in the curator's head, and the two rows differ only in *which words earned it*, which is what a single card
    with two evidence blocks shows better than two cards with duplicate headings.

    **The verdict stays per evidence.** A group is presentation over a faithful log: each member
    keeps its own key, its own buttons and its own editor, so keeping one occurrence never silently
    settles the other. That is the same rule ``daphne`` carrying one code twice already relies on.

    A group takes over the **minimum** band and the **minimum** segment of its members, so a term
    with one open question sorts with the open work, not with the settled rows around it.
    """
    groups: dict[str, dict] = {}
    for entry in entries:
        code = entry.get("effective_code") or entry.get("code", "")
        group = groups.get(code)
        if group is None:
            groups[code] = {"code": code, "name": entry.get("name") or code,
                            "members": [entry]}
        else:
            group["members"].append(entry)

    out = []
    for group in groups.values():
        group["members"].sort(key=order_key)
        first = group["members"][0]
        group["order"] = (min(band(m.get("status", "")) for m in group["members"]),
                          order_key(first)[1], group["code"])
        out.append(group)
    out.sort(key=lambda g: g["order"])
    return out


def _stands_in(record: dict, present: set[str]) -> str:
    """Why a lower-precedence source is being ruled on, when a better one exists in the cohort.

    A prior_annotation row is on this screen when ``daphne`` dropped that term, and the drop is
    itself a judgement somebody made. Saying so on the row turns "why am I looking at this one?"
    into the actual question: *was dropping it right?*
    """
    better = [source for source in sources.ADJUDICATION_ORDER
              if sources.precedence(source) < sources.precedence(record["source"])]
    if not better or any(source in present for source in better):
        return ""
    return (f"{', '.join(better)} carries no annotation for this term, this "
            f"{record['source']} row stands in its place")


def _carried_over(registry, view: dict, rows: dict, sources_by_code, pb_codes, loaded) -> list[dict]:
    """Rows the log has already decided that no current input nominates.

    Three ways to get one: a source was repointed or retired (every ``raw|…`` verdict from before
    ``prior_annotation`` replaced it), a PhenoBERT detection somebody kept back when detections were
    adjudicable, or a term ``daphne`` gained after a prior_annotation row for it had been ruled on.

    They are shown, and not quietly dropped, because a decided row that vanishes from the screen is
    a decision nobody can revisit, and one that carries an open deletion proposal would be a
    question nobody can answer.
    """
    entries = []
    for key, row in rows.items():
        if key in view["existing_keys"] or row.get("source") == "new" or not row.get("status"):
            continue
        code = row.get("hpo_code", "")
        label, why = common.agreement(sources_by_code.get(code, set()), loaded, code in pb_codes)
        entries.append({
            "key": key, "source": row.get("source", ""), "code": code, "effective_code": code,
            "name": row.get("hpo_name") or registry.search.label(code),
            "known": registry.search.known(code),
            "status": row.get("status", ""),
            "segment_idx": row.get("segment_idx"),
            "trigger_word": row.get("trigger_word", ""),
            "note": row.get("note", ""),
            "agreement": label, "agreement_why": why,
            "candidates": [],
            "labels": row.get("labels") or [],
            "difficulty": row.get("difficulty", ""),
            "delete_reason": row.get("delete_reason", ""),
            "delete_by": row.get("delete_by", ""),
            "edited": [],
            "stands_in": "",
            "carried": "no current source nominates this row, it is here because somebody has "
                       "already ruled on it, and a decided row that vanished would be a decision "
                       "nobody could revisit",
            "detail": "",
        })
    return entries


def _existing_group_card(registry, group: dict, mode: str):
    """One card per HPO term, one block per annotation of it.

    The head is the term and everything true of it at the term level, which files carry it, what
    the ontology calls it. Each member below carries what is true of *that* annotation: where it
    sits, what its file said, its own verdict buttons, its own editor, its own qualifiers.
    """
    palette = theme.palette(mode)
    members = group["members"]

    sources_present = list(dict.fromkeys(m["source"] for m in members))
    head = [common.chip(source, theme.source_color(source, palette),
                        title=theme.SOURCE_HELP.get(source, ""))
            for source in sources_present]
    head.append(common.chip(members[0]["agreement"], palette["neutral"],
                            title=members[0]["agreement_why"]))
    head.append(common.code_chip(group["code"], group["name"], palette["text"],
                                 title=registry.search.describe(group["code"])))
    if not members[0]["known"]:
        head.append(common.chip("unknown code", palette["warning"],
                                title="not a phenotypic-abnormality term in this hpo.json, it "
                                      "can be curated, but nothing can score against it"))

    body = [html.Div(head, className="row-head")]
    if len(members) > 1:
        body.append(html.Div(
            f"{len(members)} annotations of this term, each is judged on its own evidence",
            className="stat-sub"))
    body.extend(_member_block(registry, entry, mode, sole=len(members) == 1)
                for entry in members)
    return html.Div(body, className="row-card")


def _member_block(registry, entry: dict, mode: str, sole: bool):
    """One annotation inside a term's card: its evidence, its verdict, its repairs."""
    palette = theme.palette(mode)
    colors = theme.status_colors(mode)
    status = entry["status"]

    line = []
    if status:
        line.append(common.chip(status, colors.get(status, palette["neutral"]),
                                title=theme.STATUS_HELP.get(status, "")))
    if not sole:
        # With one annotation the card head already names the source. With several it is the thing
        # that tells two otherwise identical blocks apart.
        line.append(common.chip(entry["source"], theme.source_color(entry["source"], palette),
                                title=theme.SOURCE_HELP.get(entry["source"], "")))
    if entry["edited"]:
        line.append(common.chip("edited: " + ", ".join(entry["edited"]), palette["warning"],
                                title="this app changed it from what its file said, the row keeps "
                                      "its original address, and the log has both"))

    body = [html.Div(line, className="row-head")] if line else []
    if entry.get("stands_in"):
        body.append(html.Div(entry["stands_in"], className="stat-sub"))
    if entry.get("carried"):
        body.append(html.Div(entry["carried"], className="stat-sub"))
    if status == store.DELETE_SUGGESTED:
        body.append(html.Div([
            html.Span("Deletion proposed", className="banner-title"),
            html.Div(f"“{entry['delete_reason'] or 'no reason given'}”",
                     className="banner-text"),
            html.Div(f", {entry['delete_by']} · Remove accepts, Keep rejects",
                     className="stat-sub"),
        ], className="delete-banner"))
    if entry["effective_code"] != entry["code"]:
        body.append(html.Div(f"its file says {entry['code']}, curated as "
                             f"{entry['effective_code']}", className="stat-sub"))
    if entry["detail"]:
        body.append(html.Div(entry["detail"], className="stat-sub"))
    if entry.get("segment_idx") is not None and entry.get("trigger_word"):
        body.append(html.Div(f"#{_int(entry['segment_idx'])} · trigger: "
                             f"“{entry['trigger_word']}”", className="stat-sub"))
    elif entry["candidates"]:
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
    else:
        body.append(html.Div("no lexical match in this report, set the segment and trigger by "
                             "hand below", className="stat-sub"))
    if entry["note"]:
        body.append(html.Div(entry["note"], className="stat-sub"))

    body.append(html.Div([
        html.Button(text, id={"type": "adjudicate", "key": entry["key"], "act": verdict},
                    className="primary-btn" if verdict == "kept" else "copy-btn", n_clicks=0)
        for verdict, text in VERDICTS
    ], className="row-actions"))
    body.append(editor.panel(registry, entry["key"], AT,
                             segment_idx=_none_int(entry.get("segment_idx")),
                             trigger_word=entry.get("trigger_word", ""),
                             hpo_code=entry["effective_code"], note=entry.get("note", ""),
                             labels=entry.get("labels"), mode=mode))
    body.append(labelling.qualifier_chips(entry.get("labels"), mode))
    return html.Div(body, className="row-member" + ("" if sole else " row-member-split"))


def _int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _none_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
