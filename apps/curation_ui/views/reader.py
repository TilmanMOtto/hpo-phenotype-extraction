"""The report, segment by segment, with every piece of evidence drawn in place.

This is the column the curator actually reads, so the rule it follows is that a mark on the page
must mean one thing and must be checkable:

* an **annotation** from one of the rich sources is underlined in that source's colour, on the
  words its file gave as the trigger. This is a claim somebody made, not a guess, and it is drawn
  because the source said so, ``hcy_holistic_ground_truth.csv`` ships a trigger word and a
  character offset, ``annotations_confirmed.csv`` a trigger word and a segment index. Two files
  agreeing on the same code *and* the same words are one mark that names both: marks cannot nest,
  and a mark that named only the first would say one file asserted what two did;
* a **PhenoBERT span** is underlined where PhenoBERT found it, tagged with the HPO term it
  normalised to. The underline says *where*. The tag says *what it became*, and the tag is the
  half that decides whether the extraction was right. A negated detection is struck through, which
  is the case a term list cannot express at all;
* a **trigger candidate** (:mod:`locate`) is boxed faintly. It is not a finding, it is a guess
  about where a bare ``prior_annotation_2`` code came from, so it must not look like one;
* a **curated trigger**, one this app already recorded, is boxed solidly.

The candidates are now the exception: they are computed only for codes no source could point at,
so a guess never sits beside the evidence it is guessing about.

The offsets come from ``baseline_phenobert`` and index the verbatim staged report, translated onto segments by
:mod:`spans`. Where that translation failed, the segment is drawn plain and the header says how many
segments are in that state. Approximately-right underlining in a tool whose output becomes ground
truth is worse than none.

Above the report sits the **document annotations** panel: every term any annotation file carries
for this patient, listed rather than counted, grouped by code and tagged with the sources that
carry it, including the ones Approve mode will not ask about, which are marked ``reference``. For a
rich source it also says *how* the mark was placed, the file's own segment index,
its character offset, its sentence context, or a bare lexical scan, because "the file said so" and
"the trigger happens to occur here" are different claims and a curator adjudicating the annotation
is entitled to know which they are looking at. Reading the report against that list is the actual
task, and a header that said "3 annotated terms" made it impossible.

A line there is drawn **light red** when its annotation has a trigger word and nothing underlined
in the report, the file named words the report does not contain, or a sentence the trigger did not
survive into. Both need a segment and a trigger set by hand, and both used to read as ordinary
text, which made the by-hand queue invisible in the one panel that lists it.

The comparison behind ``disagree`` is against the sources carrying something **for this patient**,
not against every file that loaded: a file with no row for this report is not a file that disagrees
with the ones that have one.

Every phenotype on this screen hovers to the same thing: its name, its id and the ontology's
definition (``search.describe``). Deciding whether ``HP:0001252`` belongs on a sentence *is*
deciding what the ontology means by hypotonia, and sending the reader to another tab to find out is
how a verdict gets guessed.

**The report reads as a document, not as a list of sentences.** The segmentation is a pipeline
artefact. A clinician's report is prose, and chopping it into numbered rows makes it harder to read
in the way a curator cannot afford, a phenotype spread over two sentences, or a finding
qualified by the sentence before it, disappears when every sentence is boxed on its own line. So the
text between segments is put back from the verbatim report: the headings, the line breaks, the blank
lines. What is drawn is the report as it was written.

The segmentation is still there, as an *inline* span per segment. Hovering one lights up its extent,
so the boundaries the pipeline works with stay visible on demand. Clicking one selects it as the
target for the next suggestion. Reports whose segments could not be aligned fall back to joining
them with spaces, the prose is then approximate, and the header already says so.

Clicking a mark selects its segment, fills the trigger word **and** picks its HPO term, which is the
one-click path from "PhenoBERT saw something here" to a suggestion needing only a glance to confirm.
"""

from __future__ import annotations

import dash
from dash import ALL, Input, Output, html
from dash.exceptions import PreventUpdate

from hpo_extraction.curation import evidence_location as locations
from hpo_extraction.curation import sources as sources_mod
from .. import theme
from . import common

VIEW_ID = "reader"

#: How many segments are made individually selectable. Clinical reports run to a few hundred
#: sentences at most, so this is a guard against a pathological input, not a routine path.
#:
#: It caps the *spans*, not the text: whatever follows the last wrapped segment is still emitted as
#: prose by :func:`_prose`, so the report is never silently truncated, a curator reading a report
#: that quietly stopped halfway would have no way to know.
PAGE_SIZE = 400

#: What each placement strategy means, in the words a curator needs at the moment of judging the
#: mark. Drawn from :mod:`hpo_extraction.curation.evidence_location`. Kept here because it is hover text, not
#: logic, and it has to read as a sentence, not as an enum.
HOW_HELP = {
    "segment": "placed where its file said, the file's own segment index",
    "offset": "placed from its file's character offset into its own copy of the report",
    "context": "placed by matching its file's sentence context to a segment",
    "lexical": "placed by finding the trigger word in the report, the file gave no position, so "
               "this is the app's own scan, not the annotator's claim",
}

#: The sources whose annotations can be located by hand, the union the curated ground truth is built from,
#: ``holistic ∪ suggestions ∪ daphne``. ``marc2`` is not one: it is a code list, so it
#: never claimed a position at all, and padding a queue whose meaning is *somebody said where this
#: was and we could not find it* with every one of its terms would empty the queue of meaning.
ANCHORABLE = frozenset({"prior_annotation", "daphne", "new"})


def layout() -> html.Div:
    """The Dash layout of this view."""
    return html.Div(id="reader-body", className="reader")


def register(app: dash.Dash) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("store-segment", "data", allow_duplicate=True),
        Output("edit-trigger", "value", allow_duplicate=True),
        Output("edit-hpo", "options", allow_duplicate=True),
        Output("edit-hpo", "value", allow_duplicate=True),
        Output("edit-feedback", "children", allow_duplicate=True),
        Input({"type": "mark", "idx": ALL, "trigger": ALL, "code": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def pick_mark(_clicks):
        """A click on a mark fills the whole suggestion: segment, trigger word and term.

        The term has to arrive with its option, not just its value, a ``dcc.Dropdown`` renders a
        selection it has no option for as blank, so setting the value alone would look like nothing
        happened.

        A code the ontology cannot resolve is *not* pre-filled. PhenoBERT ships an
        older HPO release, and pre-selecting a dead code would set the curator up to save an
        annotation nothing can ever score against. The trigger is still filled and the panel says
        why the term was left empty.
        """
        registry = common.registry_or_none()
        triggered = dash.ctx.triggered_id
        if registry is None or not triggered or not any(_clicks or []):
            raise PreventUpdate

        code = triggered.get("code") or ""
        if code and registry.search.known(code):
            return (triggered["idx"], triggered["trigger"],
                    [registry.search.option(code)], code, None)
        if code:
            return (triggered["idx"], triggered["trigger"], dash.no_update, None,
                    theme.note(f"{code} is not a phenotypic-abnormality term in this hpo.json, so "
                               "it was not pre-filled, pick the term this evidence supports.",
                               "warn"))
        return triggered["idx"], triggered["trigger"], dash.no_update, dash.no_update, None

    @app.callback(
        Output("store-segment", "data", allow_duplicate=True),
        Input({"type": "seg", "idx": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def pick_segment(_clicks):
        triggered = dash.ctx.triggered_id
        if not triggered or not any(_clicks or []):
            raise PreventUpdate
        return triggered["idx"]

    @app.callback(
        Output("reader-body", "children"),
        Input("store-patient", "data"),
        Input("store-segment", "data"),
        Input("store-dirty", "data"),
        Input("store-theme", "data"),
        Input("tabs", "value"),
    )
    def draw(patient_id, segment_idx, _dirty, mode, tab):
        """One report column, shared by Curate and Evidence location. The tab decides what sits above it.

        Not gated on the tab the way Overview's cohort work is: this is the panel being switched
        *to*, so skipping the render would show the previous tab's version of it until something
        else happened to redraw. Only the panel above the prose changes, the report itself draws
        every annotation on both tabs.
        """
        return common.guard("Report", render, patient_id, segment_idx, mode or "light",
                            tab == "anchor")


# ──────────────────────────────────────────────────────────────────────────────
# rendering, pure, so selftest can drive it without a browser
# ──────────────────────────────────────────────────────────────────────────────
def render(patient_id, segment_idx, mode: str = "light", only_unanchored: bool = False):
    """The report text of *patient_id* by segment, highlighting *segment_idx*. With *only_unanchored* only segments with unplaced annotations."""
    registry = common.registry_or_none()
    if registry is None or not patient_id:
        return theme.empty(common.NO_PATIENT)
    view = registry.patient(patient_id)
    if view is None:
        return theme.note(f"{patient_id}: not in any source.", "warn")
    if not view["segments"]:
        return theme.note(f"{patient_id}: no segmented report, nothing to read.", "warn")

    curated = _curated_marks(registry.rows_for(patient_id))
    palette = theme.palette(mode)
    body = _prose(registry, view, curated, palette, segment_idx)

    # Outside the prose div, never inside it: a note is a block element, and one of those in the
    # middle of the text breaks the line and stops the report flowing.
    overflow = None
    if len(view["display"]) > PAGE_SIZE:
        overflow = theme.note(
            f"This report has {len(view['display'])} segments. The text is complete, but only the "
            f"first {PAGE_SIZE} are individually selectable; to annotate beyond that, raise "
            "PAGE_SIZE.", "warn")

    return html.Div([
        _header(registry, view, palette),
        render_gold_panel(registry, view, mode, only_unanchored),
        theme.card(
            "Report",
            html.Div([html.Div(body, className="prose"), overflow]),
            subtitle="Hover a sentence to see the segment the pipeline works with; click to make "
                     "it the target for a suggestion. Click a mark to fill the trigger and the "
                     "term as well.",
        ),
    ])


def _prose(registry, view: dict, curated: dict, palette: dict, selected_idx) -> list:
    """The report as continuous text, each segment an inline, selectable span.

    The text *between* segments comes from the verbatim report, recovered through the alignment: a
    segment ends at character 46 and the next begins at 48, so those two characters, a newline, a
    space, a heading's colon, go back where they were. That is the whole difference between a
    document and a list of sentences.

    Two things this must never do. It must not reorder: a segment whose range starts behind the
    cursor (which alignment's one backward retry can produce) is emitted in *segment* order with a
    plain space, because reading order is what a curator is checking against. And it must not drop
    text: whatever follows the last segment is emitted too, so a trailing signature block is
    visible, not silently absent.
    """
    text = view["text"]
    ranges = view["ranges"] or []
    out: list = []
    cursor = 0

    for idx, segment_text in enumerate(view["display"][:PAGE_SIZE]):
        span = ranges[idx] if idx < len(ranges) else None
        if text and span is not None and span[0] >= cursor:
            if span[0] > cursor:
                out.append(text[cursor:span[0]])
            cursor = span[1]
        elif out:
            # No verbatim gap to restore, an unaligned segment, or one out of document order.
            out.append(" ")
        out.append(_segment(registry, idx, segment_text, view, curated, palette,
                            selected=(idx == selected_idx)))

    if text and cursor < len(text):
        out.append(text[cursor:])
    return out


def _header(registry, view: dict, palette: dict):
    align = view["alignment"]
    n_placed = sum(len(rows) for rows in view["annotations"].values())
    n_unplaced = len(view["unplaced_annotations"])
    per_source = " · ".join(f"{source} {len(view['gold'].get(source, []))}"
                            for source in registry.gold_sources)
    stats = [
        theme.stat("Segments", str(align["n_segments"])),
        theme.stat("PhenoBERT", str(view["n_detections"]),
                   sub=f"{len(view['unplaced'])} unplaced" if view["unplaced"] else "placed"),
        theme.stat("Annotations", str(n_placed + n_unplaced),
                   sub=f"{n_unplaced} not placed" if n_unplaced else "all placed",
                   help_text="rows the annotation files carry for this report; 'not placed' means "
                             "the trigger word occurs nowhere in it, so nothing is underlined"),
        theme.stat("Terms", str(len(view["gold_codes"])), sub=per_source or "no source loaded"),
    ]
    notes = []
    if align["n_unaligned"]:
        notes.append(theme.note(
            f"{align['n_unaligned']} of {align['n_segments']} segments could not be aligned to the "
            "verbatim report; those are drawn without PhenoBERT marks rather than with guessed "
            "ones.", "warn"))
    if not view["text"]:
        notes.append(theme.note(
            "No verbatim report for this patient, neither a staged PhenoBERT copy nor a "
            "report_text in the prior_annotation file, so no offset can be placed. The segmentation is "
            "still shown.", "warn"))
    elif view.get("text_source") == "prior_annotation":
        notes.append(theme.note(
            "The prose is the prior_annotation file's own report_text; the PhenoBERT baseline never staged this report, "
            "so there are no PhenoBERT marks to draw.", "warn"))
    return html.Div([theme.stat_row(stats), *notes], style={"marginBottom": "10px"})


def placed_by_key(view: dict) -> dict[str, dict]:
    """``{annotation key: placed record}``, the placement, looked up the way rows are addressed.

    :func:`anchors.place_all` returns annotations grouped by segment, which is what the reader
    weaves. Every other panel asks "where did *this* row end up", so the inversion lives here, not being redone in each of them.
    """
    return {record["key"]: record
            for rows in view["annotations"].values() for record in rows}


def evidence_line(record: dict, placement: dict | None) -> str:
    """One line saying where an annotation sits, and on whose authority.

    ``#3 “convulsions” · placed from its file's character offset``, the segment, the words, and
    which of :mod:`anchors`' strategies got it there. The last part is not decoration: a mark placed
    because the file named the sentence and a mark placed because the app scanned for the word are
    different claims, and only one of them is the annotator's.
    """
    trigger = (record.get("trigger_word") or "").strip()
    if placement is None:
        if trigger:
            return f"“{trigger}” occurs nowhere in this report, locate it by hand"
        return "no trigger word in its file, nothing to place"
    where = f"#{placement['segment_idx']}"
    if placement.get("start") is None:
        return f"{where} · “{trigger}” is not in that segment · {HOW_HELP.get(placement['how'], '')}"
    return f"{where} “{trigger}” · {HOW_HELP.get(placement.get('how', ''), '')}"


def render_gold_panel(registry, view: dict, mode: str = "light", only_unanchored: bool = False):
    """Every term the annotation files carry for this patient, listed, not counted.

    Grouped by code, not by file. A term two sources carry is two rows in the data model and one
    annotation in the curator's head, and the question being answered, *is HP:0001250 right for
    this report?*, is asked once. The sources that carry it are chips on that one row, and the
    ones that do not are what ``disagree`` means.

    Each underlying row still gets its own line, because that is where the rows differ: two files
    can agree on a code and disagree entirely about which words earned it, and a panel that showed
    only the code would hide the more interesting of the two disagreements.

    Lines Approve mode will not ask about are marked ``reference``. This panel shows
    **more** than Approve does, a second annotator's agreement is worth reading and is not worth a
    verdict, so the difference has to be visible here, or it reads as something missing.

    With *only_unanchored* it becomes the Evidence location tab's panel instead: the same list, cut to the rows
    that still need a segment and a trigger word. That is the whole difference between the two
    screens. Reading a report to adjudicate means seeing every term it carries. Reading it to place
    the last few means seeing only those, because a queue you have to find inside a full list is a
    queue nobody finishes. The report beside it is **not** filtered either way, every annotation
    stays drawn on the words it came from, which is what the remaining ones are being placed against.
    """
    codes = view["gold_codes"]
    if not codes:
        return theme.card("Document annotations",
                          theme.empty("No annotation file carries a term for this patient."),
                          subtitle="; ".join(theme.SOURCE_HELP[s] for s in registry.gold_sources)
                                   or "no annotation source loaded.")

    palette = theme.palette(mode)
    placed = placed_by_key(view)
    # The log's version of each annotation, so a row this app has already located leaves the queue.
    # See ``needs_anchor``: ``anchors.place_all`` reads the files and never sees the log.
    curated = {row["key"]: row for row in registry.rows_for(view["patient_id"])}
    display = view["display"]
    by_code: dict[str, list[dict]] = {}
    for record in view["records"]:
        by_code.setdefault(record["hpo_code"], []).append(record)
    # ``holistic`` and ``daphne`` overlap by design, so a term both carry is one annotation
    # with two rows behind it. Listing both asked the curator to read the same claim twice, and on
    # The Evidence location tab, to place it twice. Rows that genuinely disagree about which words earned the
    # term are still two rows. That is the disagreement worth seeing. See ``sources.merge_identical``.
    spans = realised_spans(view, curated)
    by_code = {code: sources_mod.merge_identical(members, spans)
               for code, members in by_code.items()}

    # The sources carrying something *for this patient*. A file with no row for this report is not
    # a file that disagrees with the ones that have one, see ``Registry.gold_sources_for``.
    loaded = registry.gold_sources_for(view["patient_id"])
    rows = []
    n_disagree = 0
    n_unplaced = 0
    n_shown = 0
    for code in sorted(codes, key=lambda c: (len({r["source"] for r in by_code.get(c, [])})
                                             == len(loaded), c)):
        members = by_code.get(code, [])
        # Off ``sources``, never ``source``: a merged row's scalar names only its primary, so
        # reading that would drop ``holistic`` from every term ``daphne`` also carries, and the
        # agreement chip, which is the whole point of this row, would read "disagree" on all of them.
        present = {source for record in members
                   for source in (record.get("sources") or [record["source"]])}
        agreed = len(loaded) > 1 and present >= set(loaded)
        n_disagree += 0 if agreed or len(loaded) < 2 else 1

        flags = {record["key"]: needs_location(record, merged_placement(record, placed),
                                             merged_row(record, curated, display), display)
                 for record in members}
        n_unplaced += sum(1 for record in members if flags[record["key"]])
        if only_unanchored:
            members = [record for record in members if flags[record["key"]]]
            if not members:
                continue
        n_shown += 1

        tags = [common.chip(source, theme.source_color(source, palette),
                            title=theme.SOURCE_HELP.get(source, ""))
                for source in loaded if source in present]
        if len(loaded) > 1 and not agreed:
            tags.append(common.chip("disagree", palette["serious"],
                                    title="not every loaded annotation file carries this term"))

        lines = []
        n_needs_location = 0
        for record in members:
            confirmed = record.get("confirmed")
            # Which rows Approve will actually ask about. Saying it here is what stops "why is that
            # term not in the list?" being a question the curator has to answer by reading code.
            rules_on = bool(set(record.get("keys") or [record["key"]]) & view["existing_keys"])
            flagged = flags[record["key"]]
            n_needs_location += 1 if flagged else 0
            lines.append(html.Div([
                html.Span(" + ".join(record.get("sources") or [record["source"]]) + ": ",
                          style={"fontWeight": "600"}),
                html.Span(evidence_line(record, merged_placement(record, placed))),
                html.Span(" · unconfirmed", style={"opacity": ".8"}) if confirmed is False else None,
                None if rules_on else html.Span(
                    " · reference", style={"opacity": ".7"},
                    title="not a row Approve mode rules on, either a better source carries this "
                          "term, or this file is a cross-check that cannot enter the curated ground truth"),
            ], className="stat-sub" + (" gold-line-unplaced" if flagged else ""),
                title="nothing is underlined for this annotation, set its segment and trigger by "
                      "hand in the Locate evidence tab" if flagged else None))
            for claim in record.get("claims") or []:
                # What the other file said, where it said something different. When a term is hard
                # to place, each file's guess about where it came from is a clue, so the fold
                # names them, not keeping only the one that won.
                lines.append(html.Div(f"{claim['source']} also recorded {claim_line(claim)}",
                                      className="stat-sub"))
        if not members or all(not (r.get("trigger_word") or "").strip() for r in members):
            # A code-only source said nothing about where the term came from, so this is the one
            # place a lexical guess still earns its keep, and it is labelled as a guess.
            hits = view["triggers"].get(code, [])
            where = (f"#{hits[0]['segment_idx']} “{hits[0]['trigger_word']}”"
                     if hits else "no lexical match in this report")
            lines.append(html.Div(f"possible origin: {where}", className="stat-sub"))

        rows.append(html.Div([
            html.Div(tags + [common.code_chip(code, registry.search.label(code), palette["text"],
                                              title=registry.search.describe(code))],
                     className="row-head"),
            *lines,
        ], className="gold-row" + (" gold-row-unplaced"
                                   if members and n_needs_location == len(members) else "")))

    if only_unanchored:
        # Said out loud, not drawn as an empty box: on this tab a short list and a broken
        # filter look identical, and the report you have just finished placing leaves the list,
        # which is easier to read as a bug than as progress.
        if not rows:
            return theme.card(
                "Needs evidence (0)",
                theme.empty(f"Every one of the {len(codes)} terms in this report has its evidence located."),
                subtitle="Nothing left to place here. Prev/Next, or the worklist below, for the "
                         "next report that still has something.")
        return theme.card(
            f"Needs evidence ({n_unplaced})",
            html.Div(rows, className="gold-list"),
            subtitle=f"{n_unplaced} annotation(s) over {n_shown} of this report's {len(codes)} "
                     "terms have nothing underlined in the report: the file named words it does "
                     "not contain, a sentence the trigger did not survive into, or no trigger at "
                     "all. Set the segment and the trigger below; the report beside this shows "
                     "every annotation, located or not.")

    per_source = " · ".join(f"{source} {len(view['gold'].get(source, []))}" for source in loaded)
    return theme.card(
        f"Document annotations ({len(codes)})",
        html.Div(rows, className="gold-list"),
        subtitle=(per_source
                  + (f" · {n_disagree} the files disagree on" if n_disagree else " · agreed")
                  + (f" · {n_unplaced} need evidence" if n_unplaced else "")
                  + ". Hover a term for its definition, a source chip for what that file is. "
                    "Red lines have nothing underlined in the report. The Locate evidence tab is "
                    "where they get placed. Verdicts are passed in Approve mode."),
    )


def realised_spans(view: dict, curated: dict) -> dict:
    """``{key: (segment, start, end)}`` for every annotation that actually sits on words.

    Two ways a row gets one, and both have to count. The file's own claim may have landed
    (:func:`anchors.place_all`), or a curator may have located it here since, and a row located
    by hand is *more* settled than one the loader happened to place, so leaving the log out would
    keep re-asking about the row somebody just finished.

    A row that sits nowhere is simply absent. That is the distinction the merge turns on: a claim
    that did not land says nothing about which words, so it cannot disagree with anything.
    """
    display = view["display"]
    spans: dict[str, tuple] = {}
    for rows in view["annotations"].values():
        for record in rows:
            if record.get("start") is not None:
                spans[record["key"]] = (record["segment_idx"], record["start"], record["end"])
    for key, row in (curated or {}).items():
        hit = _curated_location(row, display)
        if hit is not None:
            spans[key] = hit
    return spans


def claim_line(claim: dict) -> str:
    """``“purpura” #4``, one folded row's own account of where the term came from."""
    trigger = (claim.get("trigger_word") or "").strip()
    where = "" if claim.get("segment_idx") is None else f" #{claim['segment_idx']}"
    return (f"“{trigger}”{where}" if trigger else (f"segment{where}" if where else "nothing"))


def merged_placement(record: dict, placed: dict):
    """Where a merged annotation ended up: the placement that **landed**, if any of them did.

    A merged row stands for several of the files' rows, and they were placed independently, one
    file naming a segment the trigger survives into and another naming nothing is the case
    the merge exists for. The annotation has evidence if any of its rows found some.
    """
    hits = [placed.get(key) for key in record.get("keys") or [record.get("key", "")]]
    hits = [hit for hit in hits if hit is not None]
    if not hits:
        return None
    return next((hit for hit in hits if hit.get("start") is not None), hits[0])


def merged_row(record: dict, curated: dict, display=None):
    """This app's own version of a merged annotation, the located one, where one is located."""
    rows = [curated.get(key) for key in record.get("keys") or [record.get("key", "")]]
    rows = [row for row in rows if row]
    if not rows:
        return None
    return next((row for row in rows if _curated_location(row, display) is not None), rows[0])


def needs_location(record: dict, placement: dict | None, row: dict | None = None,
                 display=None) -> bool:
    """Does this annotation still need a segment and a trigger word set by hand?

    Three states. They are different failures and they cost a curator the same work, which is why
    one queue holds all three:

    * the file named words the report does not contain, ``placement is None``;
    * it named a sentence the trigger did not survive into, ``placement["start"] is None``;
    * it named **no trigger at all**, so there was never anything to place. This one is not a
      failure of locating, it is an annotation with neither half of the evidence, and it reads as
      "fine" to anything that only asks whether a stated trigger could be found.

    Restricted to :data:`ANCHORABLE`. A ``prior_annotation_2`` row is in the third state by design, not by accident, and its lexical candidate is offered separately.

    *row* is the curation log's version of the same annotation, and it is essential, not
    an optimisation: :func:`anchors.place_all` reads the annotation *files* and never sees the log,
    so without folding the log back in a row would stay in this queue after being located and the
    screen would never empty. What counts is a segment **and** a trigger that actually occurs in it, a saved trigger the named sentence does not contain is the second state over again.
    """
    if record.get("source") not in ANCHORABLE:
        return False
    if _curated_location(row, display) is not None:
        return False
    if not (record.get("trigger_word") or "").strip():
        return True
    return placement is None or placement.get("start") is None


def _curated_location(row, display):
    """``(segment, start, end)`` this app has already recorded for a row, or ``None``.

    Both halves or neither: a segment with no trigger word, or a trigger word the named sentence
    does not contain, is not an evidence location, it is the unplaced state with something typed into it.
    """
    if not row or not (row.get("trigger_word") or "").strip():
        return None
    try:
        idx = int(row.get("segment_idx"))
    except (TypeError, ValueError):
        return None
    trigger = row["trigger_word"].strip()
    if display is None:
        return (idx, None, None)
    if not 0 <= idx < len(display):
        return None
    span = locations.find_trigger(display[idx], trigger)
    return None if span is None else (idx, span[0], span[1])


def _curated_marks(rows) -> dict[int, list[dict]]:
    """``{segment_idx: [row, …]}`` for rows this app has already located to a segment."""
    out: dict[int, list[dict]] = {}
    for row in rows:
        if row.get("status") in ("rejected", "removed"):
            continue
        try:
            idx = int(row.get("segment_idx"))
        except (TypeError, ValueError):
            continue
        if row.get("trigger_word"):
            out.setdefault(idx, []).append(row)
    return out


def _segment(registry, idx: int, text: str, view: dict, curated: dict, palette: dict,
             selected: bool):
    """One segment, inline. The number lives in the tooltip, not in a gutter.

    A visible index per sentence would put a column of numbers back into the prose, which is the
    thing this view exists to remove. Hovering names the segment, the highlight shows its extent,
    and the Edit panel states which one is selected, three places that answer "which segment is
    this" without any of them interrupting the text.
    """
    marks = _marks_for(idx, text, view, curated)
    return html.Span(
        _weave(registry, idx, text, marks, palette),
        id={"type": "seg", "idx": idx},
        className="seg" + (" seg-selected" if selected else ""),
        title=f"segment #{idx}, click to target a suggestion here",
        n_clicks=0,
    )


#: Where each kind sits when marks compete for the same characters. ``curated`` and ``ann`` are
#: **peers**, both are somebody asserting a phenotype about these words, and neither outranks the
#: other. ``pb`` and ``cand`` are reference: a detection and a lexical guess. They never take the
#: underline from an assertion, which is what stops a wide PhenoBERT span from stretching a mark
#: across words no annotator claimed.
KIND_RANK = {"curated": 0, "ann": 0, "pb": 1, "cand": 2}

#: The tones the four kinds render in, most-decided first. A mark's tone is its dominant entry's.
TONE_ORDER = ("curated", "ann", "pb", "cand")


def _marks_for(idx: int, text: str, view: dict, curated: dict) -> list[dict]:
    """Every span to draw in this segment, non-overlapping, earliest first.

    Marks cannot nest. :func:`_weave` walks the segment with one forward cursor, so a mark starting
    behind it would emit the same characters twice and the prose would stop being the report. Every
    claim about one stretch of words therefore has to become **one mark carrying several entries**, one tag per phenotype, not several marks.

    Two overlapping triggers is not an edge case, it is how the ontology works: *seizure* is a
    substring of *focal onset seizure*, *delay* of *developmental delay*, and both members of such a
    pair are routinely annotated. So:

    * the **widest** span wins the underline, because that is the trigger word being read;
    * everything it covers, a second annotation, this app's own curated trigger, a PhenoBERT
      detection, folds onto it as an extra tag instead of being dropped;
    * an assertion outranks a reference (:data:`KIND_RANK`), so a PhenoBERT span never takes the
      underline from the narrower annotation inside it.

    This used to keep one mark and throw the rest away, with a single exception for an annotation
    nested inside another annotation. A curated trigger inside an annotation, an annotation inside a
    curated trigger, a detection over either, and any *crossing* pair all lost a mark silently, and
    in the crossing case the one lost was the longer of the two, so the screen underlined *onset*
    and not *focal onset seizures*. A curator reading the report against the panel found nothing
    there, and nothing distinguished that from an annotation which was never placed at all.
    """
    marks: list[dict] = []

    for det in view["detections"].get(idx, []):
        marks.append(_mark_for_entry(det["local_start"], det["local_end"], {
            "kind": "pb",
            "code": det.get("hpo_id", ""), "label": det.get("hpo_label", ""),
            "negated": bool(det.get("negated")), "resolved": bool(det.get("resolved", True)),
        }))

    # An annotation mark is keyed on the **span alone** and carries a list of entries: two files
    # agreeing on a code and the words are one entry naming both, and two *different* phenotypes on
    # The same words are two entries on one mark. Keying on (span, code), as this once did, made the
    # second case a collision the overlap pass then resolved by throwing one of them away.
    merged: dict[tuple, dict] = {}
    for record in view["annotations"].get(idx, []):
        # An annotation whose file named the sentence but whose trigger did not survive into it
        # has no character range, so there is nothing to underline. It is still listed in the
        # panel above. Drawing it across the whole sentence would be a claim its file never made.
        if record.get("start") is None or record.get("end") is None:
            continue
        at = (record["start"], record["end"])
        mark = merged.setdefault(at, {
            "start": record["start"], "end": record["end"], "kind": "ann", "entries": [],
        })
        _add_entry(mark, record)
    marks.extend(merged.values())

    for code, hits in view["triggers"].items():
        for hit in hits:
            if hit["segment_idx"] == idx:
                marks.append(_mark_for_entry(hit["start"], hit["end"], {
                    "kind": "cand", "code": code, "label": hit["phrase"],
                    "tier": hit.get("tier", "exact"),
                }))

    for row in curated.get(idx, []):
        trigger = str(row.get("trigger_word", ""))
        at = text.lower().find(trigger.lower()) if trigger else -1
        if at >= 0:
            marks.append(_mark_for_entry(at, at + len(trigger), {
                "kind": "curated", "code": row.get("hpo_code", ""),
                "label": row.get("hpo_name", ""), "status": row.get("status", ""),
            }))

    marks = [mark for mark in marks
             if mark["start"] >= 0 and mark["end"] <= len(text) and mark["end"] > mark["start"]]

    # Hosts are chosen over the whole segment, not against ``kept[-1]``: the previous pass
    # only ever compared with the last mark kept, which was correct solely because it never kept two
    # overlapping marks in the first place. Widest-first within a rank is what makes the bigger
    # trigger word the one that gets underlined.
    marks.sort(key=lambda m: (KIND_RANK[m["kind"]], -(m["end"] - m["start"]), m["start"]))
    kept: list[dict] = []
    covered: list[dict] = []
    for mark in marks:
        if any(mark["start"] < host["end"] and host["start"] < mark["end"] for host in kept):
            covered.append(mark)
        else:
            kept.append(mark)

    for mark in covered:
        host = _host_for(mark, kept)
        if host is None:  # unreachable, a covered mark overlaps something by design
            continue
        for entry in mark["entries"]:
            _merge_entry(host, entry)

    for mark in kept:
        _rank_entries(mark)
    kept.sort(key=lambda m: m["start"])
    return kept


def _mark_for_entry(start: int, end: int, entry: dict) -> dict:
    """A single-entry mark. Every kind carries an entry list, so one renderer serves all four."""
    return {"start": start, "end": end, "kind": entry["kind"], "entries": [_entry(entry)]}


def _entry(fields: dict) -> dict:
    """One entry, with every field the renderer may read present. Missing keys read as absent."""
    return {"kind": "ann", "code": "", "label": "", "sources": [], "how": "",
            "unconfirmed": False, "negated": False, "resolved": True, "status": "", "tier": "",
            **fields}


def _add_entry(mark: dict, record: dict) -> None:
    """Fold one annotation record into a mark's entry list, by code."""
    _merge_entry(mark, _entry({
        "kind": "ann",
        "code": record.get("hpo_code", ""), "label": record.get("hpo_name", ""),
        "sources": [record.get("source", "")], "how": record.get("how", ""),
        "unconfirmed": record.get("confirmed") is False,
    }))


def _merge_entry(mark: dict, entry: dict) -> None:
    """One entry per ``(kind, code)`` on a mark.

    Two *files* asserting one code are one entry naming both, that is what the source list is for.
    A PhenoBERT detection and a ground truth annotation for the same code are **not** one entry: they are
    different claims in different colours, and collapsing them would let a detection take over an
    annotation's authority. The redundant half is dropped later by :func:`_rank_entries`, where the
    decision is about what to *draw*, not about what is true.
    """
    for existing in mark["entries"]:
        if existing["kind"] == entry["kind"] and existing["code"] == entry["code"]:
            for source in entry["sources"]:
                if source not in existing["sources"]:
                    existing["sources"].append(source)
            existing["unconfirmed"] = existing["unconfirmed"] or entry["unconfirmed"]
            return
    mark["entries"].append(dict(entry))


def _rank_entries(mark: dict) -> None:
    """Order a mark's entries most-decided first, and drop the references that say nothing new.

    A PhenoBERT detection or a lexical candidate for a code an annotation on the same words already
    asserts adds no information, the agreement is in the panel above, and two identical tags side
    by side read as two phenotypes. A candidate is dropped against a *detection* too: it is a guess
    about where a bare code came from, and beside a detection of that same code it is a guess about
    something no longer in doubt. A reference naming a **different** code stays, because that is a
    disagreement, and disagreements are the reason to look.
    """
    asserted = {entry["code"] for entry in mark["entries"]
                if entry["kind"] in ("ann", "curated") and entry["code"]}
    detected = asserted | {entry["code"] for entry in mark["entries"]
                           if entry["kind"] == "pb" and entry["code"]}
    mark["entries"] = [
        entry for entry in mark["entries"]
        if entry["kind"] in ("ann", "curated")
        or entry["code"] not in (detected if entry["kind"] == "cand" else asserted)
    ]
    mark["entries"].sort(key=lambda entry: KIND_RANK[entry["kind"]])
    if mark["entries"]:
        mark["kind"] = mark["entries"][0]["kind"]


def _host_for(mark: dict, kept: list[dict]):
    """The kept mark *mark* belongs to: the one it shares the most characters with.

    Most-overlap, not first-overlap because a mark can straddle two hosts, and the one it
    mostly sits inside is the one whose words its tag is a claim about. Ties go to the earlier host,
    so the choice does not depend on the order the marks happened to be built in.
    """
    best, best_overlap = None, 0
    for host in sorted(kept, key=lambda h: h["start"]):
        overlap = min(mark["end"], host["end"]) - max(mark["start"], host["start"])
        if overlap > best_overlap:
            best, best_overlap = host, overlap
    return best


def _weave(registry, idx: int, text: str, marks: list[dict], palette: dict) -> list:
    """The segment as alternating plain text and clickable marks."""
    if not marks:
        return [text]
    out: list = []
    cursor = 0
    for mark in marks:
        if mark["start"] > cursor:
            out.append(text[cursor:mark["start"]])
        out.append(_mark(registry, idx, text[mark["start"]:mark["end"]], mark, palette))
        cursor = mark["end"]
    if cursor < len(text):
        out.append(text[cursor:])
    return out


def _mark(registry, idx: int, shown: str, mark: dict, palette: dict):
    """One highlighted span, carrying **one tag per phenotype asserted about these words**.

    The hover text always starts with :meth:`search.HPOSearch.describe`, name, id and the
    ontology's definition, and then adds what is specific to that entry. The definition is the half
    a reader cannot supply from memory, and the half that decides whether the mark is right.

    Where a mark carries more than one taggable entry the tags are the click targets and the
    underlined text is not, so "fill the suggestion from *this* term" stays unambiguous. With a
    single one the whole mark is clickable, as it has always been.
    """
    entries = mark.get("entries") or []
    if not entries:
        return html.Span(shown, className="mark-text")

    tone = _tone(mark, entries)
    # A lexical candidate is a guess about where a bare code came from, not a finding, so it has
    # never carried a tag, only the faint box and the hover text.
    taggable = [entry for entry in entries if entry["kind"] != "cand"]
    sole = len(taggable) <= 1
    title = "\n\n".join(_entry_title(registry, entry) for entry in entries)
    border = _entry_tag_color(entries[0], palette)

    children = [html.Span(shown, className="mark-text")]
    for entry in taggable:
        code = entry["code"]
        children.append(html.Span(
            code.replace("HP:", ""), className="mark-tag",
            style={"backgroundColor": _entry_tag_color(entry, palette)},
            # With one entry the whole mark is the click target. With several, each tag is its own,
            # clicking the words could only mean one of them, and picking silently is how the wrong
            # term gets pre-filled.
            **({} if sole else {"id": {"type": "mark", "idx": idx, "trigger": shown, "code": code},
                                "n_clicks": 0, "title": _entry_title(registry, entry)}),
        ))

    # The code travels in the id so a click can pre-fill the term as well as the trigger. It is what
    # The mark already *says*. Making the curator retype it would be asking them to copy the screen
    # back into the screen.
    attrs = {"id": {"type": "mark", "idx": idx, "trigger": shown,
                    "code": entries[0]["code"]}, "n_clicks": 0} if sole else {}
    return html.Span(children, className=f"mark mark-{tone}" + ("" if sole else " mark-multi"),
                     title=title, style={"borderColor": border}, **attrs)


def _tone(mark: dict, entries: list[dict]) -> str:
    """The mark's CSS tone: its most-decided entry's kind, with PhenoBERT's negation on top."""
    kinds = {entry["kind"] for entry in entries}
    for kind in TONE_ORDER:
        if kind in kinds:
            if kind == "pb" and any(e["kind"] == "pb" and e["negated"] for e in entries):
                return "pb-negated"
            return kind
    return mark["kind"]


def _entry_tag_color(entry: dict, palette: dict) -> str:
    """One entry's colour, its source's for an annotation, its kind's for everything else."""
    kind = entry["kind"]
    if kind == "ann":
        return _entry_color(entry.get("sources") or [], palette)
    if kind == "pb":
        # PhenoBERT is reference, so it takes the same neutral grey as its chips everywhere else.
        # This used to be ``categorical[2]``, marc2's amber, so a detection in the report and a
        # ``phenobert`` chip beside it were two different colours for one source.
        return theme.source_color("phenobert", palette)
    if kind == "curated":
        return palette["good"]
    return palette["text_muted"]


def _entry_title(registry, entry: dict) -> str:
    """The hover block for one entry: the ontology's definition, then what is specific to it."""
    code, label, kind = entry["code"], entry.get("label", ""), entry["kind"]

    if kind == "ann":
        sources = entry.get("sources") or [""]
        block = registry.search.describe(code, prefix=f"{' + '.join(sources)} →")
        extra = [HOW_HELP.get(entry.get("how", ""), "")]
        if label and code and label.lower() != registry.search.label(code).lower():
            extra.append(f"the file calls it “{label}”")
        if entry["unconfirmed"]:
            extra.append("not confirmed in its file")
        joined = " · ".join(part for part in extra if part)
        return block + ("\n" + joined if joined else "")

    if kind == "pb":
        block = registry.search.describe(code, prefix="PhenoBERT →")
        extra = []
        # PhenoBERT's own label, when it differs from the ontology's, is worth seeing: it is the
        # string the linker matched, and a mismatch is how a wrong normalisation shows itself.
        if label and label.lower() != registry.search.label(code).lower():
            extra.append(f"PhenoBERT called it “{label}”")
        if entry["negated"]:
            extra.append("negated, found and ruled out")
        if not entry["resolved"]:
            extra.append("code not in this HPO release")
        return block + ("\n\n" + " · ".join(extra) if extra else "")

    if kind == "curated":
        block = registry.search.describe(code, prefix="curated →")
        return block + (f"\n\nstatus: {entry['status']}" if entry["status"] else "")

    return (registry.search.describe(code, prefix="possible trigger for")
            + f"\n\nMatched the ontology phrase “{label}”"
            + (" by inflection" if entry.get("tier") == "inflected" else "")
            + ". A guess, not a finding.")


def _entry_color(sources, palette: dict) -> str:
    """The colour of the highest-precedence source among *sources*.

    A source with a precedence is one that can be adjudicated; ``prior_annotation_2`` and anything unknown have
    none and sort last. Deciding by file order instead meant the *first* file to mention a code
    coloured the mark, which is an accident of ``GOLD_SOURCES``, not a fact about the
    annotation.
    """
    ranked = sorted(
        (s for s in sources if s),
        key=lambda s: (sources_mod.precedence(s) is None, sources_mod.precedence(s) or 0),
    )
    return theme.source_color(ranked[0] if ranked else "", palette)
