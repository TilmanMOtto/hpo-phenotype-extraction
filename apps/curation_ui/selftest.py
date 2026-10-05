"""Headless acceptance check: every panel, every awkward patient, then the HTTP surface.

    python apps/curation_ui/app.py --selftest              # against the synthetic fixture
    python apps/curation_ui/app.py --selftest --hcy-dir …  # against the real cohort

Exits 0 only if all four layers pass. Run it on the cluster before trusting a screen, and in CI
against the fixture.

Why a render harness at all: a Dash callback that raises does not crash the process, it returns a
500 for one component and the page renders around the hole. On an analysis app that is a missing
chart. Here it is a curator recording a judgement against a panel that silently failed to draw the
evidence, which is a corrupted ground-truth set discovered months later. So the failure modes are provoked
on purpose.

Four layers, because they catch different things:

**Round trip**, write events, re-read the log from disk, and check the folded state matches. This
is the layer that would catch a persistence bug, and persistence is the only thing here that cannot
be recovered by reloading the page.

**Render matrix**, every panel for every patient in both modes, including the degenerate ones a
click-through never reaches: a patient with no annotation, one with no segmentation, one with no
PhenoBERT detection, a code the ontology cannot resolve, a segment that will not align. It also
pins the invariant the screen exists for: **every mark drawn from an annotation file sits on the
words that file named**, which is the one thing a curator reading a report cannot check for
themselves.

**Export**, the curated ground truth read back through ``HCYDataset.load_ground_truth``, the loader every
scoring job uses. A file this app writes that the pipeline cannot read is not an export.

**HTTP**, ``/``, ``/_dash-layout`` and ``/_dash-dependencies`` through Flask's test client. These
catch duplicate component ids and callbacks wired to targets that do not exist, neither of which
any amount of calling ``render_*`` reveals.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import traceback

logger = logging.getLogger(__name__)


def _versions() -> str:
    import dash
    import pandas

    return f"dash {dash.__version__}, pandas {pandas.__version__}, python {sys.version.split()[0]}"


class _Report:
    """Collects failures so one broken panel does not hide the other nine."""

    def __init__(self):
        self.failures: list[str] = []
        self.checks = 0

    def check(self, name: str, fn):
        """Run *fn* as a named check and record a failure instead of raising."""
        self.checks += 1
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - the point is to collect, not to propagate
            self.failures.append(f"{name}: {exc}\n{traceback.format_exc()}")
            logger.error("FAIL  %s, %s", name, exc)
        else:
            logger.info("ok    %s", name)

    def nonempty(self, name: str, fn):
        """A panel must return *something*. ``None`` is how a silently-skipped render looks."""
        def _run():
            result = fn()
            if result is None:
                raise AssertionError("rendered None")
        self.check(name, _run)


def run(hcy_dir: str | None = None, phenobert_dir: str | None = None) -> int:
    """Return an exit code. 0 only if every layer passed."""
    logging.getLogger().setLevel(logging.INFO)
    logger.info("selftest, %s", _versions())

    from hpo_extraction.curation import evidence_location as locations, labels as vocab, sources, store
    from apps.curation_ui import fixture, state
    from apps.curation_ui.registry import Registry
    from apps.curation_ui.views import (
        approve, comments, common, edit, editor, labelling, locate, overview, reader,
    )

    report = _Report()
    tmp = tempfile.TemporaryDirectory(prefix="hcy-curation-selftest-")

    real = bool(hcy_dir) and os.path.isfile(
        os.path.join(hcy_dir, sources.SEGMENTS_FILE)
    )
    if real:
        paths = sources.default_paths(hcy_dir, phenobert_dir or "")
        # Never write into the cohort directory during a test run.
        curation_dir = os.path.join(tmp.name, "curation")
        logger.info("running against the real cohort at %s (log redirected to %s)",
                    hcy_dir, curation_dir)
    else:
        paths = fixture.write(tmp.name)
        curation_dir = os.path.join(tmp.name, "curation")
        logger.info("running against the synthetic fixture in %s", tmp.name)

    registry = Registry(paths, author="selftest", curation_dir=curation_dir)
    problems = registry.validate()
    if problems:
        logger.error("registry did not validate: %s", " ".join(problems))
        return 2
    state.set_registry(registry)
    common.clear_failures()

    patients = registry.patient_ids
    logger.info("%d patient(s): %s", len(patients), ", ".join(patients[:8]))

    # ── layer 1: persistence round trip ──────────────────────────────────────
    def _round_trip():
        subject = next((p for p in patients if registry.patient(p)["segments"]), None)
        if subject is None:
            raise AssertionError("no patient has a segmented report")
        view = registry.patient(subject)
        segment = view["display"][0]
        trigger = segment.split()[0]

        event = registry.record(
            "suggest", subject, segment_idx=0, segment_text=segment,
            hpo_code="HP:0001250", hpo_name="Seizure", trigger_word=trigger, note="selftest",
        )
        registry.record("approve", subject, target_key=event["event_id"])

        key = store.target_key("raw", subject, "HP:0001252")
        registry.record("verdict", subject, target_key=key, hpo_code="HP:0001252",
                        hpo_name="Hypotonia", status="kept")
        registry.record("anchor", subject, target_key=key, segment_idx=0,
                        segment_text=segment, trigger_word=trigger)
        registry.record("confirm_patient", subject)

        # A comment on the report, and a deletion proposed against a *different* annotation,
        # The one above carries the verdict/evidence location assertions and must stay settled.
        note = registry.record("comment", subject, text="selftest, the report reads oddly")
        disputed_key = store.target_key("prior_annotation_2", subject, "HP:0001903")
        registry.record("suggest_delete", subject, target_key=disputed_key,
                        hpo_code="HP:0001903", text="selftest, unsupported by the text")

        # Difficulty labels, at both levels. These are the rows a stratified recall table is built
        # from, so losing one is losing a judgement nobody will notice is missing.
        registry.record("label", subject, labels=["family_member", "paraphrase"],
                        difficulty="hard", note="selftest")
        registry.record("label", subject, target_key=key, labels=["family_member", "paraphrase"],
                        difficulty="unfair", hpo_code="HP:0001252")

        # Re-read from disk, not from the in-memory log: this is the layer that proves a restart
        # resumes, which is the one failure a curator cannot work around.
        reloaded = Registry(paths, author="selftest", curation_dir=curation_dir)
        rows = {r["key"]: r for r in reloaded.rows_for(subject)}
        if rows.get(event["event_id"], {}).get("status") != "approved":
            raise AssertionError("approved suggestion did not survive a reload")
        anchored = rows.get(key, {})
        if anchored.get("status") != "kept" or anchored.get("trigger_word") != trigger:
            raise AssertionError(f"verdict/evidence location did not survive a reload: {anchored}")
        if not reloaded.is_confirmed(subject):
            raise AssertionError("confirm_patient did not survive a reload")

        comments = reloaded.comments_for(subject)
        if [c["comment_id"] for c in comments] != [note["event_id"]]:
            raise AssertionError(f"the comment did not survive a reload: {comments}")
        if not os.path.isfile(reloaded.log.comments_path):
            raise AssertionError("curation_comments.csv was not written")

        disputed = next((r for r in reloaded.rows_for(subject) if r["key"] == disputed_key), {})
        if disputed.get("status") != store.DELETE_SUGGESTED:
            raise AssertionError(f"the deletion proposal did not survive a reload: {disputed}")
        if disputed.get("status") in store.IN_GOLD:
            raise AssertionError("a disputed annotation is still in the ground truth")
        if disputed.get("delete_reason") != "selftest, unsupported by the text":
            raise AssertionError(f"the proposal lost its reason: {disputed}")

        # Withdrawing must return it to undecided, not to kept.
        reloaded.record("withdraw_delete", subject, target_key=disputed_key)
        after = next(r for r in reloaded.rows_for(subject) if r["key"] == disputed_key)
        if after.get("status") or after.get("delete_reason"):
            raise AssertionError(f"withdrawing left the proposal behind: {after}")

        doc = reloaded.patient_labels(subject)
        if doc["difficulty"] != "hard" or "family_member" not in doc["labels"]:
            raise AssertionError(f"report labels did not survive a reload: {doc}")
        relabelled = reloaded.rows_for(subject)
        tagged = next((r for r in relabelled if r["key"] == key), {})
        if tagged.get("difficulty") != "unfair" or "family_member" not in (tagged.get("labels") or []):
            raise AssertionError(f"annotation labels did not survive a reload: {tagged}")
        if tagged.get("status") != "kept":
            raise AssertionError("labelling overwrote the verdict")

        # Replace-not-merge: unticking must be possible, or nobody will risk ticking.
        reloaded.record("label", subject, target_key=key, labels=[], difficulty="",
                        hpo_code="HP:0001252")
        cleared = next(r for r in reloaded.rows_for(subject) if r["key"] == key)
        if cleared.get("labels") or cleared.get("difficulty"):
            raise AssertionError(f"labels could not be cleared: {cleared}")

        # The long-form file is what the whole feature is for.
        for path in (reloaded.log.labels_path, reloaded.log.report_labels_path):
            if not os.path.isfile(path):
                raise AssertionError(f"{os.path.basename(path)} was not written")

        # A torn final line is what a crash mid-append leaves. It must cost that line, not the log.
        with open(reloaded.log.path, "a", encoding="utf-8") as handle:
            handle.write('{"action":"appr')
        torn = Registry(paths, author="selftest", curation_dir=curation_dir)
        if len(torn.rows_for(subject)) != len(rows):
            raise AssertionError("a torn trailing line changed the folded state")

    report.check("persistence round trip", _round_trip)

    # ── layer 2: render matrix ───────────────────────────────────────────────
    for mode in ("light", "dark"):
        for patient_id in patients:
            view = registry.patient(patient_id)
            n_segments = len(view["segments"])
            # Both a valid selection and no selection at all: the second is the state the screen
            # opens in, and the one an Edit panel is most likely to get wrong.
            for segment_idx in ({None, 0, n_segments - 1} if n_segments else {None}):
                label = f"{patient_id}/{mode}/seg={segment_idx}"
                report.nonempty(f"reader {label}",
                                lambda p=patient_id, s=segment_idx, m=mode:
                                    reader.render(p, s, m))
                report.nonempty(f"edit-segment {label}",
                                lambda p=patient_id, s=segment_idx:
                                    edit.render_selected(p, s))
                report.check(f"approve {label}",
                             lambda p=patient_id, m=mode: _assert_renders(approve.render(p, m)))
            report.check(f"edit-list {patient_id}/{mode}",
                         lambda p=patient_id, m=mode: edit.render_list(p, m))
            report.nonempty(f"ground truth panel {patient_id}/{mode}",
                            lambda p=patient_id, m=mode: reader.render_gold_panel(
                                registry, registry.patient(p), m))
            # The Evidence location tab's version of the same panel: the same list, cut to the by-hand queue.
            report.nonempty(f"ground truth panel unanchored {patient_id}/{mode}",
                            lambda p=patient_id, m=mode: reader.render_gold_panel(
                                registry, registry.patient(p), m, only_unanchored=True))
            report.nonempty(f"evidence location queue {patient_id}/{mode}",
                            lambda p=patient_id, m=mode: locate.render_queue(p, m))
            report.nonempty(f"report labels {patient_id}/{mode}",
                            lambda p=patient_id, m=mode: labelling.report_summary(
                                registry, p, m))
            report.nonempty(f"deletable {patient_id}/{mode}",
                            lambda p=patient_id, m=mode: edit.render_deletable(p, m))
            report.check(f"comments {patient_id}/{mode}",
                         lambda p=patient_id, m=mode: comments.render_comments(p, m))

    for mode in ("light", "dark"):
        report.nonempty(f"evidence location worklist {mode}", lambda m=mode: locate.render_worklist(m))

    # Degenerate inputs the sidebar can produce but a click-through cannot.
    report.nonempty("reader with no patient", lambda: reader.render(None, None, "light"))
    report.nonempty("reader on the evidence location tab",
                    lambda: reader.render(patients[0], None, "light", only_unanchored=True))
    report.nonempty("evidence location queue with no patient", lambda: locate.render_queue(None, "light"))
    report.nonempty("evidence location queue with unknown patient",
                    lambda: locate.render_queue("NOT_A_PATIENT", "light"))
    report.nonempty("reader with unknown patient",
                    lambda: reader.render("NOT_A_PATIENT", None, "light"))
    report.nonempty("approve with no patient", lambda: approve.render(None, "light"))
    report.nonempty("edit-segment out of range",
                    lambda: edit.render_selected(patients[0], 10_000))
    # The qualifier chip line. The empty case renders *nothing* on purpose, most rows carry no
    # qualifier and a line saying so on each of them is a line of noise per row, so it is checked
    # rather than asserted non-empty.
    report.check("annotation qualifiers, empty",
                 lambda: _assert_no_chips(labelling.qualifier_chips([], "light")))
    report.nonempty("annotation qualifiers, populated",
                    lambda: labelling.qualifier_chips(["negated", "family"], "dark"))
    report.nonempty("qualifier controls",
                    lambda: labelling.qualifier_controls("selftest-labels", ["negated"]))
    report.nonempty("deletable with no patient", lambda: edit.render_deletable(None, "light"))
    report.nonempty("editor panel, empty",
                    lambda: editor.panel(registry, "k", "approve"))
    # One per ``at``: a panel mounted with an ``at`` the editor's callbacks do not match on is a
    # form whose Save silently does nothing, and nothing else on this list would notice.
    report.nonempty("editor panel, evidence location tab",
                    lambda: editor.panel(registry, "k", locate.AT, segment_idx=0,
                                         trigger_word="seizures", hpo_code="HP:0001250"))
    report.nonempty("editor panel, populated",
                    lambda: editor.panel(registry, "k", "edit", segment_idx=0,
                                         trigger_word="seizures", hpo_code="HP:0001250",
                                         note="a note", labels=["negated"], mode="dark"))
    report.nonempty("overview difficulty", lambda: overview.render_difficulty("light"))
    report.nonempty("overview stats", lambda: overview.render_stats("light"))
    report.nonempty("overview table", lambda: overview.render_table())
    report.check("overview markdown", lambda: _assert_text(overview.markdown_summary()))

    # The Edit form's own guard rail. A suggestion whose trigger is not in its segment is the one
    # bad record this app can produce, so the refusal is tested, not assumed.
    def _validation():
        subject = next(p for p in patients if registry.patient(p)["segments"])
        segment = registry.patient(subject)["display"][0]
        cases = [
            (None, "x", "HP:0001250", "no segment"),
            (0, "", "HP:0001250", "no trigger"),
            (0, "zzzz-not-in-this-segment", "HP:0001250", "trigger not in segment"),
            (0, segment.split()[0], None, "no code"),
        ]
        for segment_idx, trigger, code, why in cases:
            if not edit._validate(registry, subject, segment_idx, trigger, code):
                raise AssertionError(f"accepted a suggestion with {why}")
        if edit._validate(registry, subject, 0, segment.split()[0], "HP:0001250"):
            raise AssertionError("rejected a complete, consistent suggestion")

    report.check("edit validation", _validation)

    # Reports pasted out of a lab system pad their columns with tabs, and every route a trigger
    # arrives by has already flattened them, HTML collapses whitespace when it renders, so the
    # browser's selection and anything retyped carry single spaces. Matching literally refused a
    # trigger over a difference nothing on screen can show. Both halves are fixed: the match has
    # to succeed, and what gets *written* has to be the report's own spelling.
    def _whitespace_trigger():
        padded = "Serum Ammonia\t\t87 umol/l, and hypotonia."
        for typed, expect in [
            ("Serum Ammonia 87", "Serum Ammonia\t\t87"),
            ("serum ammonia  87 umol/l", "Serum Ammonia\t\t87 umol/l"),
            ("hypotonia", "hypotonia"),
        ]:
            span = locations.find_trigger(padded, typed)
            if span is None:
                raise AssertionError(f"{typed!r} not found in a tab-padded segment")
            if padded[span[0]:span[1]] != expect:
                raise AssertionError(f"{typed!r} → {padded[span[0]:span[1]]!r}, wanted {expect!r}")
        if locations.find_trigger(padded, "Serum 87") is not None:
            raise AssertionError("whitespace tolerance matched across an unrelated word")

        # …and the round trip through a real segment: flattening its whitespace must still
        # validate, and must snap back to the exact characters the report carries.
        subject = next(p for p in patients if registry.patient(p)["segments"])
        for idx, segment in enumerate(registry.patient(subject)["display"][:5]):
            words = segment.split()
            if len(words) < 3:
                continue
            flat = " ".join(words[:3])
            verbatim = segment[slice(*locations.find_trigger(segment, flat))]
            if common.validate_annotation(registry, subject, idx, flat, "HP:0001250"):
                raise AssertionError(f"rejected a flattened trigger on segment #{idx}")
            snapped = common.snap_trigger(registry, subject, idx, flat)
            if snapped != verbatim:
                raise AssertionError(f"snap gave {snapped!r}, report spells it {verbatim!r}")

    report.check("whitespace-tolerant triggers", _whitespace_trigger)

    # The search contract: never more than a page, never anything on a keystroke.
    def _search():
        from apps.curation_ui import search as search_mod

        if len(registry.search.search("a" * 1)) != 0:
            raise AssertionError("a one-character query returned options")
        wide = registry.search.search("ab")
        if len(wide) > search_mod.MAX_RESULTS:
            raise AssertionError(f"returned {len(wide)} options, cap is {search_mod.MAX_RESULTS}")
        if not registry.search.search("HP:0001250"):
            raise AssertionError("an exact HPO id found nothing")

    report.check("hpo search contract", _search)

    # Ids are read off ground truth files, where the ``HP:`` prefix is often already gone. Every spelling a
    # curator can plausibly paste has to land on the same term, and a shorter digit string has to
    # narrow, not fail, that is what makes typing an id a lookup instead of a gamble.
    def _search_by_id():
        for query in ("HP:0001250", "hp:0001250", "HP0001250", "0001250", "1250"):
            hits = registry.search.search(query)
            if not hits or hits[0]["value"] != "HP:0001250":
                raise AssertionError(f"{query!r} did not resolve to HP:0001250: {hits[:3]}")
        prefix = registry.search.search("00012")
        if not prefix or any(not h["value"].startswith("HP:00012") for h in prefix):
            raise AssertionError(f"a partial id returned non-matching codes: {prefix[:5]}")
        if len(prefix) < 2:
            raise AssertionError("a partial id narrowed to a single term")
        # A digit string is an id. A query that merely contains digits is still text.
        if registry.search.search("HP:9999999"):
            raise AssertionError("an id no release carries returned options")

    report.check("hpo search by id", _search_by_id)

    # The completion is clientside, so nothing here can exercise it, but a missing asset is a
    # silent 404 that leaves the field working as it did before, which is the failure mode
    # nobody notices. Fix that Dash is actually serving it.
    def _trigger_asset():
        from apps.curation_ui.app import build_app

        app = build_app(registry)
        client = app.server.test_client()
        response = client.get("/assets/trigger_complete.js")
        if response.status_code != 200:
            raise AssertionError(f"trigger_complete.js → HTTP {response.status_code}")
        if b"MIN_PREFIX" not in response.data:
            raise AssertionError("trigger_complete.js served, but not the file we wrote")

    report.check("trigger completion asset", _trigger_asset)

    # Definitions are the hover text on every phenotype in the app. A term the ontology does not
    # define must say so, not hovering to an empty box.
    def _definitions():
        described = registry.search.describe("HP:0001250")
        if "HP:0001250" not in described or len(described.splitlines()) < 2:
            raise AssertionError(f"describe() produced no definition block: {described!r}")
        unknown = registry.search.describe("HP:9999999")
        if unknown.count("HP:9999999") != 1 or "hpo.json" not in unknown:
            raise AssertionError(f"an unknown code described badly: {unknown!r}")

    report.check("term definitions", _definitions)

    # The report is drawn as prose, so what is drawn must *be* the report. A rendering that dropped
    # a sentence would be invisible to a curator, who is reading it because they do not
    # already know what it says.
    def _prose_is_verbatim():
        checked = 0
        for patient_id in patients:
            view = registry.patient(patient_id)
            if not view["text"] or view["alignment"]["n_unaligned"]:
                continue                # An unalignable segment is an addition, not a loss
            body = reader._prose(registry, view, {}, {"categorical": ["#000"] * 8,
                                                      "good": "#0a0", "text_muted": "#888",
                                                      "text": "#000"}, None)
            rendered = "".join(_strings(body))
            if rendered != view["text"]:
                raise AssertionError(
                    f"{patient_id}: the prose is not the report\n  got  {rendered!r}\n"
                    f"  want {view['text']!r}")
            checked += 1
        if not checked:
            raise AssertionError("no patient could be checked for verbatim prose")

    report.check("prose is the verbatim report", _prose_is_verbatim)

    # Every mark must carry the code it displays, or a click cannot pre-fill the term.
    def _marks_carry_their_code():
        for patient_id in patients:
            view = registry.patient(patient_id)
            for ident in _mark_ids(reader._prose(
                    registry, view, {}, {"categorical": ["#000"] * 8, "good": "#0a0",
                                         "text_muted": "#888", "text": "#000"}, None)):
                if not {"idx", "trigger", "code"} <= set(ident):
                    raise AssertionError(f"{patient_id}: mark id is incomplete: {ident}")

    report.check("marks carry their code", _marks_carry_their_code)

    # Anything arriving from a browser is untrusted: a label from the wrong level, or one this
    # version does not know, must be dropped, not written into the export.
    def _label_vocabulary():
        cleaned = vocab.clean(["family_member", "granularity", "nonsense", "hard"], "patient")
        if cleaned != ["family_member"]:
            raise AssertionError(f"scope/validity filtering is wrong: {cleaned}")
        # The two levels are disjoint now. A report label posted against an annotation, or a
        # qualifier posted against a report, is dropped, not stored, the levels are two
        # files and two joins, and a value in the wrong one is a row nothing can interpret.
        if vocab.clean(["family_member", "negated"], "annotation") != ["negated"]:
            raise AssertionError("a report label was accepted at annotation level")
        if vocab.clean(["negated", "family_member"], "patient") != ["family_member"]:
            raise AssertionError("a qualifier was accepted at report level")
        if vocab.grade_ok("unfair", "patient"):
            raise AssertionError("unfair was offered at report level")
        if vocab.difficulty_for("annotation"):
            raise AssertionError("the annotation level still offers a grade")
        # Retired but still readable: a log that recorded one of the old fine-grained labels must
        # still render its name, even though nothing offers it any more.
        if vocab.display("granularity") == "granularity":
            raise AssertionError("a retired label lost its display name")
        if vocab.display("nonsense") != "nonsense":
            raise AssertionError("an unknown label did not fall back to its id")
        # Ordering must come from the vocabulary, not the clicks, or two curators who tick the
        # same boxes produce two different rows.
        first = vocab.clean(["negated", "family"], "annotation")
        second = vocab.clean(["family", "negated"], "annotation")
        if first != second:
            raise AssertionError(f"label order depends on click order: {first} vs {second}")

    report.check("label vocabulary", _label_vocabulary)

    # Every mark drawn from an annotation file must sit on the words that file named. This is the
    # one thing on the screen a curator cannot check: they are reading the report because
    # they do not already know where the annotation belongs.
    def _placements_are_the_trigger():
        checked = 0
        for patient_id in patients:
            view = registry.patient(patient_id)
            for segment_idx, rows in view["annotations"].items():
                segment = view["display"][segment_idx]
                for record in rows:
                    if record["start"] is None:
                        # A segment claimed, a trigger that did not survive into it. Allowed, and
                        # The reader draws no underline for it, but it must still be a real
                        # segment, or the panel would name one that is not on screen.
                        if not (0 <= segment_idx < len(view["display"])):
                            raise AssertionError(
                                f"{patient_id}: annotation placed on segment #{segment_idx}, "
                                f"which does not exist")
                        continue
                    found = segment[record["start"]:record["end"]]
                    if found.lower() != record["trigger_word"].strip().lower():
                        raise AssertionError(
                            f"{patient_id}/{record['source']}/{record['hpo_code']}: mark covers "
                            f"{found!r}, trigger is {record['trigger_word']!r}")
                    checked += 1
        if not checked and registry.gold_sources:
            raise AssertionError("no annotation was placed anywhere in the cohort")

    report.check("annotation marks sit on their trigger", _placements_are_the_trigger)

    # A source that says nothing about position is not "unplaced", it never claimed a place. The
    # unplaced list is the by-hand queue, and padding it with every code-only term empties it of
    # meaning.
    def _unplaced_means_something():
        for patient_id in patients:
            for record in registry.patient(patient_id)["unplaced_annotations"]:
                if not locations.has_position(record):
                    raise AssertionError(
                        f"{patient_id}: {record['source']}/{record['hpo_code']} is listed unplaced "
                        "but never claimed a position")

    report.check("unplaced means a claim that failed", _unplaced_means_something)

    # The Evidence location tab's whole premise: placing an annotation by hand takes it out of the queue.
    # ``anchors.place_all`` reads the annotation *files* and never sees the curation log, so nothing
    # about the placement pass alone would ever empty this screen, the log has to be folded back
    # in, and this is the check that says so.
    def _locating_empties_the_queue():
        subject, entry = None, None
        for patient_id in patients:
            queue, _ = locate.queue_for(registry, patient_id)
            queue = [row for row in queue if (row["record"].get("trigger_word") or "").strip()]
            if queue:
                subject, entry = patient_id, queue[0]
                break
        if entry is None:
            raise AssertionError("no report in the cohort has an annotation needing an evidence location")

        view = registry.patient(subject)
        segment_idx = 0
        segment = view["display"][segment_idx]
        trigger = segment.split()[0]
        before = len(locate.queue_for(registry, subject)[0])

        registry.record("edit", subject, target_key=entry["key"], segment_idx=segment_idx,
                        segment_text=segment, trigger_word=trigger,
                        hpo_code=entry["code"], note="selftest, located by hand")
        after = [row["key"] for row in locate.queue_for(registry, subject)[0]]
        if entry["key"] in after:
            raise AssertionError(f"{subject}: {entry['key']} stayed in the queue after being "
                                 "given a segment and a trigger word")
        if len(after) != before - 1:
            raise AssertionError(f"{subject}: locating one row moved the queue from {before} to "
                                 f"{len(after)}")

        # A trigger the named sentence does not contain is the unplaced state over again, so the
        # row has to come back, not count as done.
        registry.record("edit", subject, target_key=entry["key"], segment_idx=segment_idx,
                        segment_text=segment, trigger_word="notinthissentenceatall",
                        hpo_code=entry["code"])
        if entry["key"] not in [row["key"] for row in locate.queue_for(registry, subject)[0]]:
            raise AssertionError(f"{subject}: a trigger absent from its own segment counted as "
                                 "anchored")

    report.check("locating empties the queue", _locating_empties_the_queue)

    # ``marc2`` is not in the union the curated ground truth is built from, and it never claimed a position
    # in the first place. One of its rows in this queue would put every code-only term in it.
    def _the_queue_is_the_union():
        for patient_id in patients:
            for entry in locate.queue_for(registry, patient_id)[0]:
                if entry["source"] not in reader.ANCHORABLE:
                    raise AssertionError(
                        f"{patient_id}: {entry['source']}/{entry['code']} is in the locating "
                        "queue, but its source is not one the curated ground truth is built from")

    report.check("the locating queue is prior_annotation + daphne + suggestions", _the_queue_is_the_union)

    # The complaint this merge exists to answer, stated as an invariant. Two rows for one code can
    # only stay apart by sitting on **different words**, and a row that sits on words at all is not
    # in this queue, so a code appearing twice here means something asked a curator to place one
    # phenotype twice, and the second time there is nothing left to place.
    def _no_code_is_queued_twice():
        for patient_id in patients:
            seen: dict[str, str] = {}
            for entry in locate.queue_for(registry, patient_id)[0]:
                if entry["code"] in seen:
                    raise AssertionError(
                        f"{patient_id}: {entry['code']} is in the locating queue twice, "
                        f"{seen[entry['code']]} and {'+'.join(entry['sources'])}")
                seen[entry["code"]] = "+".join(entry["sources"])

    report.check("no phenotype is queued for locating twice", _no_code_is_queued_twice)

    # The one row nothing else could ever close: a file's claim that will never start working, for a
    # term the curator has since placed by hand. Without the fold the queue goes on asking for a
    # phenotype that is already sitting on the words it came from.
    def _placing_a_term_closes_its_broken_file_row():
        for patient_id in patients:
            view = registry.patient(patient_id)
            queue = [e for e in locate.queue_for(registry, patient_id)[0]
                     if view and view["display"]]
            if not queue:
                continue
            entry = queue[0]
            segment_idx = 0
            segment = view["display"][segment_idx]
            registry.record("suggest", patient_id, segment_idx=segment_idx,
                            segment_text=segment, hpo_code=entry["code"],
                            hpo_name=entry["record"].get("hpo_name", ""),
                            trigger_word=segment.split()[0], note="selftest, placed by hand")
            still = {e["code"] for e in locate.queue_for(registry, patient_id)[0]}
            if entry["code"] in still:
                raise AssertionError(
                    f"{patient_id}: {entry['code']} is located by this app and still queued")
            return
        raise AssertionError("no report in the cohort has an annotation needing an evidence location")

    report.check("placing a term closes its broken file row",
                 _placing_a_term_closes_its_broken_file_row)

    # Editing an annotation: the repair must survive a reload, and must not decide anything.
    def _edit_round_trip():
        subject, record = next(
            ((p, r) for p in patients for r in registry.records_for(p)
             if registry.patient(p)["segments"]),
            (None, None))
        if subject is None:
            raise AssertionError("no patient has both a segmentation and an annotation")
        key = record["key"]
        view = registry.patient(subject)
        segment = view["display"][0]
        trigger = segment.split()[0]

        # Its own log. The round-trip check above leaves a torn final line behind, and
        # The next append after one of those is glued onto it and lost, which is the correct
        # behaviour under test there, and would silently eat the verdict written here.
        edit_dir = os.path.join(tmp.name, "curation-edit")
        writable = Registry(paths, author="selftest", curation_dir=edit_dir)
        writable.record("verdict", subject, target_key=key, hpo_code=record["hpo_code"],
                        status="needs_work")
        writable.record("edit", subject, target_key=key, segment_idx=0, segment_text=segment,
                        trigger_word=trigger, hpo_code="HP:0001250", hpo_name="Seizure",
                        note="selftest, repaired")

        reloaded = Registry(paths, author="selftest", curation_dir=edit_dir)
        row = next((r for r in reloaded.rows_for(subject) if r["key"] == key), {})
        if row.get("trigger_word") != trigger or _as_int(row.get("segment_idx")) != 0:
            raise AssertionError(f"the edit did not survive a reload: {row}")
        if row.get("hpo_code") != "HP:0001250" or row.get("note") != "selftest, repaired":
            raise AssertionError(f"the edit lost a field: {row}")
        if row.get("status") != "needs_work":
            raise AssertionError("an edit changed the verdict")
        if row.get("key") != key:
            raise AssertionError("an edit re-keyed the annotation away from its file")

        # And clearing a note must be possible, or the generic truthy-write rule has crept back.
        reloaded.record("edit", subject, target_key=key, note="")
        cleared = next(r for r in reloaded.rows_for(subject) if r["key"] == key)
        if cleared.get("note"):
            raise AssertionError(f"a note could not be cleared by an edit: {cleared}")

    report.check("edit round trip", _edit_round_trip)

    # Which rows are ruled on. Getting this wrong is invisible on screen, the panel looks complete
    # either way, and produces either a ground-truth set assembled from a file that was only a cross-check,
    # or one missing every term the confirmed pass dropped.
    def _adjudication_order():
        reference = [spec["id"] for spec in sources.GOLD_SOURCES if spec["precedence"] is None]
        for patient_id in patients:
            view = registry.patient(patient_id)
            existing = view["existing"]
            seen = {record["source"] for record in existing}
            leaked = seen & set(reference)
            if leaked:
                raise AssertionError(f"{patient_id}: reference source(s) {sorted(leaked)} are "
                                     "being ruled on")
            # Per code, one source, the best one that has an annotation for it.
            by_code: dict[str, set[str]] = {}
            for record in existing:
                by_code.setdefault(record["hpo_code"], set()).add(record["source"])
            for code, from_sources in by_code.items():
                if len(from_sources) > 1:
                    raise AssertionError(
                        f"{patient_id}/{code}: ruled on from {sorted(from_sources)} at once, the "
                        "same question twice, free to get two answers")
                carried = {r["source"] for r in view["records"] if r["hpo_code"] == code}
                best = next(s for s in sources.ADJUDICATION_ORDER
                            if s in carried and s in registry.gold_sources)
                if from_sources != {best}:
                    raise AssertionError(
                        f"{patient_id}/{code}: ruled on from {from_sources}, best available is "
                        f"{best}")

    report.check("adjudication order", _adjudication_order)

    # The agreement label is what every row says about the disagreement this app exists to resolve.
    def _agreement_labels():
        loaded = registry.gold_sources
        if not loaded:
            raise AssertionError("no annotation source loaded")
        label, _ = common.agreement(set(loaded), loaded, False)
        if len(loaded) > 1 and label != "all sources":
            raise AssertionError(f"every source carrying it did not read as agreement: {label}")
        only, why = common.agreement({loaded[0]}, loaded, False)
        if not only.startswith(loaded[0]) or loaded[0] not in why:
            raise AssertionError(f"a single-source label named the wrong file: {only} / {why}")
        none_label, _ = common.agreement(set(), loaded, True)
        if none_label != "PhenoBERT only":
            raise AssertionError(f"a PhenoBERT-only code read as {none_label}")

    report.check("agreement labels", _agreement_labels)

    # The sidebar's "unlabelled reports only" filter. It decides what a curator can reach with the
    # Prev/Next buttons, so a filter that drops the wrong report is a report nobody visits again,
    # and the case it has to get right is the one that happens every single time it is used: the
    # report leaving the list the moment its labels are saved.
    def _patient_filter():
        from apps.curation_ui.app import (
            render_patient_list, step_patient, unlabelled_only, visible_patient_ids,
        )

        if unlabelled_only([]) or unlabelled_only(None) or not unlabelled_only(["unlabelled"]):
            raise AssertionError("the filter value is not read as a checklist")

        # Its own log, for the same reason as the edit round trip: the persistence check above
        # leaves a torn final line behind, and an append after one of those is lost.
        filter_dir = os.path.join(tmp.name, "curation-filter")
        writable = Registry(paths, author="selftest", curation_dir=filter_dir)
        state.set_registry(writable)
        try:
            everyone = visible_patient_ids(writable, False)
            if everyone != writable.patient_ids:
                raise AssertionError("the unfiltered list is not the cohort")
            if visible_patient_ids(writable, True) != everyone:
                raise AssertionError("an uncurated cohort had a labelled report in it")

            subject = everyone[0]
            writable.record("label", subject, labels=["family_member"], difficulty="")
            after = visible_patient_ids(writable, True)
            if subject in after:
                raise AssertionError("a labelled report survived the filter")
            if [p for p in everyone if p != subject] != after:
                raise AssertionError(f"the filter dropped more than the labelled report: {after}")

            # A grade with no labels is a characterisation too.
            if len(everyone) > 1:
                graded = everyone[1]
                writable.record("label", graded, labels=[], difficulty="hard")
                if graded in visible_patient_ids(writable, True):
                    raise AssertionError("a graded report with no labels survived the filter")

            # Clearing both puts it back, the filter reads the state, it does not latch.
            writable.record("label", subject, labels=[], difficulty="")
            if subject not in visible_patient_ids(writable, True):
                raise AssertionError("unlabelling did not return the report to the queue")

            # Stepping. The interesting case is ``current`` no longer being in the list, which is
            # what saving labels for the report you are on does.
            order = ["A", "B", "C", "D"]
            visible = ["A", "C", "D"]
            cases = [
                ("A", visible, 1, "C"),      # next visible, skipping the labelled one
                ("C", visible, -1, "A"),     # and back
                ("A", visible, -1, "A"),     # clamped at the front
                ("D", visible, 1, "D"),      # clamped at the back
                ("B", visible, 1, "C"),      # current filtered out: the next one still to do
                ("B", visible, -1, "A"),     # …and the previous one
                ("D", ["A", "B"], 1, "D"),   # nothing left ahead, stay put, not go back
                ("A", [], 1, "A"),           # An empty list moves nobody
            ]
            for current, options, step, expect in cases:
                got = step_patient(current, options, order, step)
                if got != expect:
                    raise AssertionError(
                        f"step_patient({current!r}, {options}, {step}) → {got!r}, want {expect!r}")

            # And the rendered list: the filter must actually reach the screen.
            writable.record("label", subject, labels=["family_member"], difficulty="")
            drawn = render_patient_list(subject, "light", True)
            ids = {i["pid"] for i in _pattern_ids(drawn, "pat")}
            if subject in ids:
                raise AssertionError("the filtered list still drew the labelled report")
            if ids != set(visible_patient_ids(writable, True)):
                raise AssertionError(f"the list and the filter disagree: {sorted(ids)}")
            if render_patient_list(subject, "light", False) is None:
                raise AssertionError("the unfiltered list rendered None")
        finally:
            state.set_registry(registry)

    report.check("patient list filter", _patient_filter)

    # ── layer 3: export ──────────────────────────────────────────────────────
    def _export():
        from hpo_extraction.evaluation.datasets.hcy import HCYDataset

        result = registry.export_gold()
        if result["n_patients"] != len(patients):
            raise AssertionError(
                f"export wrote {result['n_patients']} patients, cohort has {len(patients)}")
        loaded = HCYDataset(result["path"], "").load_ground_truth()
        if set(loaded) != set(patients):
            raise AssertionError("the pipeline's own loader read a different patient set")

    report.check("curated ground truth export", _export)

    # ── layer 4: HTTP ────────────────────────────────────────────────────────
    def _http():
        from apps.curation_ui.app import build_app

        app = build_app(registry)
        client = app.server.test_client()
        for route in ("/", "/_dash-layout", "/_dash-dependencies"):
            response = client.get(route)
            if response.status_code != 200:
                raise AssertionError(f"{route} → HTTP {response.status_code}")

    report.check("http surface", _http)

    # A panel that failed inside ``common.guard`` renders an error box and returns normally, so it
    # would pass the render matrix. This is where that is caught.
    leftover = common.failures()
    if leftover:
        for name, tb in leftover.items():
            report.failures.append(f"guarded render failed: {name}\n{tb}")

    tmp.cleanup()

    if report.failures:
        logger.error("%d of %d check(s) FAILED", len(report.failures), report.checks)
        for failure in report.failures:
            logger.error("%s", failure)
        return 1
    logger.info("all %d check(s) passed", report.checks)
    return 0


def _assert_no_chips(result) -> None:
    if result is not None:
        raise AssertionError(f"an unqualified row drew a chip line: {result!r}")


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _strings(node, out=None) -> list:
    """Every string leaf of a component tree, in document order.

    Superscript code tags are skipped: they are chrome the reader draws on top of the report, not
    text the report contains, and counting them would make "is this verbatim?" unanswerable.
    """
    out = [] if out is None else out
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, list):
        for child in node:
            _strings(child, out)
    elif hasattr(node, "children"):
        if "mark-tag" not in (getattr(node, "className", "") or ""):
            _strings(node.children, out)
    return out


def _pattern_ids(node, kind: str, out=None) -> list:
    """Every pattern id of ``type == kind`` in a component tree."""
    out = [] if out is None else out
    if isinstance(node, list):
        for child in node:
            _pattern_ids(child, kind, out)
    elif hasattr(node, "children"):
        ident = getattr(node, "id", None)
        if isinstance(ident, dict) and ident.get("type") == kind:
            out.append(ident)
        _pattern_ids(node.children, kind, out)
    return out


def _mark_ids(node, out=None) -> list:
    """Every mark's pattern id in a component tree."""
    out = [] if out is None else out
    if isinstance(node, list):
        for child in node:
            _mark_ids(child, out)
    elif hasattr(node, "children"):
        ident = getattr(node, "id", None)
        if isinstance(ident, dict) and ident.get("type") == "mark":
            out.append(ident)
        _mark_ids(node.children, out)
    return out


def _assert_renders(result) -> None:
    if result is None:
        raise AssertionError("rendered None")


def _assert_text(text) -> None:
    if not isinstance(text, str) or not text.strip():
        raise AssertionError("produced no text")
