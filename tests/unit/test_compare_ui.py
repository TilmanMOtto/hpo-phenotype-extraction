"""What a render harness cannot assert: the arithmetic, and the contracts between the halves.

``python apps/compare_ui/app.py --selftest`` renders every panel in both themes and drives the
whole build against the fixture. It is the acceptance check and it covers the ground once. What is
left for pytest is what that harness structurally cannot see:

- **Placement is a coordinate system, not a lookup.** Five methods recorded their evidence in four
  different frames, and the bugs that live there are off-by-a-few-characters ones that render
  perfectly. So the span arithmetic is checked directly, at the boundaries where it is easiest to
  be silently wrong: a detection whose offsets disagree with its own phrase, one that runs past its
  segment, a phrase that occurs twice.
- **The bundle is a contract**, and ``validate`` is the only place it is written down as code. Each
  way of violating it is checked to actually raise, because a validator nobody tested is a
  validator that passes everything.
- **The tree re-run is a claim about another experiment's output.** That its accepted set equals
  ``nested_cv_pooled.csv`` is the single most important thing this app asserts, and it is asserted
  on the fixture where the answer was worked out by hand.
- **Degradation.** A missing artifact must grey out one column and leave the other four working.
- **The second cohort is a second ground truth, not a second path.** GSC+ under RAG-HPO's annotation
  carries no offsets, so where a term is drawn comes from a ladder whose rungs mean different
  things. Each rung is checked to fire on the case it is for and not on the next one down --
  a ladder that silently always reaches its bottom rung still renders a full page.

Run it as ``pytest tests/unit/test_compare_ui.py --no-cov``: ``pytest.ini`` carries
``--cov=src --cov-fail-under=70`` unconditionally, and app code lives outside ``src/``, so a
targeted run would otherwise trip the gate with every test passing.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (_REPO,):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from apps.compare_ui import bundles, fixture  # noqa: E402
from apps.compare_ui.methods import base  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    """The synthetic cohort, built once. Module-scoped: the ontology is the expensive part."""
    root = str(tmp_path_factory.mktemp("compare_ui"))
    written = fixture.write(root)
    ctx, built = fixture.build(written)
    skipped = built.pop("_skipped")
    return {"root": root, "written": written, "ctx": ctx, "built": built, "skipped": skipped,
            "codes": fixture._codes()}


@pytest.fixture(scope="module")
def bundle(cohort):
    return cohort["built"]["SYN001"]


@pytest.fixture(scope="module")
def gsc(tmp_path_factory):
    """The GSC+ cohort, built once. Two abstracts. See ``fixture.write_gsc``."""
    root = str(tmp_path_factory.mktemp("compare_ui_gsc"))
    written = fixture.write_gsc(root)
    ctx, built = fixture.build_gsc(written)
    skipped = built.pop("_skipped")
    return {"root": root, "written": written, "ctx": ctx, "built": built,
            "skipped": skipped, "codes": fixture._codes()}


# ── the coordinate system ────────────────────────────────────────────────────

class TestPlacement:
    """Offsets, spans and phrases -- the arithmetic behind every underline on screen."""

    def test_the_drawn_span_spells_the_trigger_word(self, bundle):
        """The one invariant the whole ground truth column rests on.

        A span that is off by three characters underlines the wrong word and looks fine. Nothing
        downstream can catch that, so it is checked here against the segment text itself.
        """
        for row in bundle["gold"]:
            if row["span"] is None:
                continue
            text = bundle["segments"][row["segment_idx"]]["text"]
            assert text[row["span"][0]:row["span"][1]] == row["trigger"]

    def test_every_span_indexes_its_own_segment(self, bundle):
        for key, block in bundle["methods"].items():
            for mark in block["marks"]:
                if mark["span"] is None:
                    continue
                length = len(bundle["segments"][mark["segment_idx"]]["text"])
                assert 0 <= mark["span"][0] < mark["span"][1] <= length, (key, mark)

    def test_offsets_that_contradict_their_phrase_fall_back_to_the_phrase(self, cohort):
        """PhenoBERT's offsets index its own staged file. When the two disagree, the phrase wins.

        A disagreement means the offsets were measured against a different staging of this report,
        and trusting them would underline characters nobody detected.
        """
        from apps.compare_ui.methods import phenobert

        rv = cohort["ctx"].report_view("SYN001")
        row = {"hpo_id": "HP:0001250", "phrase": "Anfälle", "start": 0, "end": 7}
        where = phenobert._locate(rv, row)
        assert where["located_by"] == "phrase"
        assert rv.display[where["segment_idx"]][where["span"][0]:where["span"][1]] == "Anfälle"

    def test_a_span_running_past_its_segment_is_clamped_not_dropped(self, cohort):
        """A phrase can straddle a boundary the tokenizer drew. Its head is still in one segment."""
        rv = cohort["ctx"].report_view("SYN001")
        start = rv.ranges[1][0] + 3
        idx, local_start, local_end = rv.local(start, rv.ranges[1][1] + 500)
        assert idx == 1
        assert local_end <= len(rv.display[1])
        assert local_start < local_end

    def test_an_offset_in_no_segment_is_unplaceable(self, cohort):
        rv = cohort["ctx"].report_view("SYN001")
        assert rv.local(10 ** 6, 10 ** 6 + 5) is None

    def test_locate_phrase_prefers_the_segment_it_was_told_about(self, cohort):
        """With no offset recorded, a repeated phrase has several equally good homes.

        RAG-HPO is the case: it names the sentence and no character range, so the caller narrows
        the search and the first hit inside that sentence wins.
        """
        rv = cohort["ctx"].report_view("SYN001")
        first, _ = base.locate_phrase(rv, "e")
        preferred, _ = base.locate_phrase(rv, "e", prefer_segment=2)
        assert first == 0
        assert preferred == 2

    def test_the_segments_carry_the_text_between_them(self, bundle):
        """Sentence tokenization does not partition a report. The gaps are content, not noise."""
        gaps = [s["gap_before"] for s in bundle["segments"]]
        assert any(gap.strip() == "" and gap for gap in gaps), gaps


# ── the grid ─────────────────────────────────────────────────────────────────

class TestGridStructure:
    """The shape of the reading surface, which "it rendered something" cannot see.

    The first version made the container a grid *and* every row a grid inside it. Because a grid
    container lays each **child** into one of its own columns, seven consecutive rows were placed
    into the seven columns of a single grid row, stacked on top of each other and unreadable. Every
    render check passed: the panel was a ``Div``, it was non-empty, no callback raised. So the
    structure is asserted directly.
    """

    def _grid(self, registry, bundle):
        from apps.compare_ui.views import reader

        rendered = reader.render_grid(registry, bundle)
        grid = rendered.children[0]
        assert "cmp-grid" in grid.className
        return grid

    @pytest.fixture
    def registry(self, cohort, tmp_path_factory):
        from apps.compare_ui import build as build_mod
        from apps.compare_ui.registry import Registry

        out = str(tmp_path_factory.mktemp("bundles"))
        for one in cohort["built"].values():
            bundles.write(out, one)
        bundles.write_index(out, build_mod.make_index(
            cohort["ctx"], dict(cohort["built"], _skipped=cohort["skipped"])))
        return Registry(out, cohort["written"]["output_base"])

    def test_the_grid_is_flat(self, registry, bundle):
        """Every direct child is a cell. A child that is itself a row is the bug."""
        grid = self._grid(registry, bundle)
        for cell in grid.children:
            assert "cmp-cell" in (cell.className or ""), cell.className
            assert "cmp-grid" not in (cell.className or "")

    def test_the_cells_divide_into_whole_rows(self, registry, bundle):
        """n_columns cells per row, or the columns do not line up with each other."""
        grid = self._grid(registry, bundle)
        n_columns = 2 + len(registry.methods())
        body = [c for c in grid.children if "cmp-span-all" not in (c.className or "")]
        assert len(body) % n_columns == 0, (len(body), n_columns)

    def test_the_template_declares_one_track_per_column(self, registry, bundle):
        grid = self._grid(registry, bundle)
        template = grid.style["gridTemplateColumns"]
        # minmax(...) carries its own spaces, so count the tracks rather than the words.
        tracks = template.count("minmax(") + len(
            [t for t in template.split() if t.endswith("rem")])
        assert tracks == 2 + len(registry.methods()), template

    def test_the_header_is_one_sticky_cell_per_column(self, registry, bundle):
        """There are no rows to make sticky, so each header cell carries it."""
        grid = self._grid(registry, bundle)
        n_columns = 2 + len(registry.methods())
        head = grid.children[:n_columns]
        assert all("cmp-head" in c.className for c in head)
        assert "cmp-head" not in (grid.children[n_columns].className or "")

    def test_a_full_width_cell_spans_every_column(self, registry, bundle):
        """The trailing text belongs to no column. It must say so, not land in column 1."""
        from apps.compare_ui.views import reader

        spanning = [c for c in self._grid(registry, bundle).children
                    if "cmp-span-all" in (c.className or "")]
        assert len(spanning) <= 1
        assert reader._columns(5).count("minmax(") == 2


# ── the bundle contract ──────────────────────────────────────────────────────

class TestSchema:
    """``validate`` is the only written-down form of the shape every view assumes."""

    def test_the_fixture_bundle_validates(self, bundle):
        bundles.validate(bundle)

    @pytest.mark.parametrize("mutate,expected", [
        (lambda b: b.pop("terms"), "missing"),
        (lambda b: b.__setitem__("schema", 99), "schema"),
        (lambda b: b["segments"][1].__setitem__("idx", 7), "position"),
        (lambda b: b["gold"][0].__setitem__("span", [0, 10 ** 6]), "spans"),
        (lambda b: b["gold"][0].__setitem__("hpo_id", "HP:9999999"), "no entry in terms"),
    ])
    def test_a_broken_bundle_is_refused(self, bundle, mutate, expected):
        broken = json.loads(json.dumps(bundle))
        mutate(broken)
        with pytest.raises((ValueError, TypeError), match=expected):
            bundles.validate(broken)

    def test_an_outcome_outside_the_vocabulary_is_refused(self, bundle):
        broken = json.loads(json.dumps(bundle))
        broken["methods"]["phenobert"]["marks"][0]["outcome"] = "maybe"
        with pytest.raises(ValueError, match="outcome"):
            bundles.validate(broken)

    def test_the_size_cap_is_enforced_on_write(self, bundle, tmp_path):
        big = json.loads(json.dumps(bundle))
        big["methods"]["treephenorag"]["reasons"]["bloat"] = {"x": "y" * bundles.MAX_BUNDLE_BYTES}
        with pytest.raises(ValueError, match="over the"):
            bundles.write(str(tmp_path), big)

    def test_a_bundle_round_trips(self, bundle, tmp_path):
        bundles.write(str(tmp_path), bundle)
        assert bundles.read(str(tmp_path), "SYN001") == bundle

    def test_a_bundle_from_another_schema_is_not_read(self, bundle, tmp_path):
        older = json.loads(json.dumps(bundle))
        older["schema"] = bundles.SCHEMA_VERSION - 1
        path = bundles.bundle_path(tmp_path, "SYN001")
        path.write_text(json.dumps(older), encoding="utf-8")
        assert bundles.read(str(tmp_path), "SYN001") is None

    def test_drift_is_silent_about_files_it_cannot_reach(self, bundle):
        """Absence of evidence is not evidence of change. Otherwise the banner is always on."""
        assert bundles.drifted(bundle, "/definitely/not/here") == []

    def test_drift_is_reported_when_an_input_really_moves(self, cohort, bundle):
        base_dir = cohort["written"]["output_base"]
        record = bundle["inputs"]["phenobert_predictions"]
        path = os.path.join(base_dir, record["path"])
        original = open(path, "rb").read()
        try:
            with open(path, "ab") as handle:
                handle.write(b'{"report_id": "SYN999", "summary": true, "predicted_set": []}\n')
            assert "phenobert_predictions" in bundles.drifted(bundle, base_dir)
        finally:
            with open(path, "wb") as handle:
                handle.write(original)


# ── membership ───────────────────────────────────────────────────────────────

class TestOutcomes:
    """What counts as a true positive, decided once in ``build`` so five adapters cannot differ."""

    def test_an_alt_id_is_the_same_term(self, bundle, cohort):
        """A ground truth file predates the release it is scored against. Raw strings read that as a miss."""
        marks = {m["hpo_id"]: m["outcome"] for m in bundle["methods"]["raghpo_70b"]["marks"]}
        assert marks[cohort["codes"]["C"]] == "tp"
        assert cohort["codes"]["C_ALT"] not in marks, "the alt id should not appear as its own term"

    def test_every_annotated_term_gets_an_outcome_under_every_method(self, bundle):
        gold = {row["hpo_id"] for row in bundle["gold"]}
        for key, block in bundle["methods"].items():
            assert gold <= {m["hpo_id"] for m in block["marks"]}, key

    def test_a_report_with_no_gold_gets_no_rate(self):
        """Zero would be a claim about a report on which nothing could be scored."""
        from apps.compare_ui import build as build_mod

        score = build_mod._score({"HP:1": "fp"})
        assert score["recall"] is None and score["f1"] is None
        assert score["precision"] == 0.0

    def test_counts_and_rates_agree(self, bundle):
        for key, score in bundle["metrics"].items():
            if score["tp"] + score["fp"]:
                assert score["precision"] == pytest.approx(
                    score["tp"] / (score["tp"] + score["fp"])), key
            if score["tp"] + score["fn"]:
                assert score["recall"] == pytest.approx(
                    score["tp"] / (score["tp"] + score["fn"])), key


# ── the predictions contract ─────────────────────────────────────────────────

class TestPredictionSets:
    """The summary line is authoritative. The per-term lines are not."""

    def test_the_summary_line_wins_over_the_rows(self, tmp_path):
        """A method that post-filters would otherwise be credited with what it filtered out."""
        path = tmp_path / "p.jsonl"
        path.write_text(
            json.dumps({"report_id": "r1", "hpo_id": "HP:1", "prediction": 1}) + "\n"
            + json.dumps({"report_id": "r1", "summary": True, "predicted_set": []}) + "\n",
            encoding="utf-8")
        assert base.load_prediction_sets(str(path)) == {"r1": set()}

    def test_a_report_predicted_empty_survives(self, tmp_path):
        """Dropping it would hand the method free precision on every report it said nothing about."""
        path = tmp_path / "p.jsonl"
        path.write_text(
            json.dumps({"report_id": "r1", "summary": True, "predicted_set": []}) + "\n",
            encoding="utf-8")
        assert "r1" in base.load_prediction_sets(str(path))

    def test_both_id_spellings_are_accepted(self, tmp_path):
        """Detections files say report_id. Extraction files say patient_id."""
        path = tmp_path / "x.jsonl"
        path.write_text(
            json.dumps({"patient_id": "r1", "v": 1}) + "\n"
            + json.dumps({"report_id": "r2", "v": 2}) + "\n", encoding="utf-8")
        assert set(base.group_jsonl(str(path))) == {"r1", "r2"}

    def test_an_undecodable_line_costs_one_line(self, tmp_path):
        path = tmp_path / "x.jsonl"
        path.write_text('{"report_id": "r1"}\nnot json\n{"report_id": "r2"}\n', encoding="utf-8")
        assert set(base.group_jsonl(str(path))) == {"r1", "r2"}


# ── TreePhenoRAG ─────────────────────────────────────────────────────────────

class TestStoredScores:
    """The claim this column makes about the TreePhenoRAG protocol's output, checked, not asserted in prose."""

    def test_the_offline_evaluation_reproduces_the_pooled_prediction_set(self, bundle):
        """If these disagree, every number on the TreePhenoRAG column is explaining another run."""
        block = bundle["methods"]["treephenorag"]
        accepted = {code for code, reason in block["reasons"].items() if reason.get("accepted")}
        assert accepted == set(block["predicted"])

    def test_the_fold_configuration_is_repetition_zero(self, bundle):
        """Only repetition 0 pools, so only its choices explain a prediction.

        The fixture writes a second repetition with a different configuration. Picking
        it up would change every threshold on screen.
        """
        config = bundle["methods"]["treephenorag"]["reasons"][
            list(bundle["methods"]["treephenorag"]["reasons"])[0]]["config"]
        assert config["pool_pr"] == "P1" and config["pool_acc"] == "lse_beta1"
        assert config["tau_prune"] == pytest.approx(fixture.TAU_PRUNE)
        assert config["tau_accept"] == pytest.approx(fixture.TAU_ACCEPT)

    def test_a_pruned_term_names_the_ancestor_that_blocked_it(self, bundle, cohort):
        reason = bundle["methods"]["treephenorag"]["reasons"][cohort["codes"]["F"]]
        assert reason["pruned"] is True
        assert reason["lost_at"] == cohort["codes"]["E"]
        assert reason["bucket"] == "pruning"
        assert reason["visited"] is False

    def test_a_scored_but_rejected_term_is_filed_under_pooling(self, bundle, cohort):
        """The two misses want opposite fixes, and the bucket is what distinguishes them.

        ``D`` is reached, its best verifier margin is positive, and its pooled acceptance score
        falls short of ``tau_accept`` -- which is the *pooling* bucket. It lands in ``residual``
        instead if ``_cause_at`` is handed the pruning score and ``tau_prune``: a visited node's
        pruning score always clears ``tau_prune`` by definition, so that mistake empties the
        pooling bucket into a word meaning "we do not know". It is invisible on the fixture's
        counts and it silently emptied 121 of 167 real misses on the first cluster run.
        """
        reason = bundle["methods"]["treephenorag"]["reasons"][cohort["codes"]["D"]]
        assert reason["visited"] is True
        assert reason["accepted"] is False
        assert reason["pruned"] is False
        assert reason["bucket"] == "pooling"

    def test_the_curated_segment_is_used_as_retrieval_evidence(self, bundle, cohort):
        """Without it, "retrieval" can only mean "nothing was scored at all"."""
        reasons = bundle["methods"]["treephenorag"]["reasons"]
        assert reasons[cohort["codes"]["D"]]["evidence_available"] is True
        # ``F`` is pruned, so the attribution point is an ancestor and the ground truth says nothing
        # about where an ancestor should have been found.
        assert reasons[cohort["codes"]["F"]]["evidence_available"] is False

    def test_the_calls_carry_the_sentence_they_judged(self, bundle, cohort):
        calls = bundle["methods"]["treephenorag"]["reasons"][cohort["codes"]["C"]]["calls"]
        assert calls
        for call in calls:
            assert call["sentence"] == bundle["segments"][call["sent_index"]]["text"]
            assert call["verdict"] == ("Yes" if call["margin"] > 0 else "No")

    def test_the_calls_are_capped(self, bundle, cohort):
        from apps.compare_ui.methods import tree

        for reason in bundle["methods"]["treephenorag"]["reasons"].values():
            assert len(reason.get("calls") or ()) <= tree.MAX_CALLS


# ── PhenoJury ────────────────────────────────────────────────────────────────

class TestJury:
    def test_the_vote_uses_the_folds_own_configuration(self, bundle, cohort):
        reason = bundle["methods"]["phenojury"]["reasons"][cohort["codes"]["C"]]
        assert reason["k"] == 2 and reason["unit"] == "segment" and reason["rule"] == "exact"
        assert reason["outer_fold"] == 0
        assert reason["config_known"] is True

    def test_the_vote_is_recomputed_by_the_protocol_itself(self, bundle, cohort):
        """The unit semantics are the method. A second copy of them would be wrong eventually."""
        reason = bundle["methods"]["phenojury"]["reasons"][cohort["codes"]["C"]]
        assert reason["vote_recomputed"] is True
        assert reason["votes"] == 3

    def test_only_the_selected_subset_votes(self, bundle, cohort):
        reason = bundle["methods"]["phenojury"]["reasons"][cohort["codes"]["C"]]
        voted = {j["model"] for j in reason["jurors"] if j["voted"]}
        assert voted <= set(reason["subset"])

    def test_generations_are_stored_once_per_model_and_sentence(self, bundle):
        """One sentence explains several terms. Duplicating it per term is how a bundle bloats."""
        generations = bundle["methods"]["phenojury"]["generations"]
        assert generations
        for key, record in generations.items():
            assert key == "{}|{}".format(record["model"], record["sentence_number"])

    def test_every_referenced_generation_exists(self, bundle):
        generations = bundle["methods"]["phenojury"]["generations"]
        for reason in bundle["methods"]["phenojury"]["reasons"].values():
            for model, sent in reason.get("refs") or ():
                assert "{}|{}".format(model, sent) in generations


# ── the other three columns ──────────────────────────────────────────────────

class TestExternalMethods:
    def test_phenobert_separates_its_three_ways_of_losing_a_term(self, bundle, cohort):
        reasons = bundle["methods"]["phenobert"]["reasons"]
        assert "negated" in reasons[cohort["codes"]["D"]]["why"]
        assert "never detected" in reasons[cohort["codes"]["F"]]["why"]

    def test_autopcr_keeps_the_linking_route(self, bundle, cohort):
        """Which of dictionary / retrieval / LLM fired is the method's whole contribution."""
        mentions = bundle["methods"]["autopcr_70b"]["reasons"][
            cohort["codes"]["D"]]["mentions"]
        assert [m["route"] for m in mentions] == ["llm"]
        assert mentions[0]["menu"], "the candidate menu the linker chose from"

    def test_autopcr_and_raghpo_declare_that_the_answer_was_not_recorded(self, bundle, cohort):
        """Both called a model. Neither driver wrote the answer. The panel must not imply it did."""
        for key in ("autopcr_70b", "raghpo_70b"):
            reason = bundle["methods"][key]["reasons"][cohort["codes"]["D"]]
            assert reason["answer_recorded"] is False
            assert reason["answer_note"]

    def test_a_losing_candidate_is_not_reported_as_never_retrieved(self, bundle, cohort):
        """A linking failure and a retrieval failure want opposite fixes."""
        why = bundle["methods"]["autopcr_70b"]["reasons"][cohort["codes"]["F"]]["why"]
        assert "candidate" in why and "never" not in why

    def test_raghpo_marks_the_candidate_it_took(self, bundle, cohort):
        selections = bundle["methods"]["raghpo_70b"]["reasons"][
            cohort["codes"]["D"]]["selections"]
        assert selections
        assert any(c["chosen"] for c in selections[0]["menu"])


# ── degradation ──────────────────────────────────────────────────────────────

class TestDegradation:
    def test_a_report_whose_segmentation_does_not_align_is_skipped_with_a_reason(self, cohort):
        """Drawing it would attribute real decisions to the wrong sentences."""
        assert "SYN003" in cohort["skipped"]
        assert "segmentation" in cohort["skipped"]["SYN003"]

    def test_a_missing_artifact_greys_out_one_column_only(self, cohort, tmp_path):
        from apps.compare_ui import build as build_mod, sources

        paths = fixture.paths_for(cohort["written"])
        missing = sources.Paths(output_base=str(tmp_path / "empty"), hcy_dir=paths.hcy_dir,
                                gold_dir=paths.gold_dir, frame_dir=paths.frame_dir)
        ctx = sources.build_context(missing, view=cohort["ctx"].view,
                                    gold=cohort["ctx"].gold)
        ctx.texts, ctx.segments = cohort["ctx"].texts, cohort["ctx"].segments
        built = build_mod.build_all(ctx, ["SYN001"])
        built.pop("_skipped")
        blocks = built["SYN001"]["methods"]
        assert {b["status"] for b in blocks.values()} == {"missing"}
        for block in blocks.values():
            assert block["note"], "a greyed-out column must say which file it wanted"
        # Ground truth still draws: the report is readable even when every method is absent.
        assert built["SYN001"]["gold"]

    def test_the_registry_says_what_to_do_when_there_is_nothing_to_serve(self, tmp_path):
        from apps.compare_ui.registry import Registry

        registry = Registry(str(tmp_path / "nope"))
        assert not registry.ok
        assert "slurm/compare_ui_bundles.sbatch" in registry.problem()


# ── the roster ───────────────────────────────────────────────────────────────

class TestRoster:
    def test_every_roster_key_has_an_adapter(self):
        from apps.compare_ui import roster
        from apps.compare_ui.methods import ADAPTERS

        assert set(ADAPTERS) == {m.key for m in roster.ROSTER}

    def test_the_two_thin_columns_carry_their_note(self):
        """It is a property of the artifacts, and a reader must not read it as a property of the
        methods."""
        from apps.compare_ui import roster

        for key in ("autopcr_70b", "raghpo_70b"):
            assert roster.get(key).evidence_note

    def test_every_outcome_has_a_tone_and_a_glyph(self):
        """Colour alone is not a channel everyone has."""
        from apps.compare_ui import theme

        for outcome in bundles.OUTCOMES:
            assert outcome in theme.OUTCOME_TONE
            assert outcome in theme.OUTCOME_GLYPH
            assert outcome in theme.OUTCOME_HELP


# ── the second cohort ────────────────────────────────────────────────────────

class TestGscGold:
    """RAG-HPO's annotation has no positions, so placement is a ladder with three rungs.

    The rungs are different claims and the app says which one it used, so each has to fire on its
    own case. The failure mode being guarded is a ladder that quietly always falls to the bottom:
    every term would then sit in the footer, the page would render, and the ground truth column would say
    nothing while looking as intended.
    """

    def test_a_corpus_annotated_term_is_drawn_on_the_corpus_offsets(self, gsc):
        row = _gold_row(gsc, "GSC001", gsc["codes"]["C"])
        assert row["source"] == "gsc+ annotation"
        segment = gsc["built"]["GSC001"]["segments"][row["segment_idx"]]["text"]
        assert segment[row["span"][0]:row["span"][1]] == "seizures"

    def test_offsets_that_do_not_spell_their_mention_fall_back_to_locating_it(self, gsc):
        """A mention recorded against a differently staged corpus lands a few characters off.

        Trusting it underlines three arbitrary characters and reads as a tokenizer quirk, which is
        why the offsets are checked against the mention they claim before they are used.
        """
        row = _gold_row(gsc, "GSC002", gsc["codes"]["C"])
        assert row["source"].startswith("gsc+ annotation (located")
        segment = gsc["built"]["GSC002"]["segments"][row["segment_idx"]]["text"]
        assert segment[row["span"][0]:row["span"][1]] == "seizures"

    def test_a_raghpo_only_term_falls_back_to_its_own_description(self, gsc):
        row = _gold_row(gsc, "GSC001", gsc["codes"]["D"])
        assert row["source"] == "raghpo description match"
        segment = gsc["built"]["GSC001"]["segments"][row["segment_idx"]]["text"]
        assert segment[row["span"][0]:row["span"][1]] == row["trigger"]

    def test_and_is_flagged_as_one_the_corpus_does_not_annotate(self, gsc):
        """Roughly 11% of RAG-HPO's pairs are terms GSC+ does not carry on that document.

        That is the single most useful thing to know about a term every method missed, so it is a
        qualifier chip, not a footnote.
        """
        from apps.compare_ui import gold as gold_mod

        row = _gold_row(gsc, "GSC001", gsc["codes"]["D"])
        assert gold_mod.RAGHPO_ONLY in row["qualifiers"]
        assert gold_mod.RAGHPO_ONLY not in _gold_row(gsc, "GSC001", gsc["codes"]["C"])["qualifiers"]

    def test_a_term_nobody_located_is_unplaced_rather_than_guessed(self, gsc):
        row = _gold_row(gsc, "GSC001", gsc["codes"]["F"])
        assert row["segment_idx"] is None and row["span"] is None

    def test_the_gold_is_raghpos_annotation_and_not_the_corpus(self, gsc):
        """``M`` is annotated by the corpus on GSC001 and is not in RAG-HPO's ground truth.

        The cohort being scored is ``gsc_raghpo_ann``, whose membership is RAG-HPO's file and
        nobody else's. The corpus annotation is read for *placement* only. Mixing it into the ground truth
        would score a different cohort under this one's name -- and would hide the case the deep
        dive is for, a term one annotation carries and the other does not.
        """
        codes = gsc["codes"]
        drawn = {row["hpo_id"] for row in gsc["built"]["GSC001"]["gold"]}
        assert codes["M"] not in drawn
        marks = gsc["built"]["GSC001"]["methods"]["autopcr_70b"]["marks"]
        outcome = next(m["outcome"] for m in marks if m["hpo_id"] == codes["M"])
        assert outcome == "fp"


class TestGscCohortWiring:
    """What the bundle has to say about itself for the app to hold no cohort table."""

    def test_the_bundle_names_its_cohort_and_the_row_it_is_scored_as(self, gsc):
        from apps.compare_ui import build as build_mod

        assert gsc["built"]["GSC001"]["cohort"] == "gsc"
        index = build_mod.make_index(gsc["ctx"], dict(gsc["built"], _skipped={}))
        assert index["scored_cohort"] == "gsc_raghpo_ann"
        assert index["cohort_unit"] == "abstract"

    def test_the_corpus_sidecars_are_not_documents(self, gsc):
        """``*:Zone.Identifier`` twins double the corpus silently on a Windows checkout."""
        assert sorted(gsc["ctx"].texts) == ["GSC001", "GSC002"]

    def test_the_transfer_column_keeps_its_verdict_and_claims_no_offline_evaluation(self, gsc):
        """On GSC+ nothing is fitted, so there is no fold threshold to read a score against.

        The column must still draw -- the prediction set exists -- and must not present the
        absence of a re-run as the method having found nothing.
        """
        tree = gsc["built"]["GSC001"]["methods"]["treephenorag"]
        assert tree["status"] == "ok"
        assert set(tree["predicted"]) == {gsc["codes"]["C"]}
        assert "not folded" in tree["note"]
        assert not any(reason.get("available", False) for reason in tree["reasons"].values())

    def test_the_adapters_read_this_cohorts_artifact_directory(self, gsc):
        """Every ``fixed`` driver writes into ``{exp_id}/{cohort}/``, and the adapters must follow.

        Checked on the recorded inputs, not on a path built here: a bundle names every file
        it was built from, so a column silently reading the other cohort's artifacts shows up as a
        path with the wrong directory in it.
        """
        inputs = gsc["built"]["GSC001"]["inputs"]
        assert inputs, "the bundle records no inputs at all"
        cohorted = [record["path"] for name, record in inputs.items()
                    if name.endswith(("_predictions", "_detections", "_retrieved_segments"))]
        assert cohorted
        for path in cohorted:
            assert "hcy" not in path.replace(os.sep, "/").split("/"), path


class TestCohortDiscovery:
    """The dataset picker is a map of registries, and the keys come from the builds themselves."""

    def test_a_parent_directory_yields_one_registry_per_cohort(self, cohort, gsc, tmp_path):
        from apps.compare_ui import build as build_mod, registry as registry_mod

        root = str(tmp_path / "compare_ui_bundles")
        for key, built, ctx in (("hcy", cohort["built"], cohort["ctx"]),
                                ("gsc", gsc["built"], gsc["ctx"])):
            out = os.path.join(root, key)
            for one in built.values():
                bundles.write(out, one)
            bundles.write_index(out, build_mod.make_index(ctx, dict(built, _skipped={})))

        found = registry_mod.discover(root)
        assert registry_mod.order(found) == ["hcy", "gsc"]
        assert sorted(found["gsc"].report_ids()) == ["GSC001", "GSC002"]
        # Pointing at one cohort's own directory still works -- that is what a debugging build
        # writes, and what the single-cohort invocation used to produce.
        assert list(registry_mod.discover(os.path.join(root, "gsc"))) == ["gsc"]

    def test_a_registry_matches_the_comparison_row_on_its_scored_cohort(self, gsc, tmp_path):
        """``gsc`` artifacts are scored as ``gsc_raghpo_ann``. Matching the wrong one prints the
        228-document corpus's numbers under a 114-document heading."""
        import csv

        from apps.compare_ui import build as build_mod, sources
        from apps.compare_ui.registry import Registry

        out = str(tmp_path / "bundles")
        for one in gsc["built"].values():
            bundles.write(out, one)
        bundles.write_index(out, build_mod.make_index(gsc["ctx"], dict(gsc["built"], _skipped={})))

        tables = tmp_path / "base" / sources.COMPARISON_EXP / "tables"
        tables.mkdir(parents=True)
        with open(tables / "t1_overall.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["cohort", "method", "micro_f1",
                                                        "n_reports"])
            writer.writeheader()
            writer.writerow({"cohort": "gsc_raghpo_ann", "method": "phenojury",
                             "micro_f1": "0.71", "n_reports": "114"})
            writer.writerow({"cohort": "hcy", "method": "phenojury", "micro_f1": "0.63",
                             "n_reports": "118"})

        rows = Registry(out, str(tmp_path / "base")).comparison_rows()
        assert [r["cohort"] for r in rows] == ["gsc_raghpo_ann"]


def _gold_row(gsc, report_id, hpo_id):
    rows = {row["hpo_id"]: row for row in gsc["built"][report_id]["gold"]}
    assert hpo_id in rows, "{} is not in {}'s gold".format(hpo_id, report_id)
    return rows[hpo_id]


# ── the cell rules the GSC+ frame is drawn on ────────────────────────────────

class TestGscCellRules:
    """``apps/compare_ui/select_gsc_documents.py`` decides which abstracts reach the app.

    Its own ``--selftest`` covers the draw. What is worth a second check here is the agreement
    between the cells and the **bundles**: a cell says "PhenoJury found this and the others did
    not", and the app then draws ✓/– from the same prediction sets. If the two disagreed the frame
    would put an abstract on screen to illustrate a pattern it does not show.
    """

    def test_the_cells_agree_with_the_outcomes_the_builder_computed(self, gsc):
        import sys as _sys

        _sys.path.insert(0, os.path.join(_REPO, "apps", "compare_ui"))
        import select_gsc_documents as cells

        for report_id, bundle_ in gsc["built"].items():
            marks = {key: {m["hpo_id"]: m["outcome"] for m in block["marks"]}
                     for key, block in bundle_["methods"].items()}
            for row in bundle_["gold"]:
                hpo_id = row["hpo_id"]
                found = {short: marks[key].get(hpo_id) == "tp" for short, key in cells.METHODS}
                cell = next(iter(cells.cell_of_term(found)))
                assert cell in cells.CELL_SIZES, cell
                # The fixture was written so GSC001 carries one term of each of three cells and
                # GSC002 the fourth. Anything else means the two sides have drifted apart.
                if hpo_id == gsc["codes"]["C"] and report_id == "GSC001":
                    assert cell == "all_right"
                if hpo_id == gsc["codes"]["D"]:
                    assert cell == "pj_only"
                if hpo_id == gsc["codes"]["F"]:
                    assert cell == "all_wrong"
                if report_id == "GSC002":
                    assert cell == "pj_wrong"
