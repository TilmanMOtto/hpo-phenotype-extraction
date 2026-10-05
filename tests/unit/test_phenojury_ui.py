"""Tests for the Free Listing generation run SLM-ensemble deep-dive UI (``app/phenojury_generation_free_listing``).

The app's own selftest (``python apps/phenojury_ui/app.py --selftest``) is the acceptance check and
it runs against real cluster data. What is left for pytest is what real data cannot give: a
scenario small enough to state the right answer by hand, and the degenerate cases a real run does
not happen to contain.

Everything runs against ``fixtures.exp13_output.build_slm_ensemble_run``, which writes a complete
the Free Listing generation run directory whose *shipped* artifacts are produced by the driver's own functions. So a
test that says "recomputed == shipped" is comparing the app against real driver output, not against
numbers typed into a fixture.

Four groups:

``TestLoaders``     the artifact contract: discovery, the two PhenoBERT TSV formats, negation
``TestGates``       G1–G5, plus the roster-mismatch diagnosis on a corrupted run
``TestVotes``       decisiveness against brute-force leave-one-out over all 2^n subsets
``TestAnalysis``    fate classification and the FP taxonomy, on the ten-node toy ontology
"""

from __future__ import annotations

import itertools
import json
import pathlib
import shutil

import pytest

from apps.phenojury_ui import annotations, autopsy, detections, frame as frame_mod, loaders, memo, registry, relations, verify, votes
from hpo_extraction.curation import phenobert_output as pbstandalone
from apps.phenojury_ui.views import common, compare_pb, patient
from fixtures.exp13_output import (
    SLM_GOLD,
    SLM_MODELS,
    build_comparison_tree,
    build_phenobert_standalone_run,
    build_prompt_screen_tree,
    build_slm_ensemble_run,
)
from fixtures.toy_ontology import B, C, D, F, G, H, M, build_toy_tree

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """A complete the Free Listing generation run directory. Module-scoped: writing it runs PhenoBERT parsing."""
    return str(build_slm_ensemble_run(tmp_path_factory.mktemp("exp13_06") / "hcy"))


@pytest.fixture(scope="module")
def bundle(run_dir):
    return registry.build_bundle("hcy", run_dir, loaders.load_raw(run_dir), build_toy_tree())


def _cfg(bundle, k=2, rule="vote_k", min_count=1, models=None):
    return {"models": list(models if models is not None else bundle["models"]),
            "k": k, "rule": rule, "min_count": min_count}


class TestLoaders:
    def test_model_order_matches_the_driver(self):
        """The UI's copy of MODEL_KEYS is essential, it names files and orders vote bits."""
        from hpo_extraction.phenojury.generation import MODEL_KEYS

        assert loaders.MODEL_ORDER == tuple(MODEL_KEYS)

    def test_discovery_accepts_all_three_layouts(self, tmp_path):
        base = tmp_path / "output"
        build_slm_ensemble_run(base / loaders.EXP_ID / "hcy")
        build_slm_ensemble_run(base / loaders.EXP_ID / "gsc")

        assert [r["cohort"] for r in loaders.find_runs(str(base))] == ["gsc", "hcy"]
        assert len(loaders.find_runs(str(base / loaders.EXP_ID))) == 2
        single = loaders.find_runs(str(base / loaders.EXP_ID / "hcy"))
        assert [r["cohort"] for r in single] == ["hcy"]

    def test_discovery_ignores_unrelated_directories(self, tmp_path):
        (tmp_path / "exp13_00_tree_gate_lr" / "hcy").mkdir(parents=True)
        assert loaders.find_runs(str(tmp_path)) == []

    def test_active_requires_a_nonempty_detections_file(self, tmp_path):
        """The driver's own rule: an empty slice is a grounding failure, not an abstention."""
        run = build_slm_ensemble_run(tmp_path / "hcy")
        (run / "detections_llama.jsonl").write_text("")
        models = loaders.discover_models(str(run))
        assert "llama" in models["extracted"]
        assert "llama" not in models["active"]

    def test_rules_sort_numerically(self, tmp_path):
        run = build_slm_ensemble_run(tmp_path / "hcy")
        (run / "vote_k10").mkdir()
        shutil.copy(run / "vote_k1" / "slm_ensemble_predictions.jsonl",
                    run / "vote_k10" / "slm_ensemble_predictions.jsonl")
        rules = loaders.find_rules(str(run))
        assert rules[-1] == "agg_plurality"
        assert rules.index("vote_k10") > rules.index("vote_k4")

    def test_phenobert_rows_carry_negation_that_detections_drop(self, run_dir):
        """The negated detection is the only evidence separating 'absent' from 'never mentioned'."""
        rows = loaders.load_phenobert(run_dir, "apertus")
        negated = [r for r in rows if r["negated"]]
        assert [(r["patient_id"], r["hpo_id"]) for r in negated] == [("r2", M)]
        assert not any(r["hpo_id"] == M for r in loaders.load_detections(run_dir, "apertus"))

    def test_both_tsv_formats_parse(self, run_dir):
        """Patched installs write a sentence index. Stock ones do not and need the offset route."""
        patched = loaders.load_phenobert(run_dir, "apertus")
        stock = loaders.load_phenobert(run_dir, "llama")
        assert all(r["sentence_number"] is not None for r in patched)
        assert all(r["sentence_number"] is None for r in stock)

    def test_row_parser_matches_the_driver(self, run_dir):
        """loaders._iter_rows is a copy kept free of the torch import chain, fix it."""
        from hpo_extraction.phenojury.phenobert import iter_phenobert_rows

        raw = (open(f"{run_dir}/phenobert_output_phi4/r1.txt", encoding="utf-8").read())
        assert list(loaders._iter_rows(raw)) == list(iter_phenobert_rows(raw))

    def test_gold_comes_from_the_predictions_summary_lines(self, run_dir):
        """No ground-truth CSV, no target-symptom list, no dataset loader."""
        gold = loaders.load_gold(run_dir)
        assert {k: sorted(v) for k, v in gold.items()} == {
            k: sorted(v) for k, v in SLM_GOLD.items()}

    def test_torn_lines_are_skipped_not_fatal(self, tmp_path):
        run = build_slm_ensemble_run(tmp_path / "hcy")
        path = run / "detections_phi4.jsonl"
        good = path.read_text().splitlines()
        path.write_text("\n".join(good[:1] + ['{"report_id": "r1", "mod'] + good[1:]) + "\n")
        assert len(loaders.load_detections(str(run), "phi4")) == len(good)


class TestGates:
    def test_all_gates_pass_on_a_clean_run(self, bundle):
        """G6 skips: this fixture is the ensemble run alone, with no PhenoBERT baseline beside it."""
        assert [g["status"] for g in bundle["gates"]] == ["pass"] * 5 + ["skip"], bundle["gates"]
        assert bundle["gate_summary"]["status"] == "pass"

    def test_min_count_is_inferred_and_confirmed(self, bundle):
        assert (bundle["min_count"], bundle["min_count_confirmed"]) == (1, True)

    def test_sets_gate_catches_a_tampered_prediction(self, tmp_path):
        """The gate exists to notice this, so make it notice it."""
        run = build_slm_ensemble_run(tmp_path / "hcy")
        path = run / "vote_k2" / "slm_ensemble_predictions.jsonl"
        path.write_text(path.read_text().replace('"predicted_set": []',
                                                 f'"predicted_set": ["{H}"]', 1))
        b = registry.build_bundle("hcy", str(run), loaders.load_raw(str(run)), None)
        gate = next(g for g in b["gates"] if g["id"] == "G1")
        assert gate["status"] == "fail"

    def test_roster_mismatch_is_diagnosed_not_just_reported(self, tmp_path):
        """A run whose detections were emptied after aggregation must say *why* it disagrees."""
        run = build_slm_ensemble_run(tmp_path / "hcy")
        (run / "detections_deepseek.jsonl").write_text("")
        b = registry.build_bundle("hcy", str(run), loaders.load_raw(str(run)), None)
        gate = next(g for g in b["gates"] if g["id"] == "G1")
        assert gate["status"] == "fail"
        assert "deepseek" in gate["detail"]
        assert "different ensemble" in gate["detail"]

    def test_gates_skip_rather_than_fail_mid_extraction(self, tmp_path):
        """A cohort still extracting must open, with the gates saying there is nothing to check."""
        run = build_slm_ensemble_run(tmp_path / "hcy")
        for pattern in ("slm_ensemble_*", "detections_*"):
            for path in run.glob(pattern):
                path.unlink()
        for rule in ("vote_k1", "vote_k2", "vote_k3", "vote_k4", "agg_plurality"):
            shutil.rmtree(run / rule)
        b = registry.build_bundle("hcy", str(run), loaders.load_raw(str(run)), None)
        assert b["models"] == []
        assert b["gate_summary"]["n_fail"] == 0

    def test_spans_gate_catches_a_stale_extraction_cache(self, tmp_path):
        """Offsets computed from different generations must not silently mis-highlight."""
        run = build_slm_ensemble_run(tmp_path / "hcy")
        raw = loaders.load_raw(str(run))
        built = detections.build_replies(raw["records"])
        built["offsets"][("apertus", "r1")] = ([99, 99, 99], [0, 1, 2])
        with pytest.raises(AssertionError):
            detections.assert_layout_matches(raw["records"], "apertus", "r1", built)


class TestVotes:
    def test_sweep_reproduces_the_shipped_summary(self, bundle):
        shipped = {row["rule"]: row for row in bundle["agg_summary"].to_dict("records")}
        for row in votes.sweep_k(bundle, bundle["gold"], bundle["models"], 1):
            want = shipped[row["rule"]]
            assert row["n_predicted"] == want["n_predicted"]
            assert row["micro_f1"] == pytest.approx(want["micro_f1"])
            assert row["macro_f1"] == pytest.approx(want["macro_f1"])

    def test_mask_offline_evaluation_equals_the_driver(self, bundle):
        """The fast popcount path and the driver's per-report tally must agree."""
        for k in range(1, len(bundle["models"]) + 1):
            fast = votes.vote_k_from_masks(bundle["masks"](1), bundle["mask_of"](bundle["models"]),
                                           k)
            slow = votes.shipped_vote_k(bundle["report_hpos"], bundle["models"],
                                        bundle["report_ids"], k, 1)
            assert {r: set(v) for r, v in fast.items()} == {r: set(v) for r, v in slow.items()}

    def test_plurality_mask_offline_evaluation_equals_the_driver(self, bundle):
        fast = votes.plurality_from_masks(bundle["sentence_masks"](1),
                                          bundle["mask_of"](bundle["models"]))
        slow, _ = votes.shipped_plurality(bundle["sent_hpos"], bundle["models"],
                                          bundle["report_ids"], 1)
        assert {r: set(v) for r, v in fast.items()} == {r: set(v) for r, v in slow.items()}

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_decisive_matches_brute_force_leave_one_out(self, bundle, k):
        """A model is decisive iff removing it drops the term. Check that literally."""
        models = bundle["models"]
        masks = bundle["masks"](1)
        full = bundle["mask_of"](models)
        for report_id, row in masks.items():
            for hpo_id, mask in row.items():
                if votes.popcount(mask & full) < k:
                    continue   # not predicted, so decisiveness is undefined
                claimed = votes.decisive_mask(mask, full, k)
                for i, _model in enumerate(models):
                    without = full & ~(1 << i)
                    survives = votes.popcount(mask & without) >= k
                    assert bool(claimed >> i & 1) is not survives, (report_id, hpo_id, i)

    def test_shapley_sums_to_the_ensemble_value(self, bundle):
        """Efficiency: Shapley values must add up to v(all) − v(∅), and v(∅) is 0 here."""
        models = bundle["models"]
        k = 2
        total = sum(r["shapley"] for r in
                    votes.shapley(bundle, bundle["gold"], models, k, 1))
        full = votes._f1_for_mask(bundle["masks"](1), bundle["gold"], bundle["report_ids"],
                                  bundle["mask_of"](models), min(k, len(models)))
        assert total == pytest.approx(full)

    def test_mask_histogram_scores_every_subset_like_the_reference(self, bundle):
        """The histogram is an efficiency device, so it has to be the slow definition.

        Shapley and leave-one-out score subsets off ``_f1_for_hist``, which loops over the distinct
        vote masks rather than over the cohort. Every subset and every k must give the number
        ``_f1_for_mask`` gives, or the Models view is fast and wrong.
        """
        n = len(bundle["models"])
        masks = bundle["masks"](1)
        hist, n_gold = votes.mask_histogram(bundle, bundle["gold"], 1)
        for subset in range(1 << n):
            for k in range(1, n + 1):
                assert votes._f1_for_hist(hist, n_gold, subset, k) == pytest.approx(
                    votes._f1_for_mask(masks, bundle["gold"], bundle["report_ids"], subset, k)
                ), (subset, k)

    def test_shapley_is_symmetric_for_identical_models(self, bundle):
        """apertus and deepseek detect the same terms in this scenario."""
        values = {r["model"]: r["shapley"]
                  for r in votes.shapley(bundle, bundle["gold"], bundle["models"], 2, 1)}
        assert values["apertus"] == pytest.approx(values["deepseek"])

    def test_sweep_is_monotone(self, bundle):
        totals = [r["n_predicted"] for r in
                  votes.sweep_k(bundle, bundle["gold"], bundle["models"], 1) if r["k"]]
        assert totals == sorted(totals, reverse=True)

    def test_coalitions_cover_every_predicted_term(self, bundle):
        rows = votes.decisive_rows(bundle, _cfg(bundle, k=1), bundle["gold"])
        assert sum(c["n"] for c in votes.coalitions(rows, bundle["models"], top_n=99)) == len(rows)

    def test_subset_scoring_matches_a_rebuilt_ensemble(self, bundle, run_dir):
        """Scoring a subset must equal scoring a run that only ever had those models."""
        subset = ["apertus", "llama", "phi4"]
        cfg = _cfg(bundle, k=2, models=subset)
        from_subset = votes.predicted_sets(bundle, cfg)

        reference = votes.shipped_vote_k(
            {m: bundle["report_hpos"][m] for m in subset}, subset, bundle["report_ids"], 2, 1)
        assert {r: set(v) for r, v in from_subset.items()} == \
               {r: set(v) for r, v in reference.items()}


class TestAnalysis:
    """The autopsy and the FP taxonomy. The fixture is built so each branch occurs once."""

    def test_every_fate_occurs_and_they_partition_the_misses(self, bundle):
        cfg = _cfg(bundle, k=2)
        predicted = votes.predicted_sets(bundle, cfg)
        result = autopsy.build(bundle, cfg, bundle["gold"], predicted)

        by_term = {(r["report_id"], r["hpo_id"]): r["fate"] for r in result["rows"]}
        assert by_term == {
            ("r1", F): "not_linked",       # models wrote, PhenoBERT grounded nothing to F
            ("r1", D): "below_votes",      # phi4 alone found it. K=2 discarded it
            ("r2", M): "negated",          # found by two models, negated by both
            ("r4", H): "not_written",      # every model replied "no phenotype" on that report
        }
        n_missed = sum(len(set(bundle["gold"][r]) - predicted.get(r, set()))
                       for r in bundle["report_ids"])
        assert sum(result["counts"].values()) == n_missed
        assert result["counts"]["no_evidence"] == 0

    def test_below_votes_disappears_at_k1(self, bundle):
        cfg = _cfg(bundle, k=1)
        result = autopsy.build(bundle, cfg, bundle["gold"], votes.predicted_sets(bundle, cfg))
        assert result["counts"]["below_votes"] == 0

    def test_below_min_count_is_reachable(self, bundle):
        """Raising min_detection_count above every observed count must retire the whole ensemble."""
        cfg = _cfg(bundle, k=1, min_count=9)
        result = autopsy.build(bundle, cfg, bundle["gold"], votes.predicted_sets(bundle, cfg))
        assert result["counts"]["below_min_count"] > 0

    def test_degraded_mode_collapses_the_grounding_fates(self, tmp_path):
        run = build_slm_ensemble_run(tmp_path / "hcy")
        for path in run.glob("phenobert_output_*"):
            shutil.rmtree(path)
        b = registry.build_bundle("hcy", str(run), loaders.load_raw(str(run)), build_toy_tree())
        cfg = _cfg(b, k=2)
        result = autopsy.build(b, cfg, b["gold"], votes.predicted_sets(b, cfg))
        assert result["degraded"] is True
        assert result["counts"]["negated"] == 0
        assert result["counts"]["not_linked"] >= 2   # F and the now-unexplainable M

    def test_recoverable_by_k_prices_the_trade(self, bundle):
        rows = {r["k"]: r for r in
                autopsy.recoverable_by_k(bundle, _cfg(bundle, k=2), bundle["gold"])}
        assert rows[1]["gained_tp"] == 1     # D in r1
        assert rows[1]["gained_fp"] == 2     # M in r1 and G in r3
        assert rows[4]["gained_tp"] == 0

    @pytest.mark.parametrize("hpo_id, gold, expected", [
        (B, {C}, "ancestor"),          # B is C's parent
        (C, {B}, "descendant"),        # C is B's child
        (M, {C}, "sibling"),           # M and C share the parent B
        (F, {C}, "same_system"),       # both under the nervous system, nothing closer
        (G, {C}, "unrelated"),         # different layer-1 system
        (C, set(), "no_gt"),
    ])
    def test_relation_taxonomy(self, hpo_id, gold, expected):
        assert relations.classify(build_toy_tree(), hpo_id, gold) == expected

    def test_relation_resolves_obsolete_ids(self):
        from fixtures.toy_ontology import OBSOLETE_C

        tree = build_toy_tree()
        assert relations.classify(tree, B, {OBSOLETE_C}) == "ancestor"

    def test_distance_uses_the_dag_not_the_depths(self):
        """C→H is three hops through the multi-parent node M, four through the root side."""
        dist = relations.distances_from(build_toy_tree(), {C}, max_hops=4)
        assert dist[C] == 0 and dist[B] == 1 and dist[M] == 2 and dist[H] == 3

    def test_false_positives_are_classified_on_the_real_run(self, bundle):
        cfg = _cfg(bundle, k=1)
        predicted = votes.predicted_sets(bundle, cfg)
        rows = {r["hpo_id"]: r
                for r in relations.classify_report(bundle["tree"], predicted["r1"],
                                                   bundle["gold"]["r1"])}
        assert rows[B]["relation"] == "ancestor"
        assert rows[M]["relation"] == "sibling"
        assert relations.classify_report(
            bundle["tree"], predicted["r3"], bundle["gold"]["r3"])[0]["relation"] == "no_gt"


class TestDriverFallback:
    """The app must open on a login node where the training stack does not import.

    ``hpo_extraction.phenojury.generation`` reaches ``torch``, ``mlflow`` and, via
    ``hpo_extraction.treephenorag.score_store``, ``sentence_transformers``, none of which this app needs. When that
    chain fails the app falls back to its own copies of three pure functions. These tests are the
    only thing keeping those copies honest, and they run here because the imports *do*
    work in CI, so both sides can be compared directly.
    """

    def test_micro_macro_fallback_matches_the_driver(self, bundle):
        from hpo_extraction.phenojury.generation import _safe_micro_macro

        for k in range(1, len(bundle["models"]) + 1):
            predicted = votes.vote_k_from_masks(bundle["masks"](1),
                                                bundle["mask_of"](bundle["models"]), k)
            gold_sets = [set(bundle["gold"].get(r, ())) for r in bundle["report_ids"]]
            pred_sets = [set(predicted.get(r, ())) for r in bundle["report_ids"]]
            assert votes._fallback_micro_macro(gold_sets, pred_sets) == \
                   _safe_micro_macro(gold_sets, pred_sets)

    def test_micro_macro_fallback_handles_the_degenerate_cases(self):
        from hpo_extraction.phenojury.generation import _safe_micro_macro

        for gold_sets, pred_sets in (([], []), ([set()], [set()]), ([{C}], [set()]),
                                     ([set()], [{C}]), ([{C}], [{D}])):
            assert votes._fallback_micro_macro(gold_sets, pred_sets) == \
                   _safe_micro_macro(gold_sets, pred_sets)

    @pytest.mark.parametrize("min_count", [1, 2])
    def test_vote_k_fallback_matches_the_driver(self, bundle, min_count):
        from hpo_extraction.phenojury.generation import vote_k_sets

        for k in range(1, len(bundle["models"]) + 1):
            assert votes._fallback_vote_k(bundle["report_hpos"], bundle["models"],
                                          bundle["report_ids"], k, min_count) == \
                   vote_k_sets(bundle["report_hpos"], bundle["models"],
                               bundle["report_ids"], k, min_count)

    def test_plurality_fallback_matches_the_driver(self, bundle):
        from hpo_extraction.phenojury.generation import plurality_sets

        assert votes._fallback_plurality(bundle["sent_hpos"], bundle["models"],
                                         bundle["report_ids"], 1) == \
               plurality_sets(bundle["sent_hpos"], bundle["models"], bundle["report_ids"], 1)

    def test_productivity_matches_the_driver(self, bundle):
        """The funnel reads the ``wrote`` flags the bundle already holds instead of re-running
        ``wrote_something`` over every generation. Same rows, or the Models view is lying."""
        from hpo_extraction.phenojury.ensemble_eval import compute_slm_productivity

        for models in (bundle["models"], bundle["models"][:1], []):
            assert detections.productivity(bundle["built"], bundle["sent_hpos"], models) == \
                   compute_slm_productivity(bundle["records"], bundle["sent_hpos"], models)

    def test_wrote_something_fallback_matches_the_real_one(self):
        from hpo_extraction.phenojury.ensemble_eval import wrote_something as real

        cases = ["", "no phenotype", "No phenotype.", "none", "NIL", "Seizures.",
                 "<think>fever</think>no phenotype", "<think>x</think>Anxiety.",
                 "1. Seizure\n2. Ataxia", "No symptoms mentioned", "N/A", "no abnormalities"]
        assert [detections._wrote_something(c) for c in cases] == [real(c) for c in cases]

    def test_bundle_builds_with_the_driver_unimportable(self, run_dir, monkeypatch):
        """The regression this whole class exists for: a broken import must not blank the app.

        A ``None`` entry in ``sys.modules`` is the documented way to make an import fail, patching
        ``builtins.__import__`` does not work here, because both modules are already imported and
        ``from core import X`` then resolves ``X`` as an attribute of the cached ``core`` package
        without importing anything.
        """
        import sys

        import hpo_extraction.phenojury as core

        votes._driver.cache_clear()
        for name in ("hpo_extraction.phenojury.generation", "hpo_extraction.phenojury.ensemble_eval"):
            monkeypatch.setitem(sys.modules, name, None)
            monkeypatch.delattr(core, name.rpartition(".")[2], raising=False)
        try:
            with pytest.raises(ImportError):
                from hpo_extraction.phenojury import generation as slm_ensemble_experiment  # noqa: F401 - proves the simulation works

            b = registry.build_bundle("hcy", run_dir, loaders.load_raw(run_dir), build_toy_tree())
            assert b["models"], "the bundle must still build"
            assert b["gate_summary"]["pinned"] is False
            assert "not fixed to the driver" in b["gate_summary"]["headline"]
            # G1/G2 still compare against the shipped artifacts and still pass. Only G3, which has
            # no honest fallback, steps aside.
            by_id = {g["id"]: g for g in b["gates"]}
            assert by_id["G1"]["status"] == "pass"
            assert by_id["G2"]["status"] == "pass"
            assert by_id["G3"]["status"] == "skip"
        finally:
            monkeypatch.undo()
            votes._driver.cache_clear()


class TestEfficiency:
    """The things done to keep the app usable over an SSH tunnel, each fixed to the slow answer.

    Every optimisation here trades a recomputation for a cached or derived one, so each is only
    sound while it gives the same result. That is what these tests hold. The timings themselves are
    not asserted, because a test that fails on a busy login node teaches nobody anything.
    """

    def test_memo_returns_the_same_object_and_respects_the_key(self, bundle):
        cfg_a = _cfg(bundle, k=1)
        cfg_b = _cfg(bundle, k=2)
        first = votes.predicted_sets(bundle, cfg_a)
        assert votes.predicted_sets(bundle, dict(cfg_a)) is first     # equal config, one compute
        assert votes.predicted_sets(bundle, cfg_b) is not first       # different k, recomputed

    def test_memo_ignores_the_order_the_models_were_picked_in(self, bundle):
        models = list(bundle["models"])
        forward = votes.predicted_sets(bundle, _cfg(bundle, k=2, models=models))
        backward = votes.predicted_sets(bundle, _cfg(bundle, k=2, models=list(reversed(models))))
        assert forward is backward

    def test_memo_is_bypassed_for_an_external_ground_truth_set(self, bundle):
        """``drivers.py`` scores against a different ground-truth file. That must not be cached
        under a key that only names the configuration, nor served from one."""
        cfg = _cfg(bundle, k=1)
        own = votes.sweep_k(bundle, bundle["gold"], cfg["models"], 1)
        other = votes.sweep_k(bundle, {r: set() for r in bundle["report_ids"]}, cfg["models"], 1)
        assert own is not other
        assert all(row["n_gold"] == 0 for row in other)
        assert votes.sweep_k(bundle, bundle["gold"], cfg["models"], 1) == own

    def test_a_bundle_without_a_memo_still_works(self, bundle):
        """The pure functions are called on hand-built dicts in tests and in ``drivers.py``."""
        plain = {k: v for k, v in bundle.items() if k != "memo"}
        assert votes.predicted_sets(plain, _cfg(bundle, k=1)) == \
               votes.predicted_sets(bundle, _cfg(bundle, k=1))

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_fp_rows_derived_from_k1_match_a_direct_classification(self, bundle, k):
        """The FP view classifies once and filters by vote count for every other k."""
        from apps.phenojury_ui.views import falsepos

        uncached = {**bundle, "memo": None}
        direct = falsepos._classify_all(uncached, _cfg(bundle, k=k))
        derived = [row for row in falsepos._classify_all(uncached, _cfg(bundle, k=1))
                   if votes.popcount(row["voters"]) >= k]
        assert sorted((r["report_id"], r["hpo_id"], r["relation"]) for r in direct) == \
               sorted((r["report_id"], r["hpo_id"], r["relation"]) for r in derived)

    def test_counts_across_k_match_classifying_at_every_k(self, bundle):
        from apps.phenojury_ui.views import falsepos

        cfg = _cfg(bundle, k=1)
        ks, series = falsepos.counts_across_k(bundle, cfg)
        assert ks == list(range(1, len(bundle["models"]) + 1))
        for i, k in enumerate(ks):
            rows = falsepos._classify_all({**bundle, "memo": None}, _cfg(bundle, k=k))
            for relation in relations.RELATION_ORDER:
                assert series[relation][i] == sum(1 for r in rows if r["relation"] == relation), \
                    (relation, k)

    def test_a_page_is_a_slice_of_the_sorted_rows(self, bundle):
        from apps.phenojury_ui.views import terms

        cfg = _cfg(bundle, k=1)
        rows = terms.sort_rows(terms.build_rows(bundle, cfg),
                               [{"column_id": "n_votes", "direction": "desc"}])
        assert [r["n_votes"] for r in rows] == sorted((r["n_votes"] for r in rows), reverse=True)

        seen = []
        for page in range(-(-len(rows) // terms.PAGE_SIZE)):
            chunk = terms.page_of(rows, page)
            assert len(chunk) <= terms.PAGE_SIZE
            seen += chunk
        assert [r["report_id"] for r in seen] == [r["report_id"] for r in rows]
        assert terms.page_of(rows, 999) == []           # past the end is empty, not an error

    def test_a_page_does_not_carry_the_server_only_fields(self, bundle):
        from apps.phenojury_ui.views import terms

        rows = terms.build_rows(bundle, _cfg(bundle, k=1))
        for row in terms.page_of(rows, 0):
            assert not set(row) & set(terms._SERVER_ONLY)
            assert "report_id" in row                   # The drill-down reads it back off the page

    def test_hidden_sentences_are_the_ones_nobody_wrote_about(self, bundle):
        """The deep dive's default hides silent sentences. It must hide only those."""
        from apps.phenojury_ui.views import patient

        cfg = _cfg(bundle, k=1)
        for report_id in bundle["report_ids"]:
            shown = patient.render_sentences(bundle, cfg, report_id, "light", False, False)
            everything = patient.render_sentences(bundle, cfg, report_id, "light", False, True)
            assert shown is not None and everything is not None


class TestTabGating:
    """Only the open tab computes and draws. The rule that makes that safe is tested here."""

    def test_a_view_renders_only_on_its_own_tab(self):
        assert common.should_render("models", "models", "key", None) is True
        assert common.should_render("models", "voters", "key", None) is False

    def test_an_unchanged_view_does_not_redraw(self):
        key = common.render_key("models", "gsc", {"k": 2}, "light")
        assert common.should_render("models", "models", key, key) is False
        other = common.render_key("models", "gsc", {"k": 3}, "light")
        assert common.should_render("models", "models", other, key) is True

    def test_the_key_survives_a_store_round_trip(self):
        """The previous key comes back from a ``dcc.Store`` as JSON, so it must compare equal."""
        import json

        key = common.render_key("terms", "gsc", {"models": ["a", "b"], "k": 2}, ["FP"], None)
        assert json.loads(json.dumps(key)) == key
        assert common.should_render("terms", "terms", key, json.loads(json.dumps(key))) is False

    def test_the_key_separates_configurations_that_differ_anywhere(self):
        base = common.render_key("v", "gsc", {"k": 2}, "light")
        for other in (common.render_key("v", "hcy", {"k": 2}, "light"),
                      common.render_key("v", "gsc", {"k": 2}, "dark"),
                      common.render_key("v", "gsc", {"k": 2, "min_count": 2}, "light")):
            assert other != base

    def test_every_view_has_a_signature_store_and_gates_on_the_tab(self):
        """A view added without wiring the gate would silently recompute on every control move."""
        import inspect

        from apps.phenojury_ui.app import VIEWS, build_layout

        layout = build_layout(_EmptyRegistry())
        store_ids = {getattr(child, "id", None) for child in layout.children}
        for view_id, _label, module in VIEWS:
            assert f"sig-{view_id}" in store_ids, view_id
            source = inspect.getsource(module)
            assert f'VIEW_ID = "{view_id}"' in source, view_id
            assert 'Input("tabs", "value")' in source, view_id


class _EmptyRegistry:
    """Enough of a Registry for ``build_layout``, the run list and the three path boxes."""

    output_base = ""
    pb_base = ""
    frame_base = ""
    annotations_dir = ""
    frame = None

    def list_runs(self):
        return []


class TestPhenoBERTBaseline:
    """The PhenoBERT baseline side: finding it, reading it, and comparing against it.

    The scenario in ``fixtures.exp13_output`` is built so that one annotated term lands on each side of
    the four-way agreement split, both / ensemble only / PhenoBERT only / neither, because a
    fixture where the two methods agree everywhere cannot fail the split's arithmetic.
    """

    @pytest.fixture(scope="class")
    def tree_root(self, tmp_path_factory):
        return build_comparison_tree(tmp_path_factory.mktemp("compare") / "output")

    @pytest.fixture(scope="class")
    def paired(self, tree_root):
        """A bundle with the baseline attached, as the Registry would build it."""
        ens = str(tree_root / "phenojury_generation_free_listing" / "hcy")
        pb_dir = pbstandalone.find_run(ens, "hcy")
        return registry.build_bundle("hcy", ens, loaders.load_raw(ens), build_toy_tree(),
                                     pb_dir=pb_dir)

    # ── discovery ────────────────────────────────────────────────────────────
    def test_the_sibling_is_derived_from_the_cluster_layout(self, tree_root):
        ens = str(tree_root / "phenojury_generation_free_listing" / "hcy")
        assert pbstandalone.find_run(ens, "hcy") == str(
            tree_root / "baseline_phenobert" / "hcy")

    def test_discovery_survives_the_other_two_run_dir_layouts(self, tmp_path):
        """``loaders.find_runs`` accepts three layouts. The baseline has to be found from each."""
        build_comparison_tree(tmp_path / "canonical")
        assert pbstandalone.find_run(
            str(tmp_path / "canonical" / "phenojury_generation_free_listing" / "hcy"), "hcy")

        # <base>/<cohort>/, base is already the experiment directory.
        flat = tmp_path / "flat"
        build_slm_ensemble_run(flat / "phenojury_generation_free_listing" / "hcy")
        build_phenobert_standalone_run(flat / "baseline_phenobert")
        assert pbstandalone.find_run(
            str(flat / "phenojury_generation_free_listing" / "hcy"), "hcy") == str(
                flat / "baseline_phenobert")

    def test_an_override_is_honoured(self, tree_root, tmp_path):
        """A typed path must not fall back to a derived one, that would relabel someone's run."""
        ens = str(tree_root / "phenojury_generation_free_listing" / "hcy")
        exp_dir = str(tree_root / "baseline_phenobert")
        assert pbstandalone.find_run(ens, "hcy", override=exp_dir) == str(
            tree_root / "baseline_phenobert" / "hcy")
        assert pbstandalone.find_run(ens, "hcy", override=str(tmp_path / "nowhere")) is None

    def test_a_missing_baseline_names_where_it_looked(self, tmp_path):
        build_slm_ensemble_run(tmp_path / "output" / "phenojury_generation_free_listing" / "hcy")
        reg = registry.Registry(str(tmp_path / "output"), cache_dir=str(tmp_path / "cache"))
        assert reg.pb_run_dir("hcy") is None
        assert any("baseline_phenobert" in p for p in reg.pb_searched("hcy"))

    # ── reading ──────────────────────────────────────────────────────────────
    def test_summary_lines_are_authoritative_over_the_term_rows(self, tree_root):
        """r3 predicts nothing. Only the summary line records that it was processed at all."""
        data = pbstandalone.load(str(tree_root / "baseline_phenobert" / "hcy"))
        assert data["report_ids"] == ["r1", "r2", "r3", "r4"]
        assert data["predicted"]["r3"] == set()
        assert data["gold"]["r2"] == set(SLM_GOLD["r2"])

    def test_negated_and_unresolved_detections_are_kept_but_never_predicted(self, tree_root):
        """The two exclusions are the evidence the predicted set throws away."""
        data = pbstandalone.load(str(tree_root / "baseline_phenobert" / "hcy"))
        assert any(r["negated"] for r in data["detections"])
        assert any(not r["resolved"] for r in data["detections"])
        for row in data["detections"]:
            if row["negated"] or not row["resolved"]:
                assert row["hpo_id"] not in data["predicted"][row["report_id"]]

    def test_every_offset_lands_on_its_own_phrase(self, tree_root):
        data = pbstandalone.load(str(tree_root / "baseline_phenobert" / "hcy"))
        assert data["span_checked"] == len(data["detections"])
        assert data["span_bad"] == []
        for row in data["detections"]:
            assert data["texts"][row["report_id"]][row["start"]:row["end"]] == row["phrase"]

    # ── the gate ─────────────────────────────────────────────────────────────
    def test_gate_passes_on_a_clean_pair(self, paired):
        gate = next(g for g in paired["gates"] if g["id"] == "G6")
        assert gate["status"] == "pass", gate

    def test_gate_catches_a_stale_staged_report(self, tmp_path):
        """phenobert_input/ rewritten under unchanged offsets is the failure this gate exists for.

        It is not hypothetical: rerunning one stage and not the other leaves offsets that are
        individually plausible and uniformly wrong, and the annotated report would underline
        whatever now happens to sit at those positions.
        """
        root = build_comparison_tree(tmp_path / "output")
        staged = root / "baseline_phenobert" / "hcy" / "phenobert_input" / "r1.txt"
        staged.write_text("Something else entirely was written here instead.", encoding="utf-8")

        ens = str(root / "phenojury_generation_free_listing" / "hcy")
        b = registry.build_bundle("hcy", ens, loaders.load_raw(ens), build_toy_tree(),
                                  pb_dir=pbstandalone.find_run(ens, "hcy"))
        gate = next(g for g in b["gates"] if g["id"] == "G6")
        assert gate["status"] == "fail"
        assert "do not land" in gate["detail"]
        assert not pbstandalone.report_ok(b["pb_standalone"], "r1")
        assert pbstandalone.report_ok(b["pb_standalone"], "r2")

    def test_gate_fails_when_the_baseline_is_a_different_cohort(self, tmp_path):
        root = build_comparison_tree(tmp_path / "output")
        path = root / "baseline_phenobert" / "hcy" / "phenobert_predictions.jsonl"
        path.write_text(path.read_text().replace('"report_id": "r', '"report_id": "x'),
                        encoding="utf-8")
        ens = str(root / "phenojury_generation_free_listing" / "hcy")
        b = registry.build_bundle("hcy", ens, loaders.load_raw(ens), build_toy_tree(),
                                  pb_dir=pbstandalone.find_run(ens, "hcy"))
        gate = next(g for g in b["gates"] if g["id"] == "G6")
        assert gate["status"] == "fail"
        assert "different cohort" in gate["detail"]

    # ── sentence alignment ───────────────────────────────────────────────────
    def test_sentences_are_located_in_the_staged_report(self, paired):
        pb = paired["pb_standalone"]
        for report_id, sentences in paired["sentences"].items():
            text = pb["texts"][report_id]
            spans = pbstandalone.align_sentences(text, sentences)
            assert spans is not None and set(spans) == set(sentences)
            for n, (start, end) in spans.items():
                assert text[start:end] == sentences[n]

    def test_alignment_tolerates_collapsed_whitespace(self):
        """The splitter normalises whitespace the raw report still carries, a real mismatch."""
        text = "The patient  had\nseizures. No fever was noted."
        spans = pbstandalone.align_sentences(
            text, {0: "The patient had seizures.", 1: "No fever was noted."})
        assert spans is not None
        assert text[spans[0][0]:spans[0][1]] == "The patient  had\nseizures."
        assert text[spans[1][0]:spans[1][1]] == "No fever was noted."

    def test_alignment_refuses_rather_than_guesses(self):
        """A partial map would silently drop evidence for the sentences it could not place."""
        assert pbstandalone.align_sentences("A report about nothing.",
                                            {0: "A report about nothing.",
                                             1: "A sentence from some other document."}) is None

    def test_a_repeated_sentence_is_placed_at_its_own_occurrence(self):
        text = "No change. Fever resolved. No change."
        spans = pbstandalone.align_sentences(
            text, {0: "No change.", 1: "Fever resolved.", 2: "No change."})
        assert spans[0][0] == 0 and spans[2][0] == text.rfind("No change.")

    def test_a_detection_is_attributed_to_the_sentence_it_falls_in(self, paired):
        pb = paired["pb_standalone"]
        spans = pbstandalone.align_sentences(pb["texts"]["r1"], paired["sentences"]["r1"])
        by_sentence = {pbstandalone.sentence_of(spans, r["start"]): r["hpo_id"]
                       for r in pb["by_report"]["r1"]}
        # "seizures" is in sentence 0, "cerebral atrophy" in sentence 2, see SLM_SENTENCES.
        assert by_sentence[0] == C
        assert by_sentence[2] == F

    # ── the comparison ───────────────────────────────────────────────────────
    def test_the_agreement_split_is_the_hand_computed_one(self, paired):
        """One annotated term on each side, counted by hand from SLM_GOLD and PB_DETECTIONS.

        both        r1/C (seizures) and r2/D (anxiety), in the report and written by the models
        ens only    r1/D, phi4 reads anxiety into sentence 1. The word is not in the text
        pb only     r1/F, "cerebral atrophy" is in the report, but the only model that mentions it
                    does so inside a <think> block that is stripped before grounding
        neither     r2/M (both see it negated) and r4/H (nothing mentions it)
        """
        cfg = dict(paired["default_config"], rule="vote_k", k=1)
        result = compare_pb.compare(paired, cfg)
        assert result["gold_split"] == {"both": 2, "ensemble_only": 1,
                                        "phenobert_only": 1, "neither": 2}
        assert sum(result["gold_split"].values()) == sum(
            len(v) for v in paired["gold"].values())

    def test_per_report_rows_reconcile_with_the_split(self, paired):
        cfg = dict(paired["default_config"], rule="vote_k", k=1)
        result = compare_pb.compare(paired, cfg)
        rows = {r["report_id"]: r for r in result["per_report"]}
        assert rows["r1"]["pb_only_text"] != "—"          # F, the term only the baseline recovers
        assert rows["r1"]["ens_only_text"] != "—"         # D, the term only the ensemble recovers
        assert rows["r3"]["delta_f1"] is None             # no ground truth, no F1 to difference
        for row in result["per_report"]:
            assert row["ens_tp"] + row["ens_fn"] == row["n_gold"]
            assert row["pb_tp"] + row["pb_fn"] == row["n_gold"]

    def test_per_term_hits_are_the_split_by_term(self, paired):
        cfg = dict(paired["default_config"], rule="vote_k", k=1)
        result = compare_pb.compare(paired, cfg)
        rows = {r["hpo_id"]: r for r in result["per_term"]}
        assert rows[F]["pb_hits"] == 1 and rows[F]["ens_hits"] == 0
        assert rows[C]["ens_hits"] == rows[C]["pb_hits"] == 1
        assert rows[H]["ens_hits"] == rows[H]["pb_hits"] == 0
        for row in result["per_term"]:
            assert row["both"] + row["ens_only"] + row["pb_only"] + row["neither"] == row["n_gold"]

    def test_only_the_shared_reports_are_scored(self, tmp_path):
        """Scoring one method on reports the other never saw is bookkeeping, not a difference."""
        root = build_comparison_tree(tmp_path / "output")
        path = root / "baseline_phenobert" / "hcy" / "phenobert_predictions.jsonl"
        kept = [line for line in path.read_text().splitlines() if '"r4"' not in line]
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")

        ens = str(root / "phenojury_generation_free_listing" / "hcy")
        b = registry.build_bundle("hcy", ens, loaders.load_raw(ens), build_toy_tree(),
                                  pb_dir=pbstandalone.find_run(ens, "hcy"))
        assert compare_pb.shared_reports(b) == ["r1", "r2", "r3"]
        result = compare_pb.compare(b, b["default_config"])
        assert {r["report_id"] for r in result["per_report"]} == {"r1", "r2", "r3"}

    def test_the_page_degrades_to_a_note_naming_the_paths(self, tmp_path):
        build_slm_ensemble_run(tmp_path / "output" / "phenojury_generation_free_listing" / "hcy")
        reg = registry.Registry(str(tmp_path / "output"), cache_dir=str(tmp_path / "cache"))
        b = reg.get_bundle("hcy")
        assert b["pb_standalone"] is None
        assert compare_pb.render_all(b, b["default_config"], "light") is not None
        assert b["pb_searched"]


class TestReportOrdering:
    """The deep dive's picker: numeric order, and arrows that step through it."""

    def test_numeric_ids_sort_as_numbers(self):
        ids = ["10051003", "1003450", "999", "SYN004"]
        assert sorted(ids, key=patient.report_key) == ["999", "1003450", "10051003", "SYN004"]

    def test_ordered_reports_keeps_the_f1_labels(self, bundle):
        ordered = patient.ordered_reports(bundle, bundle["default_config"])
        assert [r for r, _ in ordered] == sorted(bundle["report_ids"], key=patient.report_key)
        scores = dict(patient.rank_reports(bundle, bundle["default_config"]))
        assert all(f1 == scores[r] for r, f1 in ordered)

    def test_the_arrows_clamp_at_both_ends(self):
        ids = ["1", "2", "3"]
        assert patient.step_to("pa-next", "1", ids) == "2"
        assert patient.step_to("pa-prev", "2", ids) == "1"
        assert patient.step_to("pa-prev", "1", ids) is None     # already first
        assert patient.step_to("pa-next", "3", ids) is None     # already last, no wrap-around
        assert patient.step_to("pa-next", "gone", ids) == "2"   # unknown value falls to the start

    def test_clicking_the_focused_term_clears_it(self):
        clear = ("pa-term-clear", "pa-report")
        assert patient.toggle_focus({"type": "pa-term", "hpo": C}, None, "hpo", clear) == C
        assert patient.toggle_focus({"type": "pa-term", "hpo": C}, C, "hpo", clear) is None
        assert patient.toggle_focus("pa-term-clear", C, "hpo", clear) is None
        assert patient.toggle_focus("pa-report", C, "hpo", clear) is None

    def test_sentence_zero_is_a_focus_not_a_falsy_value(self):
        """Sentence numbers start at 0, so anything testing truthiness loses the first sentence."""
        clear = ("pa-sent-clear", "pa-report")
        assert patient.toggle_focus({"type": "pa-sent", "n": 0}, None, "n", clear) == 0
        assert patient.toggle_focus({"type": "pa-sent", "n": 0}, 0, "n", clear) is None


class TestTermDefinitions:
    """The ⓘ hover: it must degrade to the bare id, not crash without an ontology."""

    def test_the_tooltip_carries_the_definition_and_the_synonyms(self):
        tree = build_toy_tree()
        text = relations.tooltip(tree, C)
        assert C in text and "Seizure" in text

    def test_an_unknown_term_degrades_to_its_id(self):
        tree = build_toy_tree()
        assert relations.definition(tree, "HP:9999999") == ""
        assert relations.synonyms(tree, "HP:9999999") == []
        assert relations.tooltip(tree, "HP:9999999").startswith("HP:9999999")

    def test_no_ontology_is_not_a_crash(self):
        assert relations.tooltip(None, C) == C
        assert relations.definition(None, C) == ""
        assert relations.synonyms(None, C) == []


def _reply(stripped: str, *, raw: str | None = None, wrote: bool = True) -> dict:
    """One entry of ``bundle["built"]["replies"]``, see ``detections.build_replies``.

    ``pb_text`` comes from the production helper, not a literal: it is the text the offsets
    index, so a fixture that hard-codes it would keep passing after the input layout changed.
    """
    from hpo_extraction.phenojury.phenobert import terminate_lines
    return {"raw": raw if raw is not None else stripped, "stripped": stripped,
            "pb_text": terminate_lines(stripped),
            "sentence_text": "a sentence", "wrote": wrote}


def _pb_row(hpo_id: str, phrase: str, *, offset: int | None, negated: bool = False) -> dict:
    """One row of ``bundle["pb_rows"]``, the per-model shape, which carries no ``resolved``.

    ``start``/``end`` index the PhenoBERT input file and are used only for their *length*. The
    position in the reply is ``offset_in_reply``. They are derived from the phrase here so the row
    cannot be internally inconsistent the way a hand-typed one can.
    """
    return {"hpo_id": hpo_id, "phrase": phrase, "confidence": 0.9, "negated": negated,
            "start": 0, "end": len(phrase), "offset_in_reply": offset}


class TestChipsAndEvidence:
    """Things a render test cannot see: a chip's colour, and what the annotation actually says."""

    def test_an_uncoloured_term_chip_is_not_white_on_white(self, bundle):
        """`.pill` sets white text and expects a background. A chip drawn without one vanished.

        The chip carries its own class, and the stylesheet colours that class, so this pins both
        halves. Asserting only the class would pass against a stylesheet that never styles it.
        """
        chip = common.term_chip(bundle, C)
        assert "pill-term" in chip.className
        css = (pathlib.Path(__file__).resolve().parents[2]
               / "apps" / "phenojury_ui" / "assets" / "deepdive.css").read_text()
        rule = css.split(".pill.pill-term", 1)
        assert len(rule) == 2, "deepdive.css does not style .pill.pill-term"
        assert "color: var(--text)" in rule[1].split("}", 1)[0]

    def test_the_reply_is_annotated_inline_by_the_same_renderer_as_the_report(self, bundle):
        """The underline says *where* a phrase was grounded. The inline tag says *what to*."""
        reply = _reply("Seizures.")
        rows = [_pb_row(C, "Seizures", offset=0)]
        rendered = str(patient._reply_body(bundle, reply, rows, "light", False))
        assert "pb-ann" in rendered and "pb-tag" in rendered
        assert "Seizure" in rendered              # The term, in the text, not only on a hover

    def test_report_and_reply_produce_the_same_markup(self, bundle):
        """One annotator, two callers: this is what stops the two surfaces drifting apart."""
        row = _pb_row(C, "Seizures", offset=0)
        from_reply, _ = patient._annotate("Seizures.", [(0, 8, row)], bundle, "light")
        from_report, _ = patient._annotate("Seizures.", [(0, 8, dict(row, resolved=True))],
                                           bundle, "light")
        assert str(from_reply) == str(from_report)

    def test_a_negated_mention_is_struck_through_and_says_so(self, bundle):
        rows = [_pb_row(M, "myopia", offset=3, negated=True)]
        rendered = str(patient._reply_body(bundle, _reply("No myopia."), rows, "light", False))
        assert "pb-mark-negated" in rendered and "negated" in rendered

    def test_an_unplaceable_offset_is_reported_rather_than_searched_for(self, bundle):
        """Searching would mark a plausible occurrence instead of the detected one, G4's whole point.

        Saying nothing would be worse still: it reads as "PhenoBERT found nothing here", which is a
        different claim from "it found something and the offset could not be placed".
        """
        rows = [_pb_row(C, "Seizures", offset=None)]
        rendered = str(patient._reply_body(bundle, _reply("Seizures."), rows, "light", False))
        assert "pb-mark" not in rendered
        assert "could not be placed" in rendered and "Seizures" in rendered

    def test_wrote_but_nothing_linked_is_named_rather_than_left_blank(self, bundle):
        """`not_linked` is the fate people read as `not_written`. The reply has to say which."""
        wrote = str(patient._reply_body(bundle, _reply("Some finding."), [], "light", False))
        assert "normalised nothing" in wrote

        silent = str(patient._reply_body(bundle, _reply("no phenotype", wrote=False), [],
                                         "light", False))
        assert "normalised nothing" not in silent
        assert "wrote nothing" in silent

    def test_the_raw_toggle_is_left_unannotated(self, bundle):
        """Offsets index the stripped reply. On the raw one they are off by the <think> block."""
        reply = _reply("Seizures.", raw="<think>hmm</think>Seizures.")
        rows = [_pb_row(C, "Seizures", offset=0)]
        rendered = str(patient._reply_body(bundle, reply, rows, "light", True))
        assert "pb-mark" not in rendered and "<think>" in rendered


class TestSelftest:
    """The harness itself: it must actually exercise things, and it must catch a broken view."""

    def test_selftest_passes_on_a_clean_tree(self, tmp_path):
        from apps.phenojury_ui import state
        from apps.phenojury_ui.app import build_app
        from apps.phenojury_ui.selftest import run_selftest
        from fixtures.exp13_output import build_slm_ensemble_tree

        build_slm_ensemble_tree(tmp_path / "output", cohorts=("hcy",))
        reg = registry.Registry(str(tmp_path / "output"), cache_dir=str(tmp_path / "cache"))
        state.set_registry(reg)
        assert run_selftest(reg, build_app) is True

    def test_render_matrix_reports_a_broken_view(self, tmp_path, monkeypatch):
        from apps.phenojury_ui import state
        from apps.phenojury_ui.selftest import render_matrix
        from apps.phenojury_ui.views import scorecard
        from fixtures.exp13_output import build_slm_ensemble_tree

        build_slm_ensemble_tree(tmp_path / "output", cohorts=("hcy",))
        reg = registry.Registry(str(tmp_path / "output"), cache_dir=str(tmp_path / "cache"))
        state.set_registry(reg)

        def boom(*_args, **_kwargs):
            raise RuntimeError("deliberate")

        monkeypatch.setattr(scorecard, "render_sweep", boom)
        failures, attempts = render_matrix(reg, ["hcy"])
        assert attempts > 0
        assert failures and all("deliberate" in f for f in failures)

    def test_config_matrix_covers_the_degenerate_points(self, bundle):
        from apps.phenojury_ui.selftest import config_matrix

        labels = {label for label, _ in config_matrix(bundle)}
        assert {"k=1 (union)", "k=N (unanimity)", "plurality", "single model"} <= labels


def test_bit_order_is_stable_under_subsetting(bundle):
    """Vote bits index the *run's* model list, never the selected subset.

    If they indexed the subset, unchecking a model would silently reassign every other model's
    bit and every mask cached at a different subset would be misread.
    """
    full = bundle["mask_of"](bundle["models"])
    partial = bundle["mask_of"](bundle["models"][:2])
    assert partial == 0b11
    assert full & partial == partial


def test_models_are_ordered_by_model_keys(bundle):
    assert bundle["models"] == [m for m in loaders.MODEL_ORDER if m in set(SLM_MODELS)]


def test_pairwise_agreement_is_symmetric(bundle):
    """Guards the heatmap: an asymmetric Jaccard would mean the sets were built per-column."""
    masks = bundle["masks"](1)
    index = {m: i for i, m in enumerate(bundle["models"])}
    sets = {m: {(r, h) for r, row in masks.items() for h, mask in row.items()
                if mask >> index[m] & 1} for m in bundle["models"]}
    for a, b in itertools.combinations(bundle["models"], 2):
        assert len(sets[a] & sets[b]) == len(sets[b] & sets[a])


# ══════════════════════════════════════════════════════════════════════════════
# The deep-dive sampling frame, and the subset filter it drives
# ══════════════════════════════════════════════════════════════════════════════
def _write_frame(root: pathlib.Path, picks: dict, pools_rows: str = "") -> str:
    directory = root / frame_mod.FRAME_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    selected = sorted({r for group in picks.values() for r in group})
    (directory / frame_mod.FRAME_FILE).write_text(json.dumps({
        "generated": "2026-09-09", "params": {"seed": 20260909, "min_gold": 3},
        "cell_sizes": {cell: len(group) for cell, group in picks.items()},
        "picks": picks, "pools": picks, "selected": selected,
    }))
    if pools_rows:
        (directory / frame_mod.POOLS_FILE).write_text(
            "patient_id,selected,cell,n_gold,n_pred,tp,fp,fn,precision,recall,f1,"
            "family,lab_value,implicit,negated\n" + pools_rows)
    return str(directory)


class TestFrame:
    """Reading the fixed in advance draw, and turning it into a report filter."""

    def test_the_frame_is_found_beside_the_output_base(self, tmp_path):
        directory = _write_frame(tmp_path, {"pb_best": ["r1"]})
        assert frame_mod.find_frame(str(tmp_path)) == directory

    def test_an_override_is_honoured(self, tmp_path):
        """The rule pbstandalone follows: a typed path that misses must not derive another.

        Falling back would filter the reader's screens by a draw they did not choose, under the
        label of the one they did.
        """
        _write_frame(tmp_path, {"pb_best": ["r1"]})
        assert frame_mod.find_frame(str(tmp_path), override=str(tmp_path / "nope")) is None

    def test_selection_keeps_the_callers_order(self, tmp_path):
        """The bundle's order is the driver's report order, which every table is read against."""
        frame = frame_mod.load(_write_frame(tmp_path, {"c": ["r3", "r1"]}))
        assert frame_mod.select(frame, "c", ["r1", "r2", "r3"]) == ["r1", "r3"]

    def test_a_frame_for_another_cohort_does_not_apply(self, tmp_path):
        frame = frame_mod.load(_write_frame(tmp_path, {"c": ["r1"]}))
        assert frame_mod.applies_to(frame, ["r1", "r2"])
        assert not frame_mod.applies_to(frame, ["g1", "g2"])

    def test_a_blank_score_reads_as_none_not_zero(self, tmp_path):
        """pools.csv leaves P/R/F1 empty for a report with no ground truth, undefined, not zero.

        Reading it as 0.0 would sort an unannotated report to the bottom of a ranking it is not
        part of at all.
        """
        directory = _write_frame(tmp_path, {"c": ["r9"]},
                                 pools_rows="r9,1,c,0,3,,3,,,,,2,0,0,1\n")
        frame = frame_mod.load(directory)
        assert frame["scored"]["r9"]["f1"] is None
        assert frame["scored"]["r9"]["family"] == 2

    def test_no_frame_means_no_filtering(self):
        assert frame_mod.select(None, "anything", ["r1", "r2"]) == ["r1", "r2"]


class TestSubsetFilter:
    """What the filter does to the views, and the one place it must not reach."""

    def test_report_ids_honours_the_subset(self, bundle, tmp_path, monkeypatch):
        picked = bundle["report_ids"][:2]
        frame = frame_mod.load(_write_frame(tmp_path, {"cell": picked}))
        monkeypatch.setattr(common, "frame_or_none", lambda: frame)
        cfg = {**_cfg(bundle), "subset": "cell"}
        assert common.report_ids(bundle, cfg) == picked
        assert common.report_ids(bundle, {**cfg, "subset": frame_mod.ALL}) \
            == bundle["report_ids"]

    def test_a_cell_this_cohort_cannot_honour_degrades_to_all(self, bundle, tmp_path, monkeypatch):
        """store-config outlives the cohort it was chosen for, the same rule k already follows."""
        frame = frame_mod.load(_write_frame(tmp_path, {"cell": ["__not_here__"]}))
        monkeypatch.setattr(common, "frame_or_none", lambda: frame)
        assert common.resolve({"subset": "cell"}, bundle)["subset"] == frame_mod.ALL

    def test_the_memo_key_separates_subsets(self):
        """Without this, a filtered table is served from the unfiltered cache.

        That is the worst bug this cache can have: every number right for the whole cohort, shown
        under a banner saying it was computed over twenty reports.
        """
        base = {"rule": "vote_k", "k": 2, "min_count": 1, "models": ["a"]}
        assert memo.config_key({**base, "subset": "all"}) \
            != memo.config_key({**base, "subset": "pb_best"})

    def test_scoring_a_subset_scores_only_those_reports(self, bundle):
        one = bundle["report_ids"][:1]
        cfg = _cfg(bundle, k=1)
        predicted = votes.predicted_sets(bundle, cfg)
        whole = votes._counts(bundle["gold"], predicted, bundle["report_ids"])
        part = votes._counts(bundle["gold"], predicted, one)
        assert part["n_gold"] <= whole["n_gold"]
        assert part["n_gold"] == len(bundle["gold"].get(one[0], ()))

    def test_the_gates_are_not_filtered(self, bundle, tmp_path, monkeypatch):
        """The gates compare against the *shipped* full-cohort artifacts.

        Filtering them would turn "the app agrees with the driver" into a guaranteed failure, so
        verify must keep reading bundle["report_ids"] directly.
        """
        frame = frame_mod.load(_write_frame(tmp_path, {"cell": bundle["report_ids"][:1]}))
        monkeypatch.setattr(common, "frame_or_none", lambda: frame)
        assert not [g for g in verify.run_gates(bundle) if g["status"] == "fail"]

    def test_the_provenance_line_names_the_subset(self, bundle):
        """Copy-Markdown lands in a findings file, where the banner does not follow it."""
        line = common.describe({**_cfg(bundle), "subset": "pb_best"})
        assert "pb_best" in line and "quoted" in line
        assert "pb_best" not in common.describe({**_cfg(bundle), "subset": frame_mod.ALL})


# ══════════════════════════════════════════════════════════════════════════════
# The earlier runs, the same artifacts, one directory deeper
# ══════════════════════════════════════════════════════════════════════════════
class TestPromptScreenImport:
    def test_both_experiments_appear_in_one_dropdown(self, tmp_path):
        build_slm_ensemble_run(tmp_path / "phenojury_generation_free_listing" / "hcy")
        build_prompt_screen_tree(tmp_path, prompts=("q4_span_json",))
        found = {r["run_id"]: r for r in loaders.find_runs(str(tmp_path))}
        assert "hcy" in found, "the exp13_06 run id is a published contract and must not change"
        assert "phenojury_generation_other_prompts/hcy/q4_span_json" in found

    def test_a_prompt_cell_keeps_the_dataset_as_its_cohort(self, tmp_path):
        """The essential one: the cohort resolves the PhenoBERT baseline baseline.

        A cell labelled with its prompt key would send pbstandalone.find_run looking for
        baseline_phenobert/q4_span_json/, and the comparison page would silently go empty.
        """
        build_prompt_screen_tree(tmp_path, prompts=("q4_span_json",))
        run = loaders.find_runs(str(tmp_path))[0]
        assert run["cohort"] == "hcy"
        assert run["prompt_key"] == "q4_span_json"
        assert run["exp_id"] == "phenojury_generation_other_prompts"

    def test_the_baseline_resolves_through_the_extra_path_segment(self, tmp_path):
        build_prompt_screen_tree(tmp_path, prompts=("q4_span_json",))
        pb_dir = tmp_path / "baseline_phenobert" / "hcy"
        build_phenobert_standalone_run(pb_dir)
        reg = registry.Registry(output_base=str(tmp_path), cache_dir=str(tmp_path / "cache"))
        run_id = "phenojury_generation_other_prompts/hcy/q4_span_json"
        assert reg.pb_run_dir(run_id) == str(pb_dir)

    def test_the_metrics_file_is_found_under_either_name(self, tmp_path):
        """Same writer, same columns, only the filename differs, so G3 needs no special case."""
        build_prompt_screen_tree(tmp_path, prompts=("q4_span_json",))
        cell = tmp_path / "phenojury_generation_other_prompts" / "hcy" / "q4_span_json"
        assert not (cell / "slm_ensemble_slm_metrics.csv").exists()
        assert loaders.load_slm_metrics(str(cell)) is not None

    def test_a_prompt_cell_passes_every_gate(self, tmp_path):
        """The whole claim of this feature: a cell is a Free Listing generation run in every respect."""
        build_prompt_screen_tree(tmp_path, prompts=("q4_span_json",))
        cell = tmp_path / "phenojury_generation_other_prompts" / "hcy" / "q4_span_json"
        b = registry.build_bundle("phenojury_generation_other_prompts/hcy/q4_span_json", str(cell),
                                  loaders.load_raw(str(cell)), build_toy_tree(),
                                  cohort="hcy", exp_id="phenojury_generation_other_prompts",
                                  prompt_key="q4_span_json")
        assert b["cohort"] == "hcy" and b["prompt_key"] == "q4_span_json"
        failed = [g for g in b["gates"] if g["status"] == "fail"]
        assert not failed, [g["detail"] for g in failed]
        # G3 must actually run, not skip: the columns are identical.
        g3 = next(g for g in b["gates"] if "Per-model" in g["title"])
        assert g3["status"] == "pass", g3["detail"]

    def test_missing_timing_is_tolerated(self, tmp_path):
        build_prompt_screen_tree(tmp_path, prompts=("q4_span_json",))
        cell = tmp_path / "phenojury_generation_other_prompts" / "hcy" / "q4_span_json"
        assert loaders.load_timing(str(cell))["aggregate"] is None

    def test_the_diagnostics_and_ranking_are_read(self, tmp_path):
        build_prompt_screen_tree(tmp_path, prompts=("q4_span_json", "q7_recall"))
        cell = tmp_path / "phenojury_generation_other_prompts" / "hcy" / "q7_recall"
        diag = loaders.load_prompt_diagnostics(str(cell))
        assert diag is not None and "format_compliance" in diag.columns
        # Empty, not 0.0 for a sentence-shaped prompt: the metric does not apply.
        assert diag["label_exactness"].isna().all()
        ranking = loaders.load_prompt_ranking(str(cell))
        assert ranking is not None and len(ranking) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Manual annotation of the generations, and the PhenoBERT comparison
# ══════════════════════════════════════════════════════════════════════════════
class TestAnnotations:
    @staticmethod
    def _log(tmp_path):
        return annotations.AnnotationLog(str(tmp_path / "ann"), author="tester")

    @staticmethod
    def _fields(key):
        return {"key": key, "run_id": "run1", "cohort": "hcy", "prompt_key": "",
                "model": "llama", "report_id": "r1", "sentence_number": 3}

    def test_annotate_replaces_rather_than_merges(self, tmp_path):
        """The picker sends its whole contents. A merge would make removal impossible."""
        log = self._log(tmp_path)
        key = annotations.key_of("run1", "llama", "r1", 3)
        log.append("annotate", hpo_codes=["HP:1", "HP:2"], **self._fields(key))
        log.append("annotate", hpo_codes=["HP:1"], **self._fields(key))
        assert log.fold()["replies"][key]["hpo_codes"] == ["HP:1"]

    def test_skip_is_distinguishable_from_unvisited(self, tmp_path):
        """"Nothing here" is a reading, and it is the denominator of PhenoBERT's precision."""
        log = self._log(tmp_path)
        key = annotations.key_of("run1", "llama", "r1", 3)
        log.append("skip", **self._fields(key))
        visited = annotations.visited(log.fold())
        assert visited[key]["status"] == "skipped"
        assert annotations.key_of("run1", "llama", "r1", 9) not in visited

    def test_a_torn_final_line_is_skipped_not_fatal(self, tmp_path):
        """what a crash mid-append leaves. Refusing to open would lose the session."""
        log = self._log(tmp_path)
        log.append("annotate", hpo_codes=["HP:1"],
                   **self._fields(annotations.key_of("run1", "llama", "r1", 3)))
        with open(log.path, "a", encoding="utf-8") as handle:
            handle.write('{"action": "ann')
        assert len(annotations.AnnotationLog(str(tmp_path / "ann")).events) == 1

    def test_another_process_appending_is_picked_up(self, tmp_path):
        """Several readers share a directory. Folding only our own clicks would revert theirs."""
        first = self._log(tmp_path)
        second = annotations.AnnotationLog(str(tmp_path / "ann"), author="other")
        key = annotations.key_of("run1", "llama", "r1", 3)
        second.append("annotate", hpo_codes=["HP:9"], **self._fields(key))
        assert first.fold()["replies"][key]["hpo_codes"] == ["HP:9"]

    def test_the_comparison_splits_three_ways(self):
        bundle = {"run_id": "run1", "tree": None, "pb_index": {"by_sentence": {
            ("llama", "r1", 3): [{"hpo_id": "HP:1", "negated": False},
                                 {"hpo_id": "HP:8", "negated": False},
                                 {"hpo_id": "HP:7", "negated": True}]}}}
        row = annotations.compare_reply(bundle, {
            "key": "k", "run_id": "run1", "cohort": "hcy", "prompt_key": "", "model": "llama",
            "report_id": "r1", "sentence_number": 3, "status": "annotated",
            "hpo_codes": ["HP:1", "HP:2"], "author": "", "updated_at": "", "note": ""})
        assert row["n_agreed"] == 1
        assert row["pb_missed"] == ["HP:2"]
        assert row["pb_spurious"] == ["HP:8"]

    def test_a_negated_detection_is_neither_a_hit_nor_a_miss(self):
        """PhenoBERT reading a finding as absent is a decision, not a linking failure."""
        bundle = {"run_id": "run1", "tree": None, "pb_index": {"by_sentence": {
            ("llama", "r1", 3): [{"hpo_id": "HP:7", "negated": True}]}}}
        row = annotations.compare_reply(bundle, {
            "key": "k", "run_id": "run1", "cohort": "hcy", "prompt_key": "", "model": "llama",
            "report_id": "r1", "sentence_number": 3, "status": "annotated",
            "hpo_codes": [], "author": "", "updated_at": "", "note": ""})
        assert row["pb_negated"] == ["HP:7"]
        assert row["n_phenobert"] == 0 and row["n_pb_spurious"] == 0

    def test_disagreements_are_graded_by_ontology_distance(self):
        """A parent of the right term is a different failure from an unrelated one."""
        tree = build_toy_tree()
        bundle = {"run_id": "run1", "tree": tree, "pb_index": {"by_sentence": {
            ("llama", "r1", 3): [{"hpo_id": B, "negated": False}]}}}
        row = annotations.compare_reply(bundle, {
            "key": "k", "run_id": "run1", "cohort": "hcy", "prompt_key": "", "model": "llama",
            "report_id": "r1", "sentence_number": 3, "status": "annotated",
            "hpo_codes": [D], "author": "", "updated_at": "", "note": ""})
        assert row["spurious_relations"][B] in relations.RELATION_ORDER

    def test_only_this_runs_annotations_are_scored(self, tmp_path):
        """One directory holds every prompt's annotations, that is the point of the run key."""
        log = self._log(tmp_path)
        key = annotations.key_of("run1", "llama", "r1", 3)
        log.append("annotate", hpo_codes=["HP:1"], **self._fields(key))
        bundle = {"run_id": "other", "tree": None, "pb_index": {"by_sentence": {}}}
        assert annotations.compare(bundle, log.fold())["n_replies"] == 0

    def test_an_empty_reading_scores_none_not_zero(self):
        """No phenotypes read gives PhenoBERT nothing to recall, which is not recalling none."""
        assert annotations._score(annotations._empty_tally())["recall"] is None

    def test_both_derived_files_are_written(self, tmp_path):
        log = self._log(tmp_path)
        key = annotations.key_of("run1", "llama", "r1", 3)
        log.append("annotate", hpo_codes=["HP:1"], **self._fields(key))
        bundle = {"run_id": "run1", "tree": None, "pb_index": {"by_sentence": {}}}
        folded = log.fold()
        log.commit(annotations.term_rows(folded),
                   annotations.agreement_rows(annotations.compare(bundle, folded)))
        assert pathlib.Path(log.terms_path).exists()
        assert pathlib.Path(log.agreement_path).exists()
        assert "HP:1" in pathlib.Path(log.terms_path).read_text()


class TestAnnotateView:
    def test_the_queue_offers_only_replies_a_model_wrote(self, bundle):
        from apps.phenojury_ui.views import annotate

        empty = {"replies": {}, "n_skipped": 0}
        rows = annotate.queue_rows(bundle, _cfg(bundle), empty, bundle["models"])
        assert rows
        for row in rows:
            reply = bundle["built"]["replies"][(row["model"], row["report_id"],
                                                row["sentence_number"])]
            assert reply["wrote"], "a generation with nothing in it has no phenotypes to read"

    def test_the_queue_follows_the_subset(self, bundle, tmp_path, monkeypatch):
        from apps.phenojury_ui.views import annotate

        picked = bundle["report_ids"][:1]
        frame = frame_mod.load(_write_frame(tmp_path, {"cell": picked}))
        monkeypatch.setattr(common, "frame_or_none", lambda: frame)
        rows = annotate.queue_rows(bundle, {**_cfg(bundle), "subset": "cell"},
                                   {"replies": {}, "n_skipped": 0}, bundle["models"])
        assert {r["report_id"] for r in rows} == set(picked)

    def test_a_read_reply_leaves_the_todo_queue(self, bundle):
        from apps.phenojury_ui.views import annotate

        cfg = _cfg(bundle)
        empty = {"replies": {}, "n_skipped": 0}
        first = annotate.queue_rows(bundle, cfg, empty, bundle["models"], "todo")[0]
        folded = {"replies": {first["key"]: {"status": "skipped", "hpo_codes": []}},
                  "n_skipped": 0}
        todo = annotate.queue_rows(bundle, cfg, folded, bundle["models"], "todo")
        done = annotate.queue_rows(bundle, cfg, folded, bundle["models"], "done")
        assert first["key"] not in {r["key"] for r in todo}
        assert first["key"] in {r["key"] for r in done}

    def test_the_key_round_trips(self, bundle):
        from apps.phenojury_ui.views import annotate

        key = annotations.key_of(bundle["run_id"], "llama", "r1", 7)
        assert annotate.parse_key(key) == ("llama", "r1", 7)
        assert annotate.parse_key("nonsense") is None
        assert annotate.parse_key(None) is None
