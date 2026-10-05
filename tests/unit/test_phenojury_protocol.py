"""The PhenoJury protocol's selection machinery, over the ten-node toy ontology.

The three tests that matter most are the ones whose failure produces a *plausible* table:

* **The window unit must let jurors in adjacent sentences meet.** Centring each juror's window on
  its own sentences gives them disjoint buckets, so the window silently degenerates into the
  segment unit and the S5 row reads as "the window buys nothing".
* **The report-level count must be the maximum over units, not the sum.** Summing lets two jurors in
  two different sentences manufacture a vote of two that no segment ever cast, which is the report
  unit wearing a segment's name.
* **With one prompt, one reader, ``exact`` and ``report``, this must be an earlier exploratory run's procedure.** That
  is the whole warrant for generalising the juror: if the special case drifts, the published
  0.5970 / 0.6057 / 0.5387 row is no longer the thing this experiment extends.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "experiments" / "05_phenojury" / "protocol"))

from hpo_extraction.evaluation.metrics import micro_prf, normalise_pair  # noqa: E402
from fixtures.toy_ontology import A, B, C, D, F, M  # noqa: E402

from protocol import (  # noqa: E402
    Arm, Juror, anchored_generation_rate, backward_elimination, choose_pair, evaluate_conditions,
    load_curated_annotations, load_grid, max_counts, pack, packings_for, predict, predict_all_k,
    reduce_specific, score, select_config, sets_at_k,
)

pytestmark = pytest.mark.unit


def juror(model, prompt="p0", normaliser="dictionary"):
    return Juror(model, prompt, normaliser)


J0, J1, J2 = juror("m0"), juror("m1"), juror("m2")


def grid(**per_model):
    """``{Juror: {report: {sentence: {term}}}}`` from a compact literal."""
    return {juror(model): data for model, data in per_model.items()}


# ── Packing and the vote units ───────────────────────────────────────────────

class TestUnits:
    def test_report_unit_merges_every_sentence_into_one_bucket(self, toy_view):
        per_juror = grid(m0={"r1": {0: {C}, 5: {D}}})
        packed = pack([J0], per_juror, ["r1"], "report", "exact", toy_view)
        assert packed["r1"] == {0: {0: frozenset({C, D})}}

    def test_segment_unit_keeps_one_bucket_per_sentence(self, toy_view):
        per_juror = grid(m0={"r1": {0: {C}, 5: {D}}})
        packed = pack([J0], per_juror, ["r1"], "segment", "exact", toy_view)
        assert packed["r1"] == {0: {0: frozenset({C})}, 5: {0: frozenset({D})}}

    def test_window_lets_jurors_in_adjacent_sentences_meet(self, toy_view):
        """The bug this test exists for: centring on each juror's own sentences gives them
        disjoint buckets, and the window silently becomes the segment unit."""
        per_juror = grid(m0={"r1": {5: {C}}}, m1={"r1": {6: {C}}})
        packed = pack([J0, J1], per_juror, ["r1"], "window", "exact", toy_view)
        counts = max_counts(packed, ["r1"], frozenset({0, 1}))
        assert counts["r1"][C] == 2

    def test_the_segment_unit_keeps_those_same_two_jurors_apart(self, toy_view):
        """The contrast that makes the window a distinct rule rather than a relabelling."""
        per_juror = grid(m0={"r1": {5: {C}}}, m1={"r1": {6: {C}}})
        packed = pack([J0, J1], per_juror, ["r1"], "segment", "exact", toy_view)
        assert max_counts(packed, ["r1"], frozenset({0, 1}))["r1"][C] == 1

    def test_window_does_not_reach_two_sentences_away(self, toy_view):
        per_juror = grid(m0={"r1": {5: {C}}}, m1={"r1": {7: {C}}})
        packed = pack([J0, J1], per_juror, ["r1"], "window", "exact", toy_view)
        assert max_counts(packed, ["r1"], frozenset({0, 1}))["r1"][C] == 1

    def test_an_unknown_unit_or_rule_is_refused(self, toy_view):
        with pytest.raises(ValueError, match="unknown unit"):
            pack([J0], grid(m0={}), [], "paragraph", "exact", toy_view)
        with pytest.raises(ValueError, match="unknown rule"):
            pack([J0], grid(m0={}), [], "report", "majority", toy_view)


class TestMaxCounts:
    def test_votes_are_the_maximum_over_units_never_the_sum(self, toy_view):
        """Two jurors naming C in two different sentences have never agreed in any one segment.

        Summing would report two votes for a term no segment ever carried at two, the report unit
        wearing a segment's name, and undetectable in any output table.
        """
        per_juror = grid(m0={"r1": {0: {C}}}, m1={"r1": {9: {C}}})
        packed = pack([J0, J1], per_juror, ["r1"], "segment", "exact", toy_view)
        assert max_counts(packed, ["r1"], frozenset({0, 1}))["r1"][C] == 1

    def test_agreement_inside_one_segment_does_count(self, toy_view):
        per_juror = grid(m0={"r1": {4: {C}}}, m1={"r1": {4: {C}}})
        packed = pack([J0, J1], per_juror, ["r1"], "segment", "exact", toy_view)
        assert max_counts(packed, ["r1"], frozenset({0, 1}))["r1"][C] == 2

    def test_a_juror_outside_the_subset_casts_no_vote(self, toy_view):
        per_juror = grid(m0={"r1": {0: {C}}}, m1={"r1": {0: {C}}})
        packed = pack([J0, J1], per_juror, ["r1"], "report", "exact", toy_view)
        assert max_counts(packed, ["r1"], frozenset({0}))["r1"][C] == 1

    def test_sets_at_k_is_monotone_decreasing_in_k(self, toy_view):
        per_juror = grid(m0={"r1": {0: {C, D}}}, m1={"r1": {0: {C}}}, m2={"r1": {0: {C, F}}})
        packed = pack([J0, J1, J2], per_juror, ["r1"], "report", "exact", toy_view)
        counts = max_counts(packed, ["r1"], frozenset({0, 1, 2}))
        sizes = [len(sets_at_k(counts, k)["r1"]) for k in (1, 2, 3)]
        assert sizes == sorted(sizes, reverse=True)
        assert sets_at_k(counts, 3)["r1"] == {C}


class TestClosureRules:
    def test_closure_packs_the_ancestor_closure_of_each_juror_set(self, toy_view):
        per_juror = grid(m0={"r1": {0: {C}}})
        packed = pack([J0], per_juror, ["r1"], "report", "closure", toy_view)
        assert packed["r1"][0][0] == frozenset({C, B, A})

    def test_closure_lets_two_specificities_agree_on_their_shared_ancestor(self, toy_view):
        """The mechanism the rule was proposed for, one juror writes C, the next writes D."""
        per_juror = grid(m0={"r1": {0: {C}}}, m1={"r1": {0: {D}}})
        packed = pack([J0, J1], per_juror, ["r1"], "report", "closure", toy_view)
        counts = max_counts(packed, ["r1"], frozenset({0, 1}))
        assert counts[("r1")][B] == 2 and counts["r1"][C] == 1

    def test_closure_reduced_keeps_only_the_most_specific_accepted_terms(self, toy_view):
        per_juror = grid(m0={"r1": {0: {C}}}, m1={"r1": {0: {D}}})
        packed = pack([J0, J1], per_juror, ["r1"], "report", "closure_reduced", toy_view)
        assert predict_all_k(packed, ["r1"], frozenset({0, 1}), "closure_reduced",
                             toy_view)[2]["r1"] == {B}

    def test_reduce_specific_drops_a_term_that_is_an_ancestor_of_another(self, toy_view):
        assert reduce_specific({A, B, C}, toy_view) == {C}

    def test_reduce_specific_keeps_incomparable_terms(self, toy_view):
        assert reduce_specific({C, F}, toy_view) == {C, F}

    def test_a_multi_parent_node_pulls_in_both_chains(self, toy_view):
        per_juror = grid(m0={"r1": {0: {M}}})
        packed = pack([J0], per_juror, ["r1"], "report", "closure", toy_view)
        assert packed["r1"][0][0] == frozenset({M, B, A, "HP:0011842", "HP:0000924"})


# ── The an earlier exploratory run special case ────────────────────────────────────────────────

def brute_force_vote_k(per_juror, jurors, report_ids, k):
    """An earlier exploratory run's rule, written out: tally each juror's whole-report set, threshold at k."""
    out = {}
    for rid in report_ids:
        tally: Counter = Counter()
        for j in jurors:
            terms = set()
            for sentence_terms in per_juror[j].get(rid, {}).values():
                terms |= sentence_terms
            tally.update(terms)
        out[rid] = {h for h, n in tally.items() if n >= k}
    return out


def test_one_prompt_one_reader_exact_report_is_exp13_23s_own_rule(toy_view):
    """The warrant for generalising the juror: the special case must not merely resemble the
    published procedure, it must be it."""
    per_juror = grid(
        m0={"r1": {0: {C}, 1: {D}}, "r2": {0: {F}}},
        m1={"r1": {0: {C, D}}, "r2": {3: {F, C}}},
        m2={"r1": {2: {C}}, "r2": {0: {C}}},
    )
    jurors = [J0, J1, J2]
    ids = ["r1", "r2"]
    packed = pack(jurors, per_juror, ids, "report", "exact", toy_view)
    generalised = predict_all_k(packed, ids, frozenset({0, 1, 2}), "exact", toy_view)
    for k in (1, 2, 3):
        assert generalised[k] == brute_force_vote_k(per_juror, jurors, ids, k), f"k={k}"


# ── Selection ────────────────────────────────────────────────────────────────

def gold_of(**per_report):
    return {rid: set(terms) for rid, terms in per_report.items()}


class TestSelection:
    def test_select_config_prefers_the_threshold_that_scores_best_inside_the_folds(self, toy_view):
        # m2 is pure noise, so k=2 is the threshold that filters it out.
        per_juror = grid(
            m0={"r1": {0: {C}}, "r2": {0: {D}}},
            m1={"r1": {0: {C}}, "r2": {0: {D}}},
            m2={"r1": {0: {F}}, "r2": {0: {F}}},
        )
        packings = {("exact", "report"): pack([J0, J1, J2], per_juror, ["r1", "r2"],
                                              "report", "exact", toy_view)}
        gold = gold_of(r1=[C], r2=[D])
        config, mean = select_config(packings, gold, [["r1"], ["r2"]], frozenset({0, 1, 2}),
                                     toy_view, normalise_pair, micro_prf, ["exact"], ["report"])
        assert config.k == 2 and mean == pytest.approx(1.0)

    def test_ties_go_to_the_earlier_rule_then_the_earlier_unit_then_the_lower_k(self, toy_view):
        """A tie-break that drifts makes two runs of the same procedure disagree for no reason.
        This is an earlier exploratory run's order, kept so the special case reproduces it."""
        per_juror = grid(m0={"r1": {0: {C}}})
        packings = {(rule, unit): pack([J0], per_juror, ["r1"], unit, rule, toy_view)
                    for rule in ("exact", "closure_reduced") for unit in ("report", "segment")}
        gold = gold_of(r1=[C])
        config, _mean = select_config(packings, gold, [["r1"]], frozenset({0}), toy_view,
                                      normalise_pair, micro_prf,
                                      ["exact", "closure_reduced"], ["report", "segment"])
        assert (config.rule, config.unit, config.k) == ("exact", "report", 1)

    def test_backward_elimination_drops_a_juror_that_only_adds_noise(self, toy_view):
        per_juror = grid(
            m0={"r1": {0: {C}}, "r2": {0: {D}}},
            m1={"r1": {0: {C}}, "r2": {0: {D}}},
            m2={"r1": {0: {F, M}}, "r2": {0: {F, M}}},
        )
        packings = {("exact", "report"): pack([J0, J1, J2], per_juror, ["r1", "r2"],
                                              "report", "exact", toy_view)}
        gold = gold_of(r1=[C], r2=[D])
        subset, config, path = backward_elimination(
            packings, gold, [["r1"], ["r2"]], frozenset({0, 1, 2}), toy_view, normalise_pair,
            micro_prf, ["exact"], ["report"], epsilon=0.01)
        assert 2 not in subset
        assert config.k >= 1
        assert [step["size"] for step in path] == [3, 2, 1]

    def test_epsilon_buys_the_smaller_jury_when_the_larger_is_barely_better(self, toy_view):
        """Without the tolerance the procedure chases a 0.001 inner gain bought with more jurors."""
        per_juror = grid(
            m0={"r1": {0: {C}}, "r2": {0: {D}}},
            m1={"r1": {0: {C}}, "r2": {0: {D}}},
            m2={"r1": {0: {C}}, "r2": {0: {D}}},
        )
        packings = {("exact", "report"): pack([J0, J1, J2], per_juror, ["r1", "r2"],
                                              "report", "exact", toy_view)}
        gold = gold_of(r1=[C], r2=[D])
        subset, _config, _path = backward_elimination(
            packings, gold, [["r1"], ["r2"]], frozenset({0, 1, 2}), toy_view, normalise_pair,
            micro_prf, ["exact"], ["report"], epsilon=0.01)
        # Every juror is identical, so all three juries score the same and the smallest wins.
        assert len(subset) == 1

    def test_the_elimination_path_is_recorded_so_the_choice_is_auditable(self, toy_view):
        per_juror = grid(m0={"r1": {0: {C}}}, m1={"r1": {0: {D}}})
        packings = {("exact", "report"): pack([J0, J1], per_juror, ["r1"], "report", "exact",
                                              toy_view)}
        _subset, _config, path = backward_elimination(
            packings, gold_of(r1=[C]), [["r1"]], frozenset({0, 1}), toy_view, normalise_pair,
            micro_prf, ["exact"], ["report"], epsilon=0.01)
        assert all({"subset", "size", "rule", "unit", "k", "inner_f1"} <= set(s) for s in path)


# ── The (prompt, normaliser) pair, selected inside the folds ─────────────────
#
# The reason these exist: the pair used to be fixed once on a development split that holds the
# held-out reports of outer folds 1-4 of the pooled repetition, so ~80 % of the reports behind the
# main had helped choose the pair that scored them. Every test below is a way that leak, or a
# quieter version of it, could come back.

PAIR_A, PAIR_B = ("pa", "dictionary"), ("pb", "dictionary")


def two_pair_grid(a: dict, b: dict, models=("m0", "m1")):
    """Jurors under two pairs; *a* and *b* are ``{report: {sentence: {term}}}`` per pair, shared by
    every model of that pair (identical jurors, so the vote is the pair's, not a model's)."""
    jurors, per_juror = [], {}
    for (prompt, normaliser), data in ((PAIR_A, a), (PAIR_B, b)):
        for model in models:
            j = Juror(model, prompt, normaliser)
            jurors.append(j)
            per_juror[j] = data
    return sorted(jurors), per_juror


def fold(eval_ids, inner, repetition=0, outer_fold=0):
    train = sorted({rid for part in inner for rid in part})
    return {"repetition": repetition, "outer_fold": outer_fold, "eval_ids": list(eval_ids),
            "train_ids": train, "inner_folds": [list(part) for part in inner]}


def run_conditions(arms, jurors, per_juror, gold, folds, view, n_workers=1, pairs=(PAIR_A, PAIR_B)):
    ids = sorted(gold)
    return evaluate_conditions(arms, list(pairs), jurors, per_juror, gold, folds, view, normalise_pair,
                         micro_prf, ["exact", "closure_reduced"], ["report", "segment"], 0.01,
                         ids, n_workers=n_workers)


class TestPairSelectionInFold:
    def test_a_pair_that_wins_only_on_the_held_out_report_is_never_chosen(self, toy_view):
        """The leak itself. Pair B is perfect on r3 and wrong on the training reports. Pair A is
        the reverse. A pair chosen by looking at r3 would be B. The folds must choose A."""
        jurors, per_juror = two_pair_grid(
            a={"r1": {0: {C}}, "r2": {0: {D}}, "r3": {0: {M}}},
            b={"r1": {0: {M}}, "r2": {0: {M}}, "r3": {0: {F}}},
        )
        gold = gold_of(r1=[C], r2=[D], r3=[F])
        result = run_conditions([Arm("full_pool")], jurors, per_juror, gold,
                          [fold(["r3"], [["r1"], ["r2"]])], toy_view)["full_pool"]
        assert result.choices[0]["prompt"] == "pa"
        assert result.assignment["r3"].pair == PAIR_A
        # ...and it scores r3 with A's (wrong) answer. That is what an honest estimate looks like.
        assert result.pooled["r3"] == {M}

    def test_each_outer_fold_chooses_on_its_own_training_split(self, toy_view):
        """Two outer folds whose training splits favour different pairs must choose differently,
        and each held-out report is scored under ITS fold's pair."""
        jurors, per_juror = two_pair_grid(
            a={"r1": {0: {C}}, "r2": {0: {D}}, "r3": {0: {M}}, "r4": {0: {M}}},
            b={"r1": {0: {M}}, "r2": {0: {M}}, "r3": {0: {F}}, "r4": {0: {C}}},
        )
        gold = gold_of(r1=[C], r2=[D], r3=[F], r4=[C])
        folds = [fold(["r3"], [["r1"], ["r2"]], outer_fold=0),
                 fold(["r1"], [["r3"], ["r4"]], outer_fold=1)]
        result = run_conditions([Arm("full_pool")], jurors, per_juror, gold, folds,
                          toy_view)["full_pool"]
        assert [c["prompt"] for c in result.choices] == ["pa", "pb"]
        assert result.assignment["r3"].pair == PAIR_A
        assert result.assignment["r1"].pair == PAIR_B

    def test_only_repetition_zero_is_pooled_but_every_fold_is_a_choice(self, toy_view):
        jurors, per_juror = two_pair_grid(a={"r1": {0: {C}}, "r2": {0: {D}}},
                                          b={"r1": {0: {C}}, "r2": {0: {D}}})
        gold = gold_of(r1=[C], r2=[D])
        folds = [fold(["r1"], [["r2"]], repetition=0), fold(["r2"], [["r1"]], repetition=0, outer_fold=1),
                 fold(["r1"], [["r2"]], repetition=1)]
        result = run_conditions([Arm("full_pool")], jurors, per_juror, gold, folds,
                          toy_view)["full_pool"]
        assert len(result.choices) == 3
        assert set(result.pooled) == {"r1", "r2"}
        # Every pair's inner score, in every fold -- the audit trail for "this pair won by X".
        assert len(result.inner_scores) == 3 * 2
        assert sum(r["chosen"] for r in result.inner_scores) == 3

    def test_with_one_pair_it_is_the_fixed_pair_procedure(self, toy_view):
        """Regression against the procedure it replaces: one candidate pair must reproduce the
        per-fold select_config + predict it used to run, choice for choice."""
        jurors, per_juror = two_pair_grid(
            a={"r1": {0: {C}, 1: {F}}, "r2": {0: {D}}, "r3": {2: {C, M}}},
            b={},
        )
        gold = gold_of(r1=[C], r2=[D], r3=[C])
        folds = [fold(["r3"], [["r1"], ["r2"]]), fold(["r1"], [["r2"], ["r3"]], outer_fold=1)]
        result = run_conditions([Arm("full_pool")], jurors, per_juror, gold, folds, toy_view,
                          pairs=(PAIR_A,))["full_pool"]
        pool = [j for j in jurors if (j.prompt, j.normaliser) == PAIR_A]
        packings = packings_for(pool, per_juror, sorted(gold), ["exact", "closure_reduced"],
                                ["report", "segment"], toy_view)
        everyone = frozenset(range(len(pool)))
        for row, choice in zip(folds, result.choices):
            config, _mean = select_config(packings, gold, row["inner_folds"], everyone, toy_view,
                                          normalise_pair, micro_prf, ["exact", "closure_reduced"],
                                          ["report", "segment"])
            assert (choice["rule"], choice["unit"], choice["k"]) == tuple(config)
            expected = predict(packings, config.rule, config.unit, row["eval_ids"], everyone,
                               config.k, toy_view)
            assert {r: result.pooled[r] for r in row["eval_ids"]} == expected

    def test_a_single_juror_condition_selects_only_the_pair(self, toy_view):
        jurors, per_juror = two_pair_grid(a={"r1": {0: {C}}, "r2": {0: {D}}},
                                          b={"r1": {0: {M}}, "r2": {0: {M}}})
        gold = gold_of(r1=[C], r2=[D])
        arm = Arm("single:m0", models=("m0",), rules=("exact",), units=("report",))
        result = run_conditions([arm], jurors, per_juror, gold, [fold(["r2"], [["r1"]])],
                          toy_view)["single:m0"]
        choice = result.choices[0]
        assert (choice["prompt"], choice["rule"], choice["unit"], choice["k"]) == \
            ("pa", "exact", "report", 1)
        assert choice["subset"] == ["m0"]

    def test_a_fixed_jury_is_fixed_by_model_name_under_every_pair(self, toy_view):
        """Pool indices mean a different juror under each pair, so a jury is named by model."""
        jurors, per_juror = two_pair_grid(a={"r1": {0: {C}}, "r2": {0: {D}}},
                                          b={"r1": {0: {C}}, "r2": {0: {D}}},
                                          models=("m0", "m1", "m2"))
        gold = gold_of(r1=[C], r2=[D])
        result = run_conditions([Arm("lomo:m1", models=("m0", "m2"))], jurors, per_juror, gold,
                          [fold(["r2"], [["r1"]])], toy_view)["lomo:m1"]
        assert result.choices[0]["subset"] == ["m0", "m2"]

    def test_a_forced_prompt_still_lets_the_folds_choose_its_normaliser(self, toy_view):
        jurors, per_juror = two_pair_grid(a={"r1": {0: {C}}, "r2": {0: {D}}},
                                          b={"r1": {0: {C}}, "r2": {0: {D}}})
        gold = gold_of(r1=[C], r2=[D])
        result = run_conditions([Arm("prompt:pb", prompts=("pb",))], jurors, per_juror, gold,
                          [fold(["r2"], [["r1"]])], toy_view)["prompt:pb"]
        assert {r["prompt"] for r in result.inner_scores} == {"pb"}

    def test_ties_go_to_the_pair_listed_first(self):
        assert choose_pair({PAIR_B: 0.5, PAIR_A: 0.5}, [PAIR_A, PAIR_B]) == PAIR_A
        assert choose_pair({PAIR_B: 0.5, PAIR_A: 0.5}, [PAIR_B, PAIR_A]) == PAIR_B
        assert choose_pair({PAIR_B: 0.6, PAIR_A: 0.5}, [PAIR_A, PAIR_B]) == PAIR_B

    def test_a_condition_no_pair_can_seat_is_an_error_not_an_empty_row(self, toy_view):
        jurors, per_juror = two_pair_grid(a={"r1": {0: {C}}}, b={"r1": {0: {C}}})
        with pytest.raises(ValueError, match="no pair holds"):
            run_conditions([Arm("single:ghost", models=("ghost",))], jurors, per_juror,
                     gold_of(r1=[C]), [fold(["r1"], [["r1"]])], toy_view)

    def test_forked_workers_give_the_serial_answer(self, toy_view):
        jurors, per_juror = two_pair_grid(
            a={"r1": {0: {C}}, "r2": {0: {D}}, "r3": {0: {M}}, "r4": {0: {M}}},
            b={"r1": {0: {M}}, "r2": {0: {M}}, "r3": {0: {F}}, "r4": {0: {C}}},
        )
        gold = gold_of(r1=[C], r2=[D], r3=[F], r4=[C])
        folds = [fold(["r3"], [["r1"], ["r2"]]), fold(["r1"], [["r3"], ["r4"]], outer_fold=1)]
        arms = [Arm("phenojury", select_jury=True), Arm("full_pool")]
        serial = run_conditions(arms, jurors, per_juror, gold, folds, toy_view, n_workers=1)
        forked = run_conditions(arms, jurors, per_juror, gold, folds, toy_view, n_workers=2)
        for label in ("phenojury", "full_pool"):
            assert serial[label].pooled == forked[label].pooled
            assert serial[label].choices == forked[label].choices


# ── Scoring ──────────────────────────────────────────────────────────────────

def test_score_aligns_on_the_given_report_ids_and_normalises_both_sides(toy_view):
    pred = {"r1": {C}, "r2": {F}}
    gold = gold_of(r1=[C], r2=[C])
    golds, preds = score(pred, gold, ["r1", "r2"], toy_view, normalise_pair)
    assert golds == [{C}, {C}] and preds == [{C}, {F}]
    assert micro_prf(golds, preds)[2] == pytest.approx(0.5)


def test_a_report_absent_from_the_prediction_scores_as_an_empty_prediction(toy_view):
    golds, preds = score({}, gold_of(r1=[C]), ["r1"], toy_view, normalise_pair)
    assert preds == [set()] and micro_prf(golds, preds)[1] == 0.0


# ── Loading ──────────────────────────────────────────────────────────────────

def write_cell(root, cohort, prompt, model, normaliser, rows):
    import json
    cell = root / cohort / prompt
    cell.mkdir(parents=True, exist_ok=True)
    path = cell / f"detections_{model}__{normaliser}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


class TestLoadGrid:
    def test_a_juror_is_one_model_prompt_normaliser_triple(self, tmp_path):
        write_cell(tmp_path, "hcy", "p0", "llama", "dictionary",
                   [{"report_id": "r1", "sentence_number": 0, "hpo_id": C, "count": 1}])
        write_cell(tmp_path, "hcy", "p0", "llama", "sapbert",
                   [{"report_id": "r1", "sentence_number": 0, "hpo_id": D, "count": 1}])
        jurors, per_juror, _audit = load_grid(tmp_path, "hcy", ["p0"],
                                              ["dictionary", "sapbert"])
        assert len(jurors) == 2
        assert {j.normaliser for j in jurors} == {"dictionary", "sapbert"}
        assert per_juror[Juror("llama", "p0", "dictionary")] == {"r1": {0: {C}}}

    def test_a_missing_cell_becomes_an_audit_row_rather_than_an_exception(self, tmp_path):
        write_cell(tmp_path, "hcy", "p0", "llama", "dictionary",
                   [{"report_id": "r1", "sentence_number": 0, "hpo_id": C, "count": 1}])
        jurors, _per_juror, audit = load_grid(tmp_path, "hcy", ["p0", "q7"],
                                              ["dictionary"])
        assert len(jurors) == 1
        missing = [row for row in audit if not row["found"]]
        assert len(missing) == 1 and missing[0]["prompt"] == "q7"

    def test_a_reader_named_twice_still_yields_one_juror(self, tmp_path):
        """`normalisers` plus `reference_normaliser` is the obvious way to name one reader twice.

        A juror admitted twice occupies two bit positions and casts two votes, so every threshold
        is quietly halved, and nothing in any output table would show it.
        """
        write_cell(tmp_path, "hcy", "p0", "llama", "dictionary",
                   [{"report_id": "r1", "sentence_number": 0, "hpo_id": C, "count": 1}])
        jurors, per_juror, _audit = load_grid(
            tmp_path, "hcy", ["p0", "p0"], ["dictionary", "dictionary"])
        assert jurors == [Juror("llama", "p0", "dictionary")]
        assert len(per_juror) == 1

    def test_a_duplicated_juror_would_have_doubled_its_own_vote(self, tmp_path, toy_view):
        """The consequence, stated as a test so the guard above has a reason attached."""
        write_cell(tmp_path, "hcy", "p0", "llama", "dictionary",
                   [{"report_id": "r1", "sentence_number": 0, "hpo_id": C, "count": 1}])
        jurors, per_juror, _audit = load_grid(
            tmp_path, "hcy", ["p0"], ["dictionary", "dictionary"])
        packed = pack(jurors, per_juror, ["r1"], "report", "exact", toy_view)
        counts = max_counts(packed, ["r1"], frozenset(range(len(jurors))))
        assert counts["r1"][C] == 1

    def test_jurors_are_sorted_so_the_bit_order_is_reproducible(self, tmp_path):
        for model in ("phi4", "apertus", "llama"):
            write_cell(tmp_path, "hcy", "p0", model, "dictionary",
                       [{"report_id": "r1", "sentence_number": 0, "hpo_id": C, "count": 1}])
        jurors, _p, _a = load_grid(tmp_path, "hcy", ["p0"], ["dictionary"])
        assert [j.model for j in jurors] == ["apertus", "llama", "phi4"]


class TestEvidenceGenerationRate:
    def test_a_term_named_in_the_annotated_segment_counts_and_elsewhere_does_not(self, toy_view):
        per_juror = grid(m0={"r1": {3: {C}, 9: {D}}})
        packed = pack([J0], per_juror, ["r1"], "segment", "exact", toy_view)
        annotations = [{"report_id": "r1", "segment_idx": 3, "hpo_id": C, "trigger": "fits"},
                       {"report_id": "r1", "segment_idx": 3, "hpo_id": D, "trigger": "small"}]
        stats = anchored_generation_rate(packed, annotations, frozenset({0}), toy_view)
        assert stats["anchored_generation_rate"] == pytest.approx(0.5)
        # D was written, just in the wrong segment, which is the distinction this makes.
        assert stats["report_level_generation_rate"] == pytest.approx(1.0)

    def test_one_pair_annotated_in_two_segments_counts_once(self, toy_view):
        """The unit is the annotated pair, not the annotation row.

        On curated_ground_truth_2026-09-12 that is 1439 in-ground truth rows over 1146 pairs, so counting rows
        would weight the terms a curator happened to mark twice and move the denominator by a
        quarter.
        """
        per_juror = grid(m0={"r1": {3: {C}}})
        packed = pack([J0], per_juror, ["r1"], "segment", "exact", toy_view)
        annotations = [{"report_id": "r1", "segment_idx": 3, "hpo_id": C, "trigger": "fits"},
                       {"report_id": "r1", "segment_idx": 8, "hpo_id": C, "trigger": "seizure"}]
        stats = anchored_generation_rate(packed, annotations, frozenset({0}), toy_view)
        assert stats["n_gold_pairs"] == 1
        assert stats["n_annotation_rows"] == 2
        # Found in one of its two annotated segments, so the pair is located.
        assert stats["anchored_generation_rate"] == pytest.approx(1.0)

    def test_an_identifier_outside_the_release_counts_against_the_rate(self, toy_view):
        """It cannot be found by anybody, so it stays in the denominator and is reported."""
        from fixtures.toy_ontology import HALLUCINATION
        per_juror = grid(m0={"r1": {3: {C}}})
        packed = pack([J0], per_juror, ["r1"], "segment", "exact", toy_view)
        annotations = [{"report_id": "r1", "segment_idx": 3, "hpo_id": C, "trigger": "fits"},
                       {"report_id": "r1", "segment_idx": 3, "hpo_id": HALLUCINATION,
                        "trigger": "?"}]
        stats = anchored_generation_rate(packed, annotations, frozenset({0}), toy_view)
        assert stats["n_gold_pairs"] == 2 and stats["n_unresolvable"] == 1
        assert stats["anchored_generation_rate"] == pytest.approx(0.5)

    def test_a_tolerance_of_one_admits_the_neighbouring_segment(self, toy_view):
        per_juror = grid(m0={"r1": {4: {C}}})
        packed = pack([J0], per_juror, ["r1"], "segment", "exact", toy_view)
        annotations = [{"report_id": "r1", "segment_idx": 3, "hpo_id": C, "trigger": "fits"}]
        strict = anchored_generation_rate(packed, annotations, frozenset({0}), toy_view, 0)
        loose = anchored_generation_rate(packed, annotations, frozenset({0}), toy_view, 1)
        assert strict["anchored_generation_rate"] == 0.0
        assert loose["anchored_generation_rate"] == 1.0

    def test_coordinate_drift_shows_up_as_a_low_coordinate_ok_rate(self, toy_view):
        """If the ground truth's segment_idx and the run's sentence_number index different segmentations,
        the located rate is meaningless, so the check is reported, never assumed."""
        per_juror = grid(m0={"r1": {0: {C}, 1: {C}}})
        packed = pack([J0], per_juror, ["r1"], "segment", "exact", toy_view)
        annotations = [{"report_id": "r1", "segment_idx": 99, "hpo_id": C, "trigger": "fits"}]
        stats = anchored_generation_rate(packed, annotations, frozenset({0}), toy_view)
        assert stats["coordinate_ok_rate"] == 0.0


def test_load_curated_annotations_skips_rows_with_no_located_segment(tmp_path):
    path = tmp_path / "ann.csv"
    path.write_text(
        "patient_id,segment_idx,hpo_id,trigger_word\n"
        f"r1,3,{C},fits\n"
        f"r1,,{D},small\n"
        f"r2,7,{F},ataxic\n", encoding="utf-8")
    rows = load_curated_annotations(path)
    assert [(r["report_id"], r["segment_idx"]) for r in rows] == [("r1", 3), ("r2", 7)]


def test_load_curated_annotations_drops_rows_the_curation_excluded(tmp_path):
    """The sidecar is the curation RECORD, not the ground truth.

    On the shipped export 210 of 1649 rows are ``in_gold=0``, annotations the policy
    dropped, each with an ``exclude_reason``. Counting them puts terms in the located denominator
    that are not ground truth at all, which understates the rate by inflating what had to be found.
    """
    path = tmp_path / "ann.csv"
    path.write_text(
        "patient_id,hpo_code,segment_idx,trigger_word,in_gold,exclude_reason\n"
        f"r1,{C},3,fits,1,\n"
        f"r1,{D},4,small,0,family\n"
        f"r2,{F},7,ataxic,1,\n", encoding="utf-8")
    assert [r["hpo_id"] for r in load_curated_annotations(path)] == [C, F]
    assert [r["hpo_id"] for r in load_curated_annotations(path, gold_only=False)] == [C, D, F]


def test_load_curated_annotations_reads_the_shipped_hpo_code_spelling(tmp_path):
    """The sidecar says ``hpo_code``. Other tables say ``hpo_id``. Both are real, both are read."""
    path = tmp_path / "ann.csv"
    path.write_text(f"patient_id,hpo_code,segment_idx,trigger_word\nr1,{C},3,fits\n",
                    encoding="utf-8")
    assert load_curated_annotations(path)[0]["hpo_id"] == C


def test_load_curated_annotations_on_a_missing_file_is_empty_not_an_error(tmp_path):
    assert load_curated_annotations(tmp_path / "nope.csv") == []


# ── The stages added for chapter 5's specified tables ────────────────────────
# These live in `run.py`, not `protocol.py` because they orchestrate, not compute,
# but the two with real semantics are pure functions and are fixed here.

import run as run_mod  # noqa: E402


def annotation_csv(tmp_path, rows, *, with_source=True):
    """A minimal curated-annotation sidecar, in the ground-truth build's own column spelling."""
    import csv as _csv

    path = tmp_path / "hcy_curated_annotations.csv"
    fields = ["patient_id", "hpo_code"] + (["source"] if with_source else [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


class TestGoldProvenance:
    """T5.6's second block. The semantics that matter are about what is KEPT."""

    def test_a_pair_recorded_only_by_an_excluded_source_is_dropped(self, tmp_path, toy_view):
        path = annotation_csv(tmp_path, [
            {"patient_id": "R1", "hpo_code": A, "source": "new"},
            {"patient_id": "R1", "hpo_code": B, "source": "prior_annotation"},
        ])
        restricted, stats = run_mod.gold_provenance(path, {"R1": {A, B}}, toy_view, {"new"})
        assert restricted == {"R1": {B}}
        assert stats["n_removed"] == 1
        assert stats["n_gold_pairs"] == 2

    def test_a_pair_any_independent_source_also_recorded_is_kept(self, tmp_path, toy_view):
        """Two files recording one annotation, one of them independent, is an independently
        attested pair. Dropping it would overstate how much of the ground truth is at risk."""
        path = annotation_csv(tmp_path, [
            {"patient_id": "R1", "hpo_code": A, "source": "new"},
            {"patient_id": "R1", "hpo_code": A, "source": "daphne"},
        ])
        restricted, stats = run_mod.gold_provenance(path, {"R1": {A}}, toy_view, {"new"})
        assert restricted == {"R1": {A}}
        assert stats["n_removed"] == 0

    def test_a_restriction_that_removes_nothing_is_reported_as_such(self, tmp_path, toy_view):
        """The likely real outcome on this ground truth, and it is a finding, not a failure: no
        pair is attributable to the excluded source alone, so the bias threat has no instance."""
        path = annotation_csv(tmp_path, [
            {"patient_id": "R1", "hpo_code": A, "source": "prior_annotation"},
        ])
        restricted, stats = run_mod.gold_provenance(path, {"R1": {A}}, toy_view, {"new"})
        assert restricted == {"R1": {A}}
        assert stats["n_removed"] == 0
        assert stats["share_gold_from_phenobert"] == 0.0

    def test_a_sidecar_with_no_source_column_declines_rather_than_guessing(self, tmp_path,
                                                                          toy_view):
        path = annotation_csv(tmp_path, [{"patient_id": "R1", "hpo_code": A}], with_source=False)
        restricted, stats = run_mod.gold_provenance(path, {"R1": {A}}, toy_view, {"new"})
        assert restricted == {}
        assert "no source column" in stats["restriction"]

    def test_an_absent_sidecar_declines_rather_than_returning_the_full_gold(self, tmp_path,
                                                                           toy_view):
        """Returning the unrestricted ground truth would make the second block a copy of the first and
        look like evidence that the restriction changed nothing."""
        restricted, stats = run_mod.gold_provenance(tmp_path / "nope.csv", {"R1": {A}}, toy_view,
                                                    {"new"})
        assert restricted == {}
        assert stats["n_removed"] == 0

    def test_it_counts_every_source_it_saw(self, tmp_path, toy_view):
        path = annotation_csv(tmp_path, [
            {"patient_id": "R1", "hpo_code": A, "source": "new"},
            {"patient_id": "R1", "hpo_code": B, "source": "prior_annotation"},
            {"patient_id": "R2", "hpo_code": C, "source": "prior_annotation"},
        ])
        _restricted, stats = run_mod.gold_provenance(path, {"R1": {A, B}, "R2": {C}}, toy_view,
                                                     {"new"})
        assert stats["sources"] == {"new": 1, "prior_annotation": 2}


class TestCallBudget:
    def test_the_budget_is_jurors_times_segments(self, toy_view):
        """J x m is the chapter's cost claim, so the arithmetic is fixed, not asserted."""
        per_juror = {
            J0: {"R1": {0: {A}, 1: {B}}, "R2": {0: {C}}},
            J1: {"R1": {0: {A}}, "R2": {0: {C}, 1: {D}}},
        }
        rows = run_mod.stage_call_budget(
            _Cfg(), [J0, J1], per_juror, [J0, J1], ["R1", "R2"], _NoWrite())
        row = rows[0]
        assert row["n_jurors"] == 2
        # Segments per report is the MAX over jurors: the payload is keyed by segment, so the
        # fullest juror is the segmentation the run used. R1 -> 2, R2 -> 2, so 4 segments.
        assert row["mean_segments_per_report"] == 2.0
        assert row["mean_calls_per_report"] == 4.0
        assert row["total_calls"] == 8

    def test_a_report_no_juror_answered_still_counts_in_the_denominator(self, toy_view):
        """A silent report is a report the pipeline was still run on. Dropping it would make the
        per-report cost look higher than it is."""
        per_juror = {J0: {"R1": {0: {A}}}}
        rows = run_mod.stage_call_budget(_Cfg(), [J0], per_juror, [J0], ["R1", "R2"], _NoWrite())
        assert rows[0]["n_reports"] == 2
        assert rows[0]["mean_segments_per_report"] == 0.5


class _Cfg(dict):
    """Enough of a config for the two stages that only read `n_bootstrap`-free fields."""

    def get(self, key, default=None):
        return dict.get(self, key, default)


class _NoWrite:
    """A stand-in for the tables directory: the stages write, the tests only read the return."""

    def __truediv__(self, _name):
        import tempfile

        return Path(tempfile.mkdtemp()) / "out.csv"


class TestJuryErrorTaxonomy:
    """S9's false-positive half: chapter 4's rule, over the pairs this experiment scores."""

    def rows(self, tmp_path, toy_view, gold, pred):
        rows = run_mod.stage_jury_error_taxonomy(gold, toy_view, sorted(gold), tmp_path, pred)
        return {r["bucket"]: r["count"] for r in rows}

    def test_each_false_positive_is_filed_once_and_the_total_is_the_scored_one(self, tmp_path,
                                                                              toy_view):
        gold = {"r1": {C}, "r2": set()}
        # B is C's ancestor, D its sibling, HP:0000005 lies outside the scored subtree (not a
        # false positive at all) and HP:9999999 is no HPO identifier; F lands on an empty report.
        pred = {"r1": {C, B, D, "HP:0000005", "HP:9999999"}, "r2": {F}}
        got = self.rows(tmp_path, toy_view, gold, pred)
        assert got == {"ancestor": 1, "descendant": 0, "sibling": 1, "same_branch": 0,
                       "unrelated": 0, "invalid": 1, "no_gold": 1}
        pairs = [normalise_pair(gold[r], pred[r], toy_view)[:2] for r in sorted(gold)]
        fp = sum(len(p - g) for g, p in pairs)
        assert sum(got.values()) == fp == 4
        assert (tmp_path / "s9_error_taxonomy.csv").is_file()
