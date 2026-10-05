"""§Core extraction quality, eq. (1), worked by hand on two reports.

The running example, used by most tests in this file::

    report 1   ground truth = {C, D}   pred = {C, F}     TP=1  FP=1  FN=1
    report 2   ground truth = {M}      pred = {M, C}     TP=1  FP=1  FN=0

    micro            TP=2 FP=2 FN=1  ->  P=1/2      R=2/3      F1=4/7
    macro (report)   P=(1/2+1/2)/2=1/2  R=(1/2+1)/2=3/4  F1=(1/2+2/3)/2=7/12
    macro (term)     C: 1/1 tp,1 fp -> P=1/2 R=1   F1=2/3
                     D: 0 tp, 1 fn  -> P=0   R=0   F1=0
                     F: 0 tp, 1 fp  -> P=0   R=0   F1=0
                     M: 1 tp        -> P=1   R=1   F1=1
                     -> P=3/8       R=1/2       F1=5/12
"""

from __future__ import annotations

import pytest

from hpo_extraction.evaluation.metrics import (
    counts,
    flat_report,
    macro_prf_by_report,
    macro_prf_by_term,
    macro_prf_by_term_domains,
    micro_prf,
    prf,
    report_prf,
)
from fixtures.toy_ontology import C, D, F, HALLUCINATION, M

pytestmark = pytest.mark.unit

GOLD = [{C, D}, {M}]
PRED = [{C, F}, {M, C}]


# ── The equation itself ──────────────────────────────────────────────────────

def test_counts_are_tp_fp_fn(toy_view):
    assert counts({C, D}, {C, F}) == (1, 1, 1)
    assert counts({M}, {M, C}) == (1, 1, 0)


def test_prf_matches_equation_one():
    """P = TP/(TP+FP) = 1/2, R = TP/(TP+FN) = 1/2, F1 = 2PR/(P+R) = 1/2."""
    assert prf(1, 1, 1) == pytest.approx((0.5, 0.5, 0.5))


def test_prf_zero_denominators_do_not_raise():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    assert prf(0, 0, 3) == (0.0, 0.0, 0.0)


# ── Averaging schemes ────────────────────────────────────────────────────────

def test_micro_pools_counts_before_dividing():
    """TP=2, FP=2, FN=1 -> P=1/2, R=2/3, F1=4/7."""
    p, r, f = micro_prf(GOLD, PRED)
    assert (p, r, f) == pytest.approx((0.5, 2 / 3, 4 / 7))


def test_macro_over_reports_is_the_mean_of_per_report_scores():
    p, r, f, n = macro_prf_by_report(GOLD, PRED)
    assert n == 2
    assert (p, r, f) == pytest.approx((0.5, 0.75, 7 / 12))


def test_macro_over_terms_weights_every_term_equally():
    """The averaging that is "sensitive to rare phenotypes": four terms, each worth 1/4."""
    p, r, f, n = macro_prf_by_term(GOLD, PRED)
    assert n == 4
    assert (p, r, f) == pytest.approx((3 / 8, 0.5, 5 / 12))


def test_micro_and_macro_disagree_here_which_is_the_point():
    """Micro is dominated by the report with more terms. Macro weights the two reports equally."""
    assert micro_prf(GOLD, PRED)[1] != pytest.approx(macro_prf_by_report(GOLD, PRED)[1])


# ── The empty-report convention ──────────────────────────────────────────────

def test_empty_both_scores_perfect_under_the_default():
    assert report_prf(set(), set(), "perfect") == (1.0, 1.0, 1.0)


def test_empty_both_is_excluded_under_skip():
    assert report_prf(set(), set(), "skip") is None


def test_skip_changes_the_macro_mean_and_the_unit_count():
    """Two real reports plus an empty one: "perfect" hands the mean a free 1.0, "skip" does not."""
    gold = GOLD + [set()]
    pred = PRED + [set()]
    p_perfect, _, _, n_perfect = macro_prf_by_report(gold, pred, "perfect")
    p_skip, _, _, n_skip = macro_prf_by_report(gold, pred, "skip")
    assert (n_perfect, n_skip) == (3, 2)
    assert p_perfect == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert p_skip == pytest.approx(0.5)


def test_one_sided_empty_scores_zero_under_both_conventions():
    assert report_prf({C}, set(), "perfect") == (0.0, 0.0, 0.0)
    assert report_prf(set(), {C}, "skip") == (0.0, 0.0, 0.0)


# ── flat_report ──────────────────────────────────────────────────────────────

def test_flat_report_collects_every_number():
    out = flat_report(GOLD, PRED)
    assert out["n_reports"] == 2
    assert (out["tp"], out["fp"], out["fn"]) == (2, 2, 1)
    assert out["micro_f1"] == pytest.approx(4 / 7)
    assert out["macro_f1"] == pytest.approx(7 / 12)
    assert out["macro_term_f1"] == pytest.approx(5 / 12)


def test_macro_f1_and_macro_f1_of_means_are_different_quantities():
    """The skeleton's macro-F1 is the mean of per-report F1 (7/12). The repo's legacy scorer
    reports the harmonic mean of the averaged P and R (2*0.5*0.75/1.25 = 0.6). Both are returned so
    a results table can never quote one while citing the other."""
    out = flat_report(GOLD, PRED)
    assert out["macro_f1"] == pytest.approx(7 / 12)
    assert out["macro_f1_of_means"] == pytest.approx(0.6)


def test_flat_report_with_a_view_canonicalises_and_reports_the_drops(toy_view):
    out = flat_report([{C}], [{C, HALLUCINATION}], toy_view)
    assert out["n_pred_dropped"] == 1
    assert out["n_gold_dropped"] == 0
    assert (out["tp"], out["fp"], out["fn"]) == (1, 0, 0)


def test_flat_report_without_a_view_scores_raw_strings(toy_view):
    """No canonicalisation: the hallucinated code counts as a false positive rather than a drop."""
    out = flat_report([{C}], [{C, HALLUCINATION}])
    assert (out["tp"], out["fp"], out["fn"]) == (1, 1, 0)
    assert "n_pred_dropped" not in out


def test_misaligned_inputs_raise():
    with pytest.raises(ValueError, match="aligned"):
        flat_report([{C}], [{C}, {D}])


def test_an_empty_cohort_returns_zeros_rather_than_raising():
    """A results table needs every key present even when a run produced nothing."""
    out = flat_report([], [])
    assert out["n_reports"] == 0
    assert out["micro_f1"] == 0.0
    assert out["macro_f1"] == 0.0
    assert out["macro_term_f1"] == 0.0
    assert out["macro_n_units"] == 0
    assert out["macro_term_n_units"] == 0


def test_macro_helpers_are_empty_safe_on_their_own():
    assert macro_prf_by_report([], []) == (0.0, 0.0, 0.0, 0)
    assert macro_prf_by_term([], []) == (0.0, 0.0, 0.0, 0)


# ── Cross-check against the repo's existing scorer ───────────────────────────

def test_agrees_with_legacy_evaluate_micro_macro():
    """The new implementation is self-contained, so this asserts the duplication is consistent.

    ``hpo_extraction.evaluation.set_metrics.evaluate_micro_macro`` is the number every pre-thesis experiment reported.
    Micro P/R/F1 and macro P/R must match. The legacy ``macro_f1`` matches this package's
    ``macro_f1_of_means``, not its ``macro_f1``, see the test above.
    """
    from hpo_extraction.evaluation.set_metrics import evaluate_micro_macro

    legacy, _ = evaluate_micro_macro(GOLD, PRED)
    new = flat_report(GOLD, PRED)

    for key in ("micro_precision", "micro_recall", "micro_f1",
                "macro_precision", "macro_recall"):
        assert new[key] == pytest.approx(legacy[key]), key
    assert new["macro_f1_of_means"] == pytest.approx(legacy["macro_f1"])


class TestLabelMacroDomains:
    """Each average over the term set it is defined on, with the denominators printed.

    The worked example, small enough to check by hand::

        ground truth = [{A, B}, {B, C}]      pred = [{A, Z}, {B}]

        A  (TP=1, FP=0, FN=0)    P=1     R=1     F=1
        B  (TP=1, FP=0, FN=1)    P=1     R=1/2   F=2/3
        C  (TP=0, FP=0, FN=1)    -       R=0     F=0
        Z  (TP=0, FP=1, FN=0)    P=0     -       F=0

        precision over {A, B, Z} = 2/3      recall over {A, B, C} = 1/2
        F1 over all four = (1 + 2/3) / 4 = 5/12
    """

    GOLD = [{"A", "B"}, {"B", "C"}]
    PRED = [{"A", "Z"}, {"B"}]

    def test_hand_computed_values(self):
        got = macro_prf_by_term_domains(self.GOLD, self.PRED)
        assert got["precision"] == pytest.approx(2 / 3)
        assert got["recall"] == pytest.approx(1 / 2)
        assert got["f1"] == pytest.approx(5 / 12)

    def test_hand_computed_denominators(self):
        got = macro_prf_by_term_domains(self.GOLD, self.PRED)
        assert got["n_precision_terms"] == 3, "A, B, Z — the terms predicted at least once"
        assert got["n_recall_terms"] == 3, "A, B, C — the terms gold at least once"
        assert got["n_f1_terms"] == 4, "the union"

    def test_differs_from_the_shared_domain_version(self):
        """The whole reason the function exists: C drags shared-domain precision down.

        Under ``macro_prf_by_term`` the never-predicted annotated term C contributes ``P = 0``, even
        though the system made no precision claim about it.
        """
        shared_p, shared_r, _, shared_n = macro_prf_by_term(self.GOLD, self.PRED)
        split = macro_prf_by_term_domains(self.GOLD, self.PRED)
        assert shared_n == split["n_f1_terms"] == 4
        assert shared_p == pytest.approx(2 / 4), "C and its P=0 are inside the shared mean"
        assert split["precision"] > shared_p

    def test_a_perfect_system_scores_one_everywhere(self):
        got = macro_prf_by_term_domains(self.GOLD, self.GOLD)
        assert got["precision"] == got["recall"] == got["f1"] == pytest.approx(1.0)
        assert got["n_precision_terms"] == got["n_recall_terms"] == got["n_f1_terms"] == 3

    def test_a_system_that_predicts_nothing_has_an_empty_precision_domain(self):
        got = macro_prf_by_term_domains(self.GOLD, [set(), set()])
        assert got["n_precision_terms"] == 0
        assert got["precision"] == 0.0, "no precision claims were made; the mean is defined as 0"
        assert got["n_recall_terms"] == 3
        assert got["recall"] == 0.0

    def test_empty_cohort(self):
        got = macro_prf_by_term_domains([], [])
        assert got == {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                       "n_precision_terms": 0, "n_recall_terms": 0, "n_f1_terms": 0}

    def test_flat_report_is_opt_in(self):
        off = flat_report(self.GOLD, self.PRED)
        assert not any(k.startswith("label_macro") for k in off), (
            "default must stay byte-compatible with every shipped number"
        )
        on = flat_report(self.GOLD, self.PRED, label_macro_domains=True)
        assert on["label_macro_precision"] == pytest.approx(2 / 3)
        assert on["label_macro_n_f1_terms"] == 4

    def test_flat_report_keeps_the_legacy_columns_too(self):
        on = flat_report(self.GOLD, self.PRED, label_macro_domains=True)
        assert on["macro_term_precision"] == pytest.approx(2 / 4)
        assert on["label_macro_precision"] == pytest.approx(2 / 3)
