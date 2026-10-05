"""§Scalability, measured cost in the three regimes the skeleton keeps separate."""

from __future__ import annotations

import math

import pytest

from hpo_extraction.evaluation.metrics import (
    THESIS_METRIC_INDEX,
    break_even_corpus_size,
    deployment_cost,
    hardware_profile,
    scaling_curve,
    slm_call_stats,
)

pytestmark = pytest.mark.unit


# ── Per-report inference cost ────────────────────────────────────────────────

TRAVERSAL = [
    {"n_slm_calls": 100, "n_unique_nodes_visited": 20, "total_frontier_insertions": 30},
    {"n_slm_calls": 200, "n_unique_nodes_visited": 40, "total_frontier_insertions": 60},
]


def test_slm_call_stats_summarise_per_report_cost():
    out = slm_call_stats(TRAVERSAL)
    assert out["n_reports"] == 2
    assert out["slm_calls_per_report"]["mean"] == pytest.approx(150.0)
    assert out["slm_calls_per_report"]["total"] == pytest.approx(300.0)
    assert out["unique_nodes_per_report"]["mean"] == pytest.approx(30.0)


def test_dedup_ratio_makes_the_dag_deduplication_auditable():
    """90 frontier insertions produced 60 unique evaluations: each node was offered 1.5 times.

    "Because H is a DAG, a term may be inserted into the frontier along several paths, and we report
    unique nodes evaluated and total frontier insertions separately." A ratio of 1.0 would mean the
    ontology had behaved as a tree.
    """
    assert slm_call_stats(TRAVERSAL)["dedup_ratio"] == pytest.approx(1.5)


def test_calls_per_node_recovers_the_effective_s():
    """300 SLM calls over 60 unique nodes is S = 5 retrieved sentences per node."""
    assert slm_call_stats(TRAVERSAL)["calls_per_node"] == pytest.approx(5.0)


def test_slm_call_stats_on_an_empty_run():
    out = slm_call_stats([])
    assert out["n_reports"] == 0
    assert math.isnan(out["dedup_ratio"])


def test_hardware_profile_derives_seconds_per_report_and_carries_the_spec():
    out = hardware_profile(
        n_reports=50, wall_clock_seconds=500.0, peak_gpu_bytes=8 * 1024 ** 3,
        model_parameters=8_000_000_000, device="A100-40GB", quantisation="4bit",
        serving_stack="transformers", batch_size=1, context_length=4096,
    )
    assert out["seconds_per_report"] == pytest.approx(10.0)
    assert out["peak_gpu_gb"] == pytest.approx(8.0)
    assert out["hardware"]["quantisation"] == "4bit"
    assert out["hardware"]["device"] == "A100-40GB"


def test_tokens_per_report_stays_none_because_no_run_records_it():
    """The documented instrumentation gap. Leaving it None keeps an unmeasured column visibly
    unmeasured rather than filling it with an estimate that would read as a measurement."""
    out = hardware_profile(n_reports=10, wall_clock_seconds=100.0)
    assert out["tokens_per_report"] is None
    assert out["peak_gpu_gb"] is None


def test_hardware_profile_rejects_a_zero_report_count():
    with pytest.raises(ValueError, match="positive"):
        hardware_profile(n_reports=0, wall_clock_seconds=1.0)


# ── Deployment cost ──────────────────────────────────────────────────────────

def test_deployment_cost_normalises_per_indexed_term():
    """Makes a 9-target system and a whole-ontology system comparable on one axis."""
    out = deployment_cost(gpu_hours=20.0, index_bytes=2 * 1024 ** 3, n_terms_indexed=18_354)
    assert out["index_gb"] == pytest.approx(2.0)
    assert out["gpu_hours_per_term"] == pytest.approx(20.0 / 18_354)
    assert out["bytes_per_term"] == pytest.approx(2 * 1024 ** 3 / 18_354)


def test_deployment_cost_reports_none_for_the_figures_no_run_records():
    out = deployment_cost(gpu_hours=None)
    assert out["gpu_hours"] is None
    assert out["gpu_hours_per_term"] is None


def test_break_even_corpus_size_hand_worked():
    """2 GPU-hours = 7200 s of deployment. Saving 6 s per report pays it back after 1200 reports."""
    out = break_even_corpus_size(
        deployment_gpu_hours=2.0, baseline_seconds_per_report=10.0,
        method_seconds_per_report=4.0,
    )
    assert out["n_reports"] == pytest.approx(1200.0)
    assert out["seconds_saved_per_report"] == pytest.approx(6.0)
    assert out["economical"] is True


def test_break_even_is_infinite_when_the_method_is_not_faster():
    """No corpus size pays back a deployment cost that buys nothing. A finite number here would
    invert the conclusion."""
    out = break_even_corpus_size(2.0, baseline_seconds_per_report=4.0,
                                 method_seconds_per_report=10.0)
    assert out["n_reports"] == math.inf
    assert out["economical"] is False


# ── Scaling in ontology size ─────────────────────────────────────────────────

def test_scaling_curve_recovers_a_known_exponent():
    """Points generated from calls = 3 * N^0.5 must fit back to an exponent of 1/2."""
    points = [(100, 3 * 100 ** 0.5), (400, 3 * 400 ** 0.5), (1600, 3 * 1600 ** 0.5)]
    out = scaling_curve(points)
    assert out["exponent"] == pytest.approx(0.5, abs=1e-9)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-9)
    assert out["sublinear"] is True


def test_scaling_curve_flags_linear_growth_as_not_sublinear():
    """"This is the measurement that distinguishes a scalability claim from a runtime report."""
    points = [(100, 100.0), (400, 400.0), (1600, 1600.0)]
    out = scaling_curve(points)
    assert out["exponent"] == pytest.approx(1.0, abs=1e-9)
    assert out["sublinear"] is False


def test_linear_reference_is_fixed_at_the_smallest_measured_n():
    """It answers "what would the largest subtree have cost had scaling been linear from here"."""
    out = scaling_curve([(100, 30.0), (400, 60.0), (1600, 120.0)])
    reference = {p["n_terms"]: p["slm_calls_per_report"] for p in out["linear_reference"]}
    assert reference[100] == pytest.approx(30.0)
    assert reference[400] == pytest.approx(120.0)
    assert reference[1600] == pytest.approx(480.0)


def test_scaling_curve_sorts_its_points():
    out = scaling_curve([(1600, 120.0), (100, 30.0), (400, 60.0)])
    assert [p["n_terms"] for p in out["points"]] == [100.0, 400.0, 1600.0]


def test_scaling_curve_needs_at_least_two_points():
    with pytest.raises(ValueError, match="at least two"):
        scaling_curve([(100, 30.0)])


def test_scaling_curve_rejects_non_positive_values_a_log_fit_cannot_take():
    with pytest.raises(ValueError, match="positive"):
        scaling_curve([(0, 30.0), (400, 60.0)])


# ── The index that makes the results chapter checkable ───────────────────────

def test_thesis_metric_index_covers_every_skeleton_section():
    """The index is the lookup table from a number in the thesis to the formula behind it, so a
    missing section here means a metric with no traceable implementation."""
    sections = list(THESIS_METRIC_INDEX)
    for expected in ("eq. (1)", "eq. (2)", "eq. (3)", "eq. (4)", "eq. (5)",
                     "eq. (6)", "eq. (7)", "eq. (8)"):
        assert any(expected in s for s in sections), expected
    for expected in ("§Blocking depth", "§Threshold transferability",
                     "§Scalability — deployment cost",
                     "§Scalability — scaling in ontology size"):
        assert any(s.startswith(expected) for s in sections), expected


def test_every_index_entry_points_at_a_callable():
    for section, entries in THESIS_METRIC_INDEX.items():
        for label, fn in entries.items():
            assert callable(fn), f"{section} / {label}"
