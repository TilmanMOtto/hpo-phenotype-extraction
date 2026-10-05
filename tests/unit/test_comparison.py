"""The comparison's subgroup logic and the prediction-set round trip.

The failures worth a test here are the ones that produce a *plausible* table:

* a family-labelled pair that is also in the ground truth, counted as an attribution error, predicting it
  is correct, and the one real instance (``SYN101 / HP:0100502``) is this shape;
* a subgroup recall that quietly becomes a precision because the prediction was not intersected
  with the slice first;
* a report predicted empty that loses its line on the way to disk, which hands the method free
  precision on every report it said nothing about.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "experiments" / "06_comparison" / "comparison"))

import complementarity  # noqa: E402
import cost  # noqa: E402
import subgroups  # noqa: E402
from hpo_extraction.evaluation.prediction_sets import read_prediction_sets, write_prediction_sets  # noqa: E402

pytestmark = pytest.mark.unit


def annotation(patient, code, **flags):
    """One row of the curated annotation export, with every qualifier column present."""
    row = {"patient_id": patient, "hpo_code": code}
    for label in subgroups.ALL_LABELS:
        row["q_" + label] = "1" if flags.get(label) else "0"
    return row


@pytest.fixture
def rows():
    return [
        annotation("R1", "HP:0001", lab_value=True),
        annotation("R1", "HP:0002", implicit=True),
        annotation("R2", "HP:0003", lab_value=True),
        # Labelled family and NOT in the ground truth, a relative's finding.
        annotation("R3", "HP:0009", family=True),
        # Labelled family on one row. A duplicate below carries no label and survives the policy,
        # so the pair IS in the ground truth. This is the SYN101 / HP:0100502 shape.
        annotation("R2", "HP:0004", family=True),
        annotation("R2", "HP:0004"),
        # Labelled but dropped from the ground truth by some other rule, not a subgroup member.
        annotation("R1", "HP:0008", lab_value=True),
    ]


@pytest.fixture
def gold():
    return {"R1": {"HP:0001", "HP:0002", "HP:0005"},
            "R2": {"HP:0003", "HP:0004"},
            "R3": set()}


class TestBuild:
    def test_recall_subgroups_are_subsets_of_the_gold(self, rows, gold):
        groups = subgroups.build(rows, gold)
        in_gold = {(r, c) for r, codes in gold.items() for c in codes}
        for label in subgroups.RECALL_LABELS:
            assert groups[label].pairs <= in_gold

    def test_a_labelled_pair_the_gold_dropped_is_not_in_the_subgroup(self, rows, gold):
        # HP:0008 is lab_value-labelled but absent from the ground truth, so it has no denominator.
        assert ("R1", "HP:0008") not in subgroups.build(rows, gold)["lab_value"].pairs

    def test_family_pair_that_is_in_the_gold_is_not_an_attribution_error(self, rows, gold):
        """Predicting it is CORRECT. Counting it as an attribution error would invert the finding."""
        attribution = subgroups.build(rows, gold)[subgroups.ATTRIBUTION_LABEL].pairs
        assert ("R2", "HP:0004") not in attribution
        assert ("R3", "HP:0009") in attribution
        assert len(attribution) == 1

    def test_missing_qualifier_column_raises_rather_than_scoring_zero(self, gold):
        stale = [{"patient_id": "R1", "hpo_code": "HP:0001"}]
        with pytest.raises(KeyError, match="q_lab_value"):
            subgroups.build(stale, gold)


class TestRecall:
    def test_prediction_outside_the_slice_changes_nothing(self, rows, gold):
        """The mark of a recall: an unrelated prediction is neither a hit nor a miss."""
        group = subgroups.build(rows, gold)["lab_value"]
        lean = {"R1": {"HP:0001"}, "R2": {"HP:0003"}}
        noisy = {"R1": {"HP:0001", "HP:0777"}, "R2": {"HP:0003", "HP:0888", "HP:0999"}}
        assert (subgroups.recall_metric(subgroups.recall_units(group, lean))
                == subgroups.recall_metric(subgroups.recall_units(group, noisy)))

    def test_recall_counts_only_subgroup_terms(self, rows, gold):
        group = subgroups.build(rows, gold)["lab_value"]          # R1/HP:0001, R2/HP:0003
        units = subgroups.recall_units(group, {"R1": {"HP:0001"}})
        assert subgroups.recall_metric(units) == {"recall": 0.5}

    def test_units_cover_only_carrying_reports(self, rows, gold):
        group = subgroups.build(rows, gold)["implicit"]           # R1 only
        assert len(subgroups.recall_units(group, {})) == 1


class TestAttribution:
    def test_counts_emitted_pairs_and_names_them(self, rows, gold):
        group = subgroups.build(rows, gold)[subgroups.ATTRIBUTION_LABEL]
        counts = subgroups.attribution_counts(group, {"R3": {"HP:0009", "HP:0010"}})
        assert counts["n_emitted"] == 1
        assert counts["emitted_pairs"] == "R3/HP:0009"
        assert counts["n_pairs"] == 1
        # No rate is offered at all, the caller cannot accidentally quote one.
        assert "rate" not in counts

    def test_silent_method_emits_none(self, rows, gold):
        group = subgroups.build(rows, gold)[subgroups.ATTRIBUTION_LABEL]
        assert subgroups.attribution_counts(group, {})["n_emitted"] == 0


class TestAudit:
    def test_every_qualifier_gets_a_row_even_when_unused(self, rows, gold):
        audit = subgroups.audit(rows, gold)
        per_label = {r["label"] for r in audit if r["role"] != "recall subgroup union"}
        assert per_label == set(subgroups.ALL_LABELS)
        roles = {r["label"]: r["role"] for r in audit}
        assert roles["negated"] == "not sliced on"
        assert roles["family"] == "attribution set"

    def test_the_union_row_is_the_union_and_not_the_sum(self, gold):
        """T6.2 quotes the share of ground truth with no string match, which is the UNION of the two
        slices. They are separate qualifier columns, so a pair can carry both and the sum
        double-counts it. The union row exists so the table need not hedge with an upper bound."""
        both = annotation("P1", "HP:0000001", lab_value=1, implicit=1)
        lab_only = annotation("P1", "HP:0000002", lab_value=1)
        audit = subgroups.audit([both, lab_only], {"P1": {"HP:0000001", "HP:0000002"}})
        by_label = {r["label"]: r for r in audit}
        assert by_label["lab_value"]["n_pairs_in_gold"] == 2
        assert by_label["implicit"]["n_pairs_in_gold"] == 1
        union = next(r for r in audit if r["role"] == "recall subgroup union")
        assert union["n_pairs_in_gold"] == 2, "the union, not the sum of 2 + 1"
        assert union["n_pairs_in_both"] == 1

    def test_family_row_splits_in_gold_from_not(self, rows, gold):
        row = next(r for r in subgroups.audit(rows, gold) if r["label"] == "family")
        assert row["n_pairs_labelled"] == 2
        assert row["n_pairs_in_gold"] == 1
        assert row["n_pairs_not_in_gold"] == 1


class TestPredictionSets:
    def test_round_trip(self, tmp_path):
        sets = {"R1": {"HP:0001", "HP:0002"}, "R2": {"HP:0003"}}
        path = write_prediction_sets(tmp_path / "p.csv", sets)
        assert read_prediction_sets(path) == sets

    def test_empty_prediction_keeps_its_line(self, tmp_path):
        """Dropping it would hand the method free precision on every report it was silent about."""
        path = write_prediction_sets(tmp_path / "p.csv", {"R1": set(), "R2": {"HP:0003"}})
        back = read_prediction_sets(path)
        assert back == {"R1": set(), "R2": {"HP:0003"}}
        assert "R1" in back

    def test_reads_the_gold_files_own_dialect(self, tmp_path):
        """The format IS the ground truth format, so a ground truth file must read back unchanged."""
        path = tmp_path / "gold.csv"
        path.write_text("patient_id,hpo_codes\nR1,HP:0001;HP:0002\nR2,\n", encoding="utf-8")
        assert read_prediction_sets(path) == {"R1": {"HP:0001", "HP:0002"}, "R2": set()}


class TestComplementarity:
    """T6.3 -- the two numbers, and the invariant that they are not complements."""

    ATTRIBUTION = [
        # never scored: no threshold could have reached these
        {"report_id": "R1", "hpo_id": "HP:0001", "bucket": "pruning", "pruned": "True"},
        {"report_id": "R1", "hpo_id": "HP:0002", "bucket": "retrieval", "pruned": "False"},
        # scored and decided wrongly
        {"report_id": "R2", "hpo_id": "HP:0003", "bucket": "judgement", "pruned": "False"},
        {"report_id": "R2", "hpo_id": "HP:0004", "bucket": "pooling", "pruned": "False"},
        # a bucket the groups do not name: it must be counted in neither
        {"report_id": "R2", "hpo_id": "HP:0005", "bucket": "residual", "pruned": "False"},
    ]
    GOLD = {"R1": {"HP:0001", "HP:0002"}, "R2": {"HP:0003", "HP:0004", "HP:0005"}}

    class View:
        """Identity resolution -- the join is about pairs, not about alternate ids."""

        @staticmethod
        def resolve(term):
            return term

    def test_it_counts_recovery_per_group_with_the_right_denominator(self):
        recovered = {"R1": {"HP:0001"}, "R2": {"HP:0003"}}
        rows = complementarity.build(self.ATTRIBUTION, recovered, self.GOLD, self.View(), "other")
        by_group = {r["buckets"]: r for r in rows}
        never = by_group["pruning+retrieval"]
        assert never["n_missed"] == 2 and never["n_recovered"] == 1
        assert never["share_recovered"] == 0.5
        scored = by_group["judgement+pooling"]
        assert scored["n_missed"] == 2 and scored["n_recovered"] == 1
        assert scored["share_recovered"] == 0.5

    def test_the_two_rows_are_not_complements(self):
        """Both can be high, or both low. Adding them would be meaningless, and the row shapes
        have to make that impossible to do by accident: different denominators, stated."""
        recovered = {"R1": {"HP:0001", "HP:0002"}, "R2": {"HP:0003", "HP:0004"}}
        rows = complementarity.build(self.ATTRIBUTION, recovered, self.GOLD, self.View(), "other")
        assert all(r["share_recovered"] == 1.0 for r in rows)
        assert len({r["n_missed"] for r in rows}) == 1      # equal here, by design
        assert sum(r["share_recovered"] for r in rows) == 2.0, "shares are per-group, not a split"

    def test_a_bucket_outside_the_named_groups_is_counted_in_neither(self):
        rows = complementarity.build(self.ATTRIBUTION, {}, self.GOLD, self.View(), "other")
        assert sum(r["n_missed"] for r in rows) == 4, "the residual row is not in either group"

    def test_a_pair_outside_this_chapters_gold_is_dropped(self):
        """The two experiments can disagree about the cohort. Counting a pair this table does not
        score would make the share wrong in a way nothing else would catch."""
        narrow = {"R1": {"HP:0001"}}
        rows = complementarity.build(self.ATTRIBUTION, {"R1": {"HP:0001"}}, narrow, self.View(),
                                     "other")
        by_group = {r["buckets"]: r for r in rows}
        assert by_group["pruning+retrieval"]["n_missed"] == 1
        assert by_group["judgement+pooling"]["n_missed"] == 0
        assert by_group["judgement+pooling"]["share_recovered"] == ""

    def test_no_attribution_yields_no_rows_rather_than_zeros(self):
        """A missing input must not read as "nothing was recovered" -- that is a measurement."""
        assert complementarity.build([], {"R1": {"HP:0001"}}, self.GOLD, self.View(), "x") == []

    def test_load_attribution_returns_empty_for_an_absent_file(self, tmp_path):
        assert complementarity.load_attribution(tmp_path / "nope.csv") == []

    def test_load_attribution_reads_the_rows_it_wrote(self, tmp_path):
        import csv as _csv

        path = tmp_path / "recall_attribution.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=["report_id", "hpo_id", "bucket", "pruned"])
            writer.writeheader()
            writer.writerows(self.ATTRIBUTION)
        got = complementarity.load_attribution(path)
        assert len(got) == len(self.ATTRIBUTION)
        assert got[0]["bucket"] == "pruning"


class TestGenerationCallCount:
    """The ensemble's call count is the one cost number with no other record, so it is COUNTED.

    A jury smaller than the whole pool needs its own count rather than a fraction of the pool's:
    the condition exists to make a claim about cost, and an estimate cannot support one. `glob` has no
    alternation, so naming three of eight models takes three patterns.
    """

    @staticmethod
    def _dumps(tmp_path, models, lines=2):
        for model in models:
            (tmp_path / f"llm_extractions_{model}.jsonl").write_text(
                "x\n" * lines, encoding="utf-8")
        return tmp_path

    def test_a_single_pattern_still_counts_the_whole_pool(self, tmp_path):
        d = self._dumps(tmp_path, ["medgemma", "medpsy", "phi4", "llama"])
        got = cost.count_generation_calls(d, "llm_extractions_*.jsonl", n_reports=2)
        assert got["n_generation_files"] == 4
        assert got["llm_calls_per_report"] == pytest.approx(4 * 2 / 2)

    def test_a_list_of_patterns_counts_only_those_models(self, tmp_path):
        d = self._dumps(tmp_path, ["medgemma", "medpsy", "phi4", "llama"])
        got = cost.count_generation_calls(
            d, ["llm_extractions_medgemma.jsonl", "llm_extractions_medpsy.jsonl",
                "llm_extractions_phi4.jsonl"], n_reports=2)
        assert got["n_generation_files"] == 3
        assert got["llm_calls_per_report"] == pytest.approx(3 * 2 / 2)

    def test_overlapping_patterns_do_not_double_count(self, tmp_path):
        """Two patterns matching one file must count it once, or the cost is inflated silently."""
        d = self._dumps(tmp_path, ["medgemma", "medpsy"])
        got = cost.count_generation_calls(
            d, ["llm_extractions_medgemma.jsonl", "llm_extractions_med*.jsonl"], n_reports=2)
        assert got["n_generation_files"] == 2

    def test_no_matching_dump_yields_no_number(self, tmp_path):
        """Absent is reported as absent. A zero here would read as a method that makes no calls."""
        assert cost.count_generation_calls(tmp_path, "llm_extractions_*.jsonl", 2) == {}


class TestTreeCostIsOneRun:
    """The tree's seconds and calls must come from the same run: the cache build.

    The row once printed 22 938 s per report (the exhaustive cache build) beside 31 228 calls (the
    re-run configuration), i.e. one run's clock against another run's calls.
    """

    @staticmethod
    def _results(tmp_path, cohort="hcy"):
        tables = tmp_path / "treephenorag_protocol" / "tables"
        tables.mkdir(parents=True)
        (tables / "cache_calls.csv").write_text(
            "cohort,retrieval_index,calls_per_report,calls_median,n_reports\n"
            f"{cohort},ontology_r3,180000.0,181000.0,118\n"
            f"{cohort},exemplar,170000.0,171000.0,118\n", encoding="utf-8")
        (tables / "core_quality.csv").write_text(
            "config,calls_per_report\nV3 selected poolings,31228.0\n", encoding="utf-8")
        log_dir = tmp_path / "treephenorag_scores_terminfo" / "hcy"
        log_dir.mkdir(parents=True)
        (log_dir / "run_0.log").write_text(
            "Done | 118 reports in 1180.0s | 18354 unique nodes | peak GPU 10.50 GB\n",
            encoding="utf-8")
        return tmp_path

    SPEC = {"exp_id": "treephenorag_scores_terminfo", "variant": "",
            "cache_calls_from": "treephenorag_protocol", "cache_index": "ontology_r3",
            "calls_from": "treephenorag_protocol", "calls_from_cohort": "hcy"}

    def test_calls_are_the_caches_and_the_operating_point_is_kept_apart(self, tmp_path):
        row = cost.collect(self.SPEC, str(self._results(tmp_path)), "hcy", "hcy")
        assert row.seconds_per_report == pytest.approx(10.0)
        assert row.llm_calls_per_report == pytest.approx(180000.0)
        assert row.operating_point_calls_per_report == pytest.approx(31228.0)

    def test_a_cohort_the_cache_does_not_cover_gets_no_calls(self, tmp_path):
        row = cost.collect(self.SPEC, str(self._results(tmp_path)), "gsc_2024_eval_206", "hcy")
        assert row.llm_calls_per_report is None
        assert row.operating_point_calls_per_report is None


class TestSequentialJuryRuntime:
    """The jury's wall clock is the SUM of its jurors', and a short sum must never look measured.

    Three failures would each produce a plausible number:

    * maxing the jurors instead of summing them, which reports the cost of the slowest model and
      calls it the ensemble's;
    * summing a subset, which prices an 8-juror condition at whatever jurors happened to write a file;
    * summing a juror that RESUMED from cached generations, whose file times only the reports it
      had left, the ensemble's version of the PhenoBERT baseline's reused-CNN trap, and the one that is short
      by hours while looking entirely legitimate.
    """

    @staticmethod
    def _timing(tmp_path, model, duration_s, n_reports=10, grounding="ok"):
        rows = [
            '{"stage": "extract", "model": "%s", "n_reports": %d, "n_sentences": 40, '
            '"n_detections": 7, "extraction_s": %.1f, "duration_s": %.1f}'
            % (model, n_reports, duration_s * 0.9, duration_s),
            '{"stage": "phenobert_%s", "duration_s": 1.0, "status": "%s"}' % (model, grounding),
        ]
        (tmp_path / f"slm_ensemble_timing_{model}.jsonl").write_text(
            "\n".join(rows) + "\n", encoding="utf-8")
        return tmp_path

    def test_jurors_are_summed_not_maxed(self, tmp_path):
        for model, seconds in (("medgemma", 100.0), ("medpsy", 200.0), ("phi4", 300.0)):
            self._timing(tmp_path, model, seconds)
        got = cost.from_per_model_timing(tmp_path, "slm_ensemble_timing_*.jsonl")
        assert got["n_models_timed"] == 3
        assert got["sequential_total_s"] == pytest.approx(600.0)
        assert got["seconds_per_report"] == pytest.approx(60.0)

    def test_a_named_subset_is_costed_on_its_own_members(self, tmp_path):
        """core-3 must be the sum of ITS three clocks, never a fraction of the pool's."""
        for model, seconds in (("medgemma", 100.0), ("medpsy", 200.0), ("phi4", 300.0),
                               ("llama", 900.0)):
            self._timing(tmp_path, model, seconds)
        got = cost.from_per_model_timing(
            tmp_path, ["slm_ensemble_timing_medgemma.jsonl", "slm_ensemble_timing_medpsy.jsonl",
                       "slm_ensemble_timing_phi4.jsonl"])
        assert got["sequential_total_s"] == pytest.approx(600.0)

    def test_a_resumed_juror_withholds_the_whole_sum(self, tmp_path):
        """Its file starts at the resubmission's t0, so the sum would be short by hours."""
        for model, seconds in (("medgemma", 100.0), ("medpsy", 200.0)):
            self._timing(tmp_path, model, seconds)
        got = cost.from_per_model_timing(tmp_path, "slm_ensemble_timing_*.jsonl",
                                         skip={"medpsy"})
        assert "seconds_per_report" not in got
        assert got["models_excluded"] == ["medpsy"]

    def test_the_run_log_is_what_names_a_resumed_juror(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text(
            "Loading medgemma from /x | greedy | max_new_tokens=512 batch=4\n"
            "Resuming medpsy from 812 cached record(s) - 3/118 report(s) still to do\n"
            "Extractions for phi4 are complete (4120 records) - no generation needed\n",
            encoding="utf-8")
        assert cost.resumed_models([log]) == {"medpsy", "phi4"}

    def test_a_clean_log_excludes_nobody(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("Extract done | model=medgemma 40 sentences (39 non-empty) -> 7 "
                       "detections in 100.0s | peak GPU 20.79 GB\n", encoding="utf-8")
        assert cost.resumed_models([log]) == set()

    def test_cached_grounding_is_reported_not_hidden(self, tmp_path):
        """A juror whose PhenoBERT pass was served from cache did not pay for it."""
        self._timing(tmp_path, "medgemma", 100.0, grounding="cached")
        self._timing(tmp_path, "medpsy", 200.0)
        got = cost.from_per_model_timing(tmp_path, "slm_ensemble_timing_*.jsonl")
        assert got["models_cached_grounding"] == ["medgemma"]

    def test_no_timing_file_yields_no_number(self, tmp_path):
        assert cost.from_per_model_timing(tmp_path, "slm_ensemble_timing_*.jsonl") == {}


class TestExtractLogFallback:
    """Recovering the jury's wall clock from the log, which is the route the shipped data takes.

    The per-model timing files are rewritten on every submission, and on both cohorts the last
    submission re-grounded cached text -- so they time PhenoBERT and not the generation. The run
    log is appended, so the original generating passes survive in it. What must not go wrong:

    * costing a re-grounding pass as if it were a generation (0.01 GB vs 8--21 GB peak);
    * quietly dropping the two jurors that hit the wall clock, instead of marking the sum a floor.
    """

    #: The shape the driver writes, both formats -- the non-empty count was added between runs.
    GENERATING = (
        "2026-07-31 10:26:50  INFO  exp13  Extract done | model=llama 2713 sentences "
        "-> 0 detections in 505.0s | peak GPU 15.37 GB\n"
        "2026-07-31 10:47:29  INFO  exp13  Extract done | model=medgemma 2713 sentences "
        "-> 0 detections in 1744.5s | peak GPU 8.13 GB\n")
    REGROUNDING = (
        "2026-08-16 21:58:37  INFO  exp13  Extract done | model=llama 2713 sentences "
        "(2713 non-empty) -> 2182 detections in 223.5s | peak GPU 0.01 GB\n"
        "2026-08-16 21:59:02  INFO  exp13  Extract done | model=deepseek 2720 sentences "
        "(2720 non-empty) -> 874 detections in 194.3s | peak GPU 0.01 GB\n")

    def _log(self, tmp_path, text):
        path = tmp_path / "run.log"
        path.write_text(text, encoding="utf-8")
        return [path]

    def test_only_the_resident_pass_counts(self, tmp_path):
        """A 0.01 GB peak is a re-grounding. Costing it would put the jury at seconds a report."""
        got = cost.from_extract_logs(self._log(tmp_path, self.GENERATING + self.REGROUNDING))
        assert sorted(got) == ["llama", "medgemma"], "deepseek only ever re-grounded"
        assert got["llama"]["seconds"] == pytest.approx(505.0), "not the 223.5s re-grounding"
        assert got["llama"]["peak_gpu_gb"] == pytest.approx(15.37)

    def test_both_log_formats_parse(self, tmp_path):
        """The detections clause changed shape between runs. The duration is what counts."""
        got = cost.from_extract_logs(self._log(tmp_path, self.REGROUNDING.replace("0.01", "9.34")))
        assert got["deepseek"]["seconds"] == pytest.approx(194.3)

    def test_a_missing_juror_makes_the_sum_a_floor(self, tmp_path):
        per = cost.from_extract_logs(self._log(tmp_path, self.GENERATING))
        seq = cost.sequential_from_extract_logs(per, ["llama", "medgemma", "deepseek"], 135)
        assert seq["models_missing"] == ["deepseek"]
        assert seq["n_models_timed"] == 2
        assert seq["seconds_per_report"] == pytest.approx((505.0 + 1744.5) / 135)

    def test_a_complete_condition_is_not_a_floor(self, tmp_path):
        per = cost.from_extract_logs(self._log(tmp_path, self.GENERATING))
        seq = cost.sequential_from_extract_logs(per, ["llama", "medgemma"], 135)
        assert seq["models_missing"] == []
        assert seq["peak_gpu_gb"] == pytest.approx(15.37), "the max over the arm, never the sum"

    def test_no_generating_pass_yields_no_number(self, tmp_path):
        per = cost.from_extract_logs(self._log(tmp_path, self.REGROUNDING))
        assert cost.sequential_from_extract_logs(per, ["llama"], 135).get(
            "seconds_per_report") is None

    def test_the_condition_membership_comes_from_the_dumps_it_counted(self, tmp_path):
        """Naming the jurors twice is how a condition's cost and its call count drift apart."""
        for model in ("medgemma", "medpsy", "phi4"):
            (tmp_path / f"llm_extractions_{model}.jsonl").write_text("x\n", encoding="utf-8")
        got = cost.count_generation_calls(tmp_path, "llm_extractions_*.jsonl", n_reports=1)
        assert got["models"] == ["medgemma", "medpsy", "phi4"]


class TestRagHpoCost:
    """RAG-HPO's cost cell printed ``n/a`` on every cohort: its driver wrote no timing file, and its
    ``Done`` line (``| N reports | M predicted terms | Xs (…) | peak Y GB``) matched neither the
    tree's regex nor the ``peak GPU`` one. Old runs have only the log. New ones write the file."""

    LOG = ("2026-08-20 10:00:00  INFO  exp13_raghpo  Done | 116 reports | 1480 predicted terms | "
           "5123.4s (44.17s/report) | peak 71.20 GB\n")
    SPEC = {"exp_id": "baseline_raghpo_70b", "variant": "rag_hpo"}

    @staticmethod
    def _cohort_dir(tmp_path):
        d = tmp_path / "baseline_raghpo_70b" / "hcy"
        d.mkdir(parents=True)
        return d

    def test_the_log_line_of_a_run_without_a_timing_file_is_read(self, tmp_path):
        (self._cohort_dir(tmp_path) / "run.log").write_text(self.LOG, encoding="utf-8")
        row = cost.collect(self.SPEC, str(tmp_path), "hcy", "hcy")
        assert row.seconds_per_report == pytest.approx(5123.4 / 116)
        assert row.peak_gpu_gb == pytest.approx(71.20)
        assert row.n_reports == 116

    def test_the_tree_line_still_wins_over_the_raghpo_form(self, tmp_path):
        log = tmp_path / "run_0.log"
        log.write_text("Done | 118 reports in 1180.0s | 18354 unique nodes | peak GPU 10.50 GB\n",
                       encoding="utf-8")
        got = cost.from_run_logs([log])
        assert got["seconds_per_report"] == pytest.approx(10.0)
        assert got["peak_gpu_gb"] == pytest.approx(10.50)

    def test_the_drivers_timing_file_is_read_with_calls_and_load(self, tmp_path):
        from hpo_extraction.baselines.rag_hpo_runner import write_timing

        d = self._cohort_dir(tmp_path)
        write_timing(str(d), ["a", "b"], {"a": {"HP:1", "HP:2"}}, n_llm_calls=10,
                     load_s=120.0, duration=100.0, peak_mem_bytes=int(40 * 1024 ** 3))
        (d / "run.log").write_text(self.LOG, encoding="utf-8")
        row = cost.collect(self.SPEC, str(tmp_path), "hcy", "hcy")
        assert row.seconds_per_report == pytest.approx(50.0), "the file wins over the log"
        assert row.llm_calls_per_report == pytest.approx(5.0)
        assert row.model_load_s == pytest.approx(120.0)
        assert "timing:rag_hpo_timing.jsonl" in row.source


class TestPhenobertTimingRerun:
    """The scored HCY PhenoBERT run reused a cached CNN output, so its 0.026 s/report timed the
    linking pass only. HCY's cost now comes from an uncached rerun in its own subdirectory, via a
    per-cohort ``cohort_dir`` map; GSC+ keeps reading the scored run."""

    SPEC = {"exp_id": "baseline_phenobert", "variant": "phenobert",
            "cohort_dir": {"hcy": "timing_rerun/baseline_phenobert/hcy"},
            "scope_if_log_contains": {"pattern": "Reusing complete PhenoBERT output",
                                      "scope": "LINKING ONLY"}}

    @staticmethod
    def _timing(d, n, seconds, log=""):
        d.mkdir(parents=True)
        (d / "phenobert_timing.jsonl").write_text(
            f'{{"stage": "annotate", "n_reports": {n}, "duration_s": {seconds}, '
            f'"mean_time_per_report_s": {seconds / n}}}\n', encoding="utf-8")
        (d / "run.log").write_text(log, encoding="utf-8")

    def test_hcy_reads_the_rerun_and_gsc_the_scored_run(self, tmp_path):
        base = tmp_path / "baseline_phenobert"
        self._timing(base / "hcy", 135, 3.5, log="Reusing complete PhenoBERT output: x\n")
        self._timing(base / "timing_rerun" / "baseline_phenobert" / "hcy", 135, 337.5)
        self._timing(base / "gsc", 228, 668.0)
        hcy = cost.collect(self.SPEC, str(tmp_path), "hcy", "hcy")
        assert hcy.seconds_per_report == pytest.approx(2.5)
        assert not hcy.scope_partial, "the rerun's log never reused the CNN"
        gsc = cost.collect(self.SPEC, str(tmp_path), "gsc_raghpo_ann", "gsc")
        assert gsc.seconds_per_report == pytest.approx(668.0 / 228)


class TestAutopcrPeakGpu:
    """AutoPCR's GPU cell printed ``---``: its timing file had no ``peak_gpu_gb`` and its ``Done``
    line no ``peak … GB``, so the peak reached MLflow only -- and even there read device 0 alone,
    a quarter of a 70B linker sharded over four cards."""

    SPEC = {"exp_id": "baseline_autopcr_70b", "variant": "autopcr"}
    LOG = ("Done | 135 reports | 900 detections (700 LLM-linked) -> 500 predicted terms | "
           "3483.0s (25.80s/report) | peak 72.40 GB\n")

    def test_the_done_line_carries_the_peak(self, tmp_path):
        d = tmp_path / "baseline_autopcr_70b" / "hcy"
        d.mkdir(parents=True)
        (d / "autopcr_timing.jsonl").write_text(
            '{"stage": "annotate", "n_reports": 135, "n_llm_calls": 699, '
            '"model_load_s": 174.3, "duration_s": 3483.0, "peak_gpu_gb": 72.4}\n',
            encoding="utf-8")
        (d / "run.log").write_text(self.LOG, encoding="utf-8")
        row = cost.collect(self.SPEC, str(tmp_path), "hcy", "hcy")
        assert row.peak_gpu_gb == pytest.approx(72.40)
        assert row.seconds_per_report == pytest.approx(3483.0 / 135)
        assert row.llm_calls_per_report == pytest.approx(699 / 135)

    def test_the_peak_is_summed_over_every_visible_card(self):
        from hpo_extraction.models.gpu_memory import peak_gpu_bytes_all_devices, reset_peak_all_devices

        class _Cuda:
            reset = []

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 4

            @staticmethod
            def max_memory_allocated(device):
                return (device + 1) * 1024 ** 3

            @classmethod
            def reset_peak_memory_stats(cls, device):
                cls.reset.append(device)

        class _Torch:
            cuda = _Cuda

        reset_peak_all_devices(_Torch)
        assert _Cuda.reset == [0, 1, 2, 3]
        assert peak_gpu_bytes_all_devices(_Torch) == 10 * 1024 ** 3

    def test_no_cuda_is_zero_not_an_error(self):
        from hpo_extraction.models.gpu_memory import peak_gpu_bytes_all_devices

        class _Torch:
            class cuda:
                @staticmethod
                def is_available():
                    return False

        assert peak_gpu_bytes_all_devices(_Torch) == 0

    def test_the_timing_files_peak_beats_an_older_runs_in_the_appended_log(self, tmp_path):
        """run.log is appended. A re-run whose MLflow block failed writes no `Done` line, so the
        log's peaks are an older run's. The timing file is rewritten per run and must win."""
        from hpo_extraction.baselines.rag_hpo_runner import write_timing

        d = tmp_path / "baseline_raghpo_70b" / "hcy"
        d.mkdir(parents=True)
        (d / "run.log").write_text(TestRagHpoCost.LOG, encoding="utf-8")   # August: peak 71.20
        write_timing(str(d), ["a", "b"], {}, n_llm_calls=4, load_s=1.0, duration=10.0,
                     peak_mem_bytes=int(66.5 * 1024 ** 3))
        row = cost.collect({"exp_id": "baseline_raghpo_70b", "variant": "rag_hpo"},
                           str(tmp_path), "hcy", "hcy")
        assert row.peak_gpu_gb == pytest.approx(66.5)
        assert row.seconds_per_report == pytest.approx(5.0)
        assert row.n_reports == 2


class TestSubsetTiming:
    """Every GSC+ system ran once over all 228 abstracts, so the run-level sources give both
    subsets the same seconds. The recovered per-subset sums replace them -- seconds only."""

    CSV = ("method,cohort,n_documents,document_seconds,overhead_seconds,seconds_per_report,exact\n"
           "raghpo_8b,gsc_raghpo_ann,114,3581.3,1.3,31.43,False\n"
           "phenojury,gsc_raghpo_ann,114,31441.8,477.9,283.30,True\n")

    def _table(self, tmp_path):
        path = tmp_path / "subset_seconds.csv"
        path.write_text(self.CSV, encoding="utf-8")
        return cost.load_subset_timing(path), path

    def test_seconds_are_replaced_and_the_floor_is_lifted(self, tmp_path):
        table, path = self._table(tmp_path)
        row = cost.CostRow(cohort="gsc_raghpo_ann", seconds_per_report=201.7,
                           llm_calls_per_report=55.4, seconds_is_lower_bound=True, source="x")
        cost.apply_subset_timing(row, "phenojury", table, path)
        assert row.seconds_per_report == pytest.approx(283.30)
        assert not row.seconds_is_lower_bound and not row.seconds_is_approximate
        assert row.llm_calls_per_report == pytest.approx(55.4)
        assert "subset timing:subset_seconds.csv" in row.source

    def test_an_apportioned_stage_marks_the_cell_approximate(self, tmp_path):
        table, path = self._table(tmp_path)
        row = cost.CostRow(cohort="gsc_raghpo_ann", seconds_per_report=26.75)
        cost.apply_subset_timing(row, "raghpo_8b", table, path)
        assert row.seconds_is_approximate and row.seconds_per_report == pytest.approx(31.43)

    def test_rows_without_a_trace_are_untouched(self, tmp_path):
        table, path = self._table(tmp_path)
        for method, cohort in (("treephenorag", "gsc_raghpo_ann"), ("raghpo_8b", "hcy")):
            row = cost.CostRow(cohort=cohort, seconds_per_report=1.0, source="x")
            cost.apply_subset_timing(row, method, table, path)
            assert row.seconds_per_report == 1.0 and row.source == "x"

    def test_a_missing_file_is_no_table(self, tmp_path):
        assert cost.load_subset_timing(tmp_path / "absent.csv") == {}
