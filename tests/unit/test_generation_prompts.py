"""Unit tests for the earlier prompt sweep.

The GPUs and the real cohorts are on the cluster, so nothing here generates text. What these tests
protect is everything that can be wrong *without crashing*, which is the whole risk surface of a
screening experiment, because a screen's output is a ranking and a wrong ranking looks like
a right one.

**The prompts must survive PhenoBERT's preprocessor.** ``experiments/findings/exp13_phenobert_input_format.md``
measured recovery of the intended HPO id at between 9.4 % and 82.1 % across input layouts *alone*, a numbered list costs 60 points, a trailing space costs 50. A prompt that asks for one of those
layouts produces a perfectly readable answer that grounds to nothing, and the resulting ranking
would measure the layout bug rather than the wording. So the templates are checked against the
constraints, and rendered output is pushed through the real ``terminate_lines`` /
``build_phenobert_input``.

**The array decode must be a bijection.** 56 SLURM tasks address 56 (prompt, model) cells by
integer. An off-by-one leaves one cell unrun and another computed twice, and the aggregation would
happily rank seven prompts anyway.

**The promotion gate must be conservative.** It spends 32 GPU array tasks at up to 12 h each. Every
branch that could open it wrongly, a missing baseline, a margin just under threshold, no
contenders, is asserted closed.

**The diagnostics must discriminate.** Echo rate and label exactness are the metrics that say
whether a prompt did what it was asked. If they cannot tell a canonical term from a copied
sentence, the screen has no signal beyond a noisy 20-report F1.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from hpo_extraction.phenojury.phenobert import build_phenobert_input, terminate_lines
from hpo_extraction.phenojury.prompt_diagnostics import (
    diagnose_records,
    echo_rate,
    format_compliance,
    label_exactness,
    marginal_gold_yield,
    negation_trap_rate,
)
from hpo_extraction.phenojury.prompts import (
    BASELINE_KEY,
    NEGATION_TRIGGER_WORDS,
    P_PROMPT_KEYS,
    PROMPT_KEYS,
    PROMPTS,
    Q_PROMPT_KEYS,
    SHAPE_JSON,
    SHAPE_PIPE_FIRST,
    SHAPE_SENTENCE,
    SHAPE_TWO_COLUMN,
    get_prompt,
)
from hpo_extraction.phenojury.generation_prompts import (
    _fmt_opt,
    _json_none,
    _mean_over_cells,
    decode_array_index,
    rank_prompts,
    select_cohort,
)
from hpo_extraction.phenojury.generation import MODEL_KEYS

pytestmark = pytest.mark.unit

SENTENCE = "The boy had a small head and floppy limbs."


# ── The prompt library ───────────────────────────────────────────────────────
class TestPromptLibrary:
    def test_every_prompt_renders_the_sentence(self):
        for spec in PROMPTS:
            assert SENTENCE in spec.render(SENTENCE), spec.key

    def test_keys_are_unique_and_ordered(self):
        assert len(set(PROMPT_KEYS)) == len(PROMPT_KEYS)
        # The SLURM array decodes against this order, so a reorder repoints queued tasks at
        # different prompts. Pinning the first entry pins the control's index at 0..7.
        assert PROMPT_KEYS[0] == BASELINE_KEY

    def test_baseline_is_exp13_06_verbatim(self):
        """The control must stay byte-identical to the prompt the Free Listing generation run shipped.

        Every number the sweep produces is a delta against it. "Improving" the baseline, even its
        spacing, would silently redefine what every margin means.
        """
        from hpo_extraction.phenojury.generation import SYSTEM_PROMPT, USER_TEMPLATE

        spec = get_prompt(BASELINE_KEY)
        assert spec.system == SYSTEM_PROMPT
        assert spec.user_template == USER_TEMPLATE

    def test_restating_prompts_forbid_numbering(self):
        """Numbered lists recover 21.9 % against 82.1 % for a terminated list.

        Scoped to the p-family on purpose. The q-family is somebody else's prompt text, registered
        verbatim, and it does NOT carry this instruction, which is a measured risk, not a
        bug, and is asserted as such by
        :meth:`TestRoundTwoIsVerbatim.test_q_family_does_not_forbid_numbering`.
        """
        for key in P_PROMPT_KEYS:
            if key == BASELINE_KEY:
                continue
            template = get_prompt(key).user_template
            assert "not number" in template or "Do not number" in template, (
                f"{key} does not forbid numbering"
            )

    def test_no_prompt_asks_for_negation_words(self):
        """A line containing a ``getNegativeWords()`` token is discarded as ``Neg``.

        The attribution variant is the trap: telling a model to write "no dysmorphic features"
        would delete its own output. It must ask for **omission** instead.

        The ``no phenotype`` sentinel is exempt and must stay exempt, ``wrote_something`` filters
        it before grounding, so it never reaches PhenoBERT's negation check. It is stripped here,
        not special-cased in the loop so the assertion below covers everything else.
        """
        for spec in PROMPTS:
            body = spec.user_template.lower().replace("no phenotype", "«sentinel»")
            for bad in ("respond 'no ", "write 'no ", "mark as not ", "state that the patient has no"):
                assert bad not in body, f"{spec.key}: instructs a negated finding ({bad!r})"

    def test_attribution_prompt_asks_for_omission_not_denial(self):
        """The specific case the rule above exists for."""
        body = get_prompt("p6_attribution").user_template.lower()
        assert "leave out" in body

    def test_fewshot_examples_are_canonical_terms(self):
        """p5's demonstrations must show restatement, not echoing, they *are* the instruction."""
        spec = get_prompt("p5_fewshot")
        for term in ("Microcephaly", "Hypotonia", "Hepatomegaly"):
            assert term in spec.user_template

    def test_unknown_key_names_the_valid_ones(self):
        with pytest.raises(ValueError, match="p0_baseline"):
            get_prompt("p9_nonexistent")


# ── The PhenoBERT layout contract ────────────────────────────────────────────
class TestPhenoBertLayout:
    """The check that has to exist before any GPU time is spent.

    A prompt's output reaches PhenoBERT through ``build_phenobert_input``. If a line arrives
    unterminated, ``process_text2phrases`` fuses it into the next one and the phrase is never a
    candidate. This is the mechanism the format finding documents, and it is invisible end to end:
    the run completes, the file is full, the ranking is meaningless.
    """

    @pytest.mark.parametrize("body", [
        "Microcephaly\nHypotonia",
        "- Microcephaly\n- Hypotonia",
        "small head => Microcephaly\nfloppy limbs => Hypotonia",
        "no phenotype",
    ])
    def test_every_line_survives_termination(self, body):
        text, _ = build_phenobert_input(
            [{"patient_id": "p1", "sentence_number": 0, "llm_output": body}], "p1"
        )
        for line in text.split("\n"):
            assert line, "blank line would fuse its neighbours"
            assert line[-1] in ".;:,!?", f"unterminated line: {line!r}"

    def test_the_marker_line_is_terminated_too(self):
        """``sentence_number: 12\\nUrinary retention`` → ``12.urinary retention``.

        The marker's own digit fusing into the first phenotype word is what cost the old layout all
        but 3.8 % of the terms it was handed.
        """
        text, _ = build_phenobert_input(
            [{"patient_id": "p1", "sentence_number": 12, "llm_output": "Urinary retention"}], "p1"
        )
        assert text.startswith("sentence_number: 12;")

    def test_trailing_whitespace_is_stripped(self):
        """A line ending in a space defeats the lookbehind entirely, worse than the fusion."""
        assert terminate_lines("Microcephaly   \nHypotonia") == "Microcephaly;\nHypotonia;"


# ── Array decoding ───────────────────────────────────────────────────────────
class TestArrayDecode:
    def test_decode_is_a_bijection_over_the_full_screen(self):
        n = len(PROMPT_KEYS) * len(MODEL_KEYS)
        cells = {decode_array_index(i, list(PROMPT_KEYS)) for i in range(n)}
        assert len(cells) == n

    def test_prompt_major_keeps_one_prompt_contiguous(self):
        """So a single failed prompt is re-runnable as ``--array=<p*8>-<p*8+7>``."""
        block = [decode_array_index(i, list(PROMPT_KEYS))[0] for i in range(len(MODEL_KEYS))]
        assert set(block) == {PROMPT_KEYS[0]}

    def test_out_of_range_raises_with_the_expected_count(self):
        n = len(PROMPT_KEYS) * len(MODEL_KEYS)
        with pytest.raises(ValueError, match=str(n)):
            decode_array_index(n, list(PROMPT_KEYS))

    def test_restricted_roster_decodes_over_the_subset(self):
        """Phase B passes two prompts and submits a 16-task array."""
        subset = ["p2_canonical", "p4_implicit"]
        cells = {decode_array_index(i, subset) for i in range(2 * len(MODEL_KEYS))}
        assert len(cells) == 2 * len(MODEL_KEYS)
        assert {p for p, _ in cells} == set(subset)


# ── The promotion gate ───────────────────────────────────────────────────────
def _rows(**scores):
    return [{"prompt_key": k, "best_micro_f1": v, "best_rule": "vote_k2"}
            for k, v in scores.items()]


class TestPromotionGate:
    def test_promotes_on_a_clear_margin(self):
        v = rank_prompts(_rows(p0_baseline=0.20, p2_canonical=0.27, p4_implicit=0.24),
                         BASELINE_KEY, 0.05, 2)
        assert v["promote"] is True
        assert [s["prompt_key"] for s in v["selected"]] == ["p2_canonical", "p4_implicit"]
        assert v["best_margin"] == pytest.approx(0.07)

    def test_refuses_just_below_the_threshold(self):
        """4 points on 20 reports is noise. Phase B is 32 tasks at up to 12 h each.

        ``selected`` must come back empty, not merely be ignored: anything downstream that reads
        it without re-checking ``promote`` would submit the prompts the screen just declined.
        """
        v = rank_prompts(_rows(p0_baseline=0.20, p2_canonical=0.24), BASELINE_KEY, 0.05, 2)
        assert v["promote"] is False
        assert v["selected"] == []
        assert "below the 0.05 threshold" in v["reason"]
        # The ranking itself is still reported, a closed gate must remain readable.
        assert [r["prompt_key"] for r in v["ranking"]] == ["p2_canonical", "p0_baseline"]

    def test_promotes_at_the_threshold(self):
        v = rank_prompts(_rows(p0_baseline=0.20, p2_canonical=0.25), BASELINE_KEY, 0.05, 2)
        assert v["promote"] is True

    def test_refuses_when_the_baseline_is_missing(self):
        """Absent must not read as zero, that would promote on any result at all."""
        v = rank_prompts(_rows(p2_canonical=0.27), BASELINE_KEY, 0.05, 2)
        assert v["promote"] is False
        assert v["baseline_micro_f1"] is None
        assert "no result" in v["reason"]

    def test_refuses_when_every_variant_loses(self):
        v = rank_prompts(_rows(p0_baseline=0.30, p2_canonical=0.11), BASELINE_KEY, 0.05, 2)
        assert v["promote"] is False
        assert v["best_margin"] < 0

    def test_baseline_is_never_promoted_to_itself(self):
        v = rank_prompts(_rows(p0_baseline=0.40, p2_canonical=0.10), BASELINE_KEY, 0.0, 2)
        assert BASELINE_KEY not in [s["prompt_key"] for s in v["selected"]]

    def test_verdict_is_json_serialisable(self):
        """The cluster gate reads this file. A non-serialisable value would strand phase B."""
        v = rank_prompts(_rows(p0_baseline=0.20, p2_canonical=0.27), BASELINE_KEY, 0.05, 2)
        assert json.loads(json.dumps(v))["promote"] is True


# ── MLflow must never be essential ────────────────────────────────────────
# ── Diagnostics ──────────────────────────────────────────────────────────────
#: A stand-in for ``HPOTree.p_phrase2HPO``: sorted normalised tokens → HPO id.
PHRASES = {"microcephaly": "HP:0000252", "hypotonia": "HP:0001252"}


class TestDiagnostics:
    def test_echo_rate_separates_copying_from_restating(self):
        assert echo_rate(["small head"], SENTENCE) == pytest.approx(1.0)
        assert echo_rate(["Microcephaly"], SENTENCE) == pytest.approx(0.0)

    def test_label_exactness_rewards_canonical_wording(self):
        assert label_exactness(["Microcephaly", "Hypotonia"], PHRASES) == pytest.approx(1.0)
        assert label_exactness(["the boy had a small head"], PHRASES) == pytest.approx(0.0)

    def test_two_column_lines_are_judged_on_the_right_half(self):
        """p3's left column is *supposed* to echo. Scoring it would penalise the variant for
        doing what it was asked."""
        assert label_exactness(["small head => Microcephaly"], PHRASES,
                               SHAPE_TWO_COLUMN) == pytest.approx(1.0)

    def test_negation_traps_are_detected(self):
        assert negation_trap_rate(["Lower limb spasticity"]) == pytest.approx(1.0)
        assert negation_trap_rate(["Microcephaly"]) == pytest.approx(0.0)
        # Every trigger must be caught as a bare token, not a substring.
        for word in NEGATION_TRIGGER_WORDS:
            assert negation_trap_rate([f"{word} finding"]) == pytest.approx(1.0)

    def test_compliance_flags_numbering(self):
        assert format_compliance(["Microcephaly"], "p2_canonical") == pytest.approx(1.0)
        assert format_compliance(["1. Microcephaly"], "p2_canonical") == pytest.approx(0.0)

    def test_compliance_flags_prose_of_any_length(self):
        """A model that writes sentences has ignored the one-per-line contract.

        Length alone misses the short case, "The boy had a small head." is six tokens, so the
        internal sentence boundary is what catches it.
        """
        assert format_compliance(["The boy had a small head. He was floppy."],
                                 "p2_canonical") == pytest.approx(0.0)
        long_prose = "the patient was noted to have a head circumference well below the third centile"
        assert format_compliance([long_prose], "p2_canonical") == pytest.approx(0.0)

    def test_a_terminated_single_finding_is_still_compliant(self):
        """``terminate_lines`` appends a ``;`` to every line, that must not read as prose."""
        assert format_compliance(["Microcephaly;"], "p2_canonical") == pytest.approx(1.0)

    def test_compliance_requires_the_separator_for_two_column(self):
        assert format_compliance(["Microcephaly"], "p3_two_column") == pytest.approx(0.0)
        assert format_compliance(["small head => Microcephaly"],
                                 "p3_two_column") == pytest.approx(1.0)

    def test_baseline_compliance_is_vacuous_by_construction(self):
        """p0 requested no shape, so it cannot fail one, its echo rate carries the comparison."""
        assert format_compliance([SENTENCE], "p0_baseline") == pytest.approx(1.0)

    def test_silence_does_not_earn_a_perfect_echo_rate(self):
        """A model answering 'no phenotype' everywhere must not look like the best restater."""
        recs = [{"llm_output": "no phenotype", "sentence_text": SENTENCE}] * 3
        d = diagnose_records(recs, "p2_canonical", PHRASES)
        assert d["n_records"] == 3
        assert d["n_wrote"] == 0
        assert d["echo_rate"] == 0.0

    def test_reasoning_output_is_judged_on_the_answer(self):
        """CoT prose would make the three reasoning models look like the worst echoers."""
        recs = [{
            "llm_output": "<think>The boy had a small head, so this is …</think>\nMicrocephaly",
            "sentence_text": SENTENCE,
        }]
        d = diagnose_records(recs, "p2_canonical", PHRASES)
        assert d["label_exactness"] == pytest.approx(1.0)
        assert d["echo_rate"] == pytest.approx(0.0)


class TestMarginalYield:
    def test_counts_gains_losses_and_costs_separately(self):
        gold = {"r1": {"HP:1", "HP:2"}}
        pred = {"r1": {"HP:2", "HP:9"}}          # gains HP:2, adds a wrong HP:9
        pb = {"r1": {"HP:1"}}                    # PhenoBERT had HP:1
        out = marginal_gold_yield(pred, gold, pb)
        assert out == {"pb_available": True, "n_gold_gained": 1,
                       "n_gold_lost": 1, "n_extra_wrong": 1}

    def test_absent_baseline_is_flagged_not_credited(self):
        """Without the PhenoBERT baseline every hit would otherwise read as a gain over nothing."""
        out = marginal_gold_yield({"r1": {"HP:1"}}, {"r1": {"HP:1"}}, {})
        assert out["pb_available"] is False


# ── Cohort selection ─────────────────────────────────────────────────────────
class _Cfg(dict):
    """Minimal stand-in for the Hydra config: attribute access over a dict."""

    def get(self, k, default=None):
        return dict.get(self, k, default)

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None


class TestCohortSelection:
    """HCY has 118 reports and only 100 annotated. A blind head-20 can pick unannotated ones,
    which score zero precision for every prompt alike and quietly shrink the sample."""

    def _fixture(self):
        reports = {f"r{i:02d}": "text" for i in range(10)}
        gold = {r: (["HP:0000252"] if i % 2 == 0 else []) for i, r in enumerate(sorted(reports))}
        return reports, gold

    def test_unannotated_reports_are_skipped(self, tmp_path, caplog):
        import logging

        reports, gold = self._fixture()
        cfg = _Cfg(dataset="hcy", max_patients=3, require_annotated=True)
        ids = select_cohort(cfg, reports, gold, str(tmp_path), logging.getLogger("t"))
        assert ids == ["r00", "r02", "r04"]

    def test_filter_can_be_turned_off_for_the_full_run(self, tmp_path):
        """The generation run of the other prompts scores every report, as the Free Listing generation run does, filtering would raise precision for
        reasons unrelated to the prompt and break comparability with its table row."""
        import logging

        reports, gold = self._fixture()
        cfg = _Cfg(dataset="hcy", max_patients=3, require_annotated=False)
        ids = select_cohort(cfg, reports, gold, str(tmp_path), logging.getLogger("t"))
        assert ids == ["r00", "r01", "r02"]

    def test_cohort_is_written_for_later_stages(self, tmp_path):
        import logging

        reports, gold = self._fixture()
        cfg = _Cfg(dataset="hcy", max_patients=2, require_annotated=True)
        ids = select_cohort(cfg, reports, gold, str(tmp_path), logging.getLogger("t"))
        saved = json.loads((tmp_path / "screen_cohort.json").read_text())
        assert saved["report_ids"] == ids
        assert saved["n_reports"] == 2

    def test_selection_is_deterministic(self, tmp_path):
        """extract, aggregate and phase B all recompute this independently, they must agree."""
        import logging

        reports, gold = self._fixture()
        cfg = _Cfg(dataset="hcy", max_patients=4, require_annotated=True)
        first = select_cohort(cfg, reports, gold, str(tmp_path), logging.getLogger("t"))
        second = select_cohort(cfg, reports, gold, str(tmp_path), logging.getLogger("t"))
        assert first == second

    def test_cached_records_outside_the_cohort_are_excluded(self, tmp_path):
        """The recommended way to get ``p0_baseline`` free is to drop the Free Listing generation run's extractions in.

        Those cover all 118 HCY reports while every other prompt was generated over the screen's
        20. Unfiltered, the control's echo rate and label exactness would be measured on a
        different report set from its competitors', making the numbers that decide whether the
        ranking is trustworthy the least comparable in the table.
        """
        import logging

        from hpo_extraction.phenojury.generation_prompts import _load_prompt_caches, prompt_dir

        run_dir = tmp_path / "hcy"
        cell = prompt_dir(str(run_dir), "p0_baseline")
        import os

        os.makedirs(cell, exist_ok=True)
        with open(os.path.join(cell, "llm_extractions_llama.jsonl"), "w") as f:
            for rid in ("r00", "r01", "r99"):     # r99 is outside the screen cohort
                f.write(json.dumps({"model": "llama", "patient_id": rid, "sentence_number": 0,
                                    "sentence_text": SENTENCE, "llm_output": "Microcephaly"}) + "\n")
        with open(os.path.join(cell, "detections_llama.jsonl"), "w") as f:
            f.write(json.dumps({"report_id": "r00", "model": "llama", "sentence_number": 0,
                                "hpo_id": "HP:0000252", "count": 1}) + "\n")

        cfg = _Cfg(slm_dirs={"llama": "/some/path"})
        records, _, active, _ = _load_prompt_caches(
            cfg, str(run_dir), "p0_baseline", ["r00", "r01"], logging.getLogger("t")
        )
        assert active == ["llama"]
        assert {r["patient_id"] for r in records["llama"]} == {"r00", "r01"}

    def test_empty_cohort_fails_loudly(self, tmp_path):
        import logging

        cfg = _Cfg(dataset="hcy", max_patients=5, require_annotated=True)
        with pytest.raises(RuntimeError, match="gold annotation"):
            select_cohort(cfg, {"r1": "t"}, {"r1": []}, str(tmp_path), logging.getLogger("t"))


# ── Registration in the thesis table ─────────────────────────────────────────
class TestDiscoveryRegistration:
    def test_exp14_rows_exist_and_carry_a_prompt_subdir(self):
        from pathlib import Path

        import hpo_extraction.evaluation.result_tables.discovery as discovery

        rows = [m for m in discovery.METHODS if m.exp_id == "phenojury_generation_other_prompts"]
        assert len(rows) == len(discovery.EXP14_PROMPT_ROWS)
        for row in rows:
            assert row.subdir, "an exp14 row without a subdir would collide with its sibling"
            assert row.run_name_prefix.endswith(row.subdir), (
                "cost attribution substring-matches the prompt before the cohort"
            )

    def test_exp13_rows_keep_an_empty_subdir(self):
        """The field was appended last so existing positional ``mlflow_experiment`` args still
        bind correctly, this is what asserts that."""
        from pathlib import Path

        import hpo_extraction.evaluation.result_tables.discovery as discovery

        spec = discovery.METHODS_BY_KEY["slm_ensemble"]
        assert spec.subdir == ""
        assert spec.mlflow_experiment == "exp13_comparison"


# ── The an earlier exploratory run round-two prompts ───────────────────────────────────────────




class TestRoundTwoIsVerbatim:
    """The q-family must be ``context/prompt_engineering.txt``, character for character.

    Editing somebody's prompt to satisfy this repo's house rules would make the screen measure the
    edit. So the rules are *asserted broken* below, not fixed, the violations are real,
    they are known, and the ranking table has a column for each of them.
    """



    def test_q1_and_q2_differ_only_in_line_order(self):
        """The pair's whole value. If anything else differs, it is no longer a position test."""
        q1, q2 = get_prompt("q1_ontology_lines"), get_prompt("q2_sentence_last")
        assert q1.system == q2.system
        assert q1.user_template != q2.user_template
        assert sorted(q1.user_template.split("\n")) == sorted(q2.user_template.split("\n"))

    def test_q_family_does_not_forbid_numbering(self):
        """A KNOWN, DELIBERATE violation of the p-family's hardest-won layout rule.

        ``experiments/findings/exp13_phenobert_input_format.md`` measures a numbered list at 21.9 %
        recovery against 82.1 % for a terminated plain list. None of these templates says "do not
        number the lines", so any model that numbers anyway is floored at the low layout and its
        condition reports a layout result wearing a wording result's name.

        This test exists so that fact cannot be forgotten, and so that adding the clause later is a
        deliberate act (delete this test), not an accident. ``format_compliance`` is the
        column that detects it happening.
        """
        for key in Q_PROMPT_KEYS:
            spec = get_prompt(key)
            body = spec.system + spec.user_template
            assert "not number" not in body and "Do not number" not in body, (
                f"{key} now forbids numbering — delete this test and move it into the p-family rule"
            )

    def test_every_q_sentinel_is_filtered_before_grounding(self):
        """The sentinels are negation vocabulary, so they must never reach PhenoBERT.

        ``NONE`` and ``NO FINDINGS`` would each be discarded as ``Neg`` anyway, but relying on that
        would make an empty reply indistinguishable from a grounding failure in the diagnostics.
        ``wrote_something`` is what has to catch them, and q4's ``{"findings": []}`` is the one
        that needs a rule of its own.
        """
        from hpo_extraction.phenojury.ensemble_eval import wrote_something

        for sentinel in ("NONE", "NO FINDINGS", '{"findings": []}', "no phenotype"):
            assert not wrote_something(sentinel), sentinel

    def test_every_q_prompt_renders_without_a_format_error(self):
        """q4's SYSTEM block contains a JSON schema with braces.

        ``render`` formats the user template only, so the braces are inert, but a future edit that
        moved the schema into the user turn would raise ``KeyError`` on the first GPU cell, four
        hours into an array job.
        """
        for key in Q_PROMPT_KEYS:
            assert SENTENCE in get_prompt(key).render(SENTENCE)


class TestLineShapes:
    """A prompt is judged on the part of its line that is supposed to carry the term.

    Before ``line_shape`` existed this was two hard-coded keys inside the metrics. The q-family
    adds three more layouts, and reading them wrong does not crash, it writes a 0.0 into the
    ranking table, where it is indistinguishable from a real failure.
    """

    def test_pipe_lines_are_judged_on_the_first_field(self):
        line = "Microcephaly | Small head circumference | Head smaller than normal"
        assert label_exactness([line], PHRASES, SHAPE_PIPE_FIRST) == pytest.approx(1.0)
        # The same line read whole is the bug this shape exists to prevent.
        assert label_exactness([line], PHRASES) == pytest.approx(0.0)

    def test_json_terms_are_extracted_from_the_object(self):
        line = '{"findings": [{"span": "small head", "term": "Microcephaly"}]}'
        assert label_exactness([line], PHRASES, SHAPE_JSON) == pytest.approx(1.0)

    def test_json_denominator_is_terms_not_lines(self):
        """Two terms on one line, one canonical: 0.5, not 0 and not 1."""
        line = ('{"findings": [{"span": "a", "term": "Microcephaly"}, '
                '{"span": "b", "term": "the boy had a small head"}]}')
        assert label_exactness([line], PHRASES, SHAPE_JSON) == pytest.approx(0.5)

    def test_near_valid_json_still_yields_its_terms(self):
        """An 8B model's JSON is often nearly valid. A strict parse would score the parser."""
        line = '{"findings": [{"span": "small head", "term": "Microcephaly"},'
        assert label_exactness([line], PHRASES, SHAPE_JSON) == pytest.approx(1.0)

    def test_a_rewriting_prompt_has_no_label_exactness(self):
        """None, not 0.0, the metric does not apply, and a zero would rank it as a failure."""
        assert label_exactness(["The patient has microcephaly."], PHRASES, SHAPE_SENTENCE) is None
        d = diagnose_records(
            [{"llm_output": "The patient has microcephaly.", "sentence_text": SENTENCE}],
            "q3_rewrite", PHRASES,
        )
        assert d["label_exactness"] is None

    def test_a_rewriting_prompt_is_compliant_when_it_wrote_one_sentence(self):
        assert format_compliance(["The patient has microcephaly and hypotonia."],
                                 "q3_rewrite") == pytest.approx(1.0)
        # Two lines is the failure mode for this contract, it asked for one.
        assert format_compliance(["The patient has microcephaly.", "Also hypotonia."],
                                 "q3_rewrite") == pytest.approx(0.0)

    def test_a_json_prompt_is_compliant_when_a_term_field_survives(self):
        assert format_compliance(['{"findings": [{"span": "a", "term": "Microcephaly"}]}'],
                                 "q4_span_json") == pytest.approx(1.0)
        assert format_compliance(["Here are the findings I identified:"],
                                 "q4_span_json") == pytest.approx(0.0)

    def test_pipe_compliance_requires_the_separator(self):
        assert format_compliance(["Microcephaly"], "q5_synonyms") == pytest.approx(0.0)
        assert format_compliance(["Microcephaly | Small head | Small head"],
                                 "q5_synonyms") == pytest.approx(1.0)

    def test_pipe_compliance_judges_length_on_the_term_not_the_line(self):
        """Three names is three times the text. A whole-line length rule would fail every q5 line
        for obeying its instructions."""
        line = ("Delayed gross motor development | Gross motor delay | "
                "Late to sit, crawl or walk without support")
        assert format_compliance([line], "q6_two_level") == pytest.approx(1.0)

    def test_plain_q_prompts_use_the_unchanged_rules(self):
        assert format_compliance(["1. Microcephaly"], "q1_ontology_lines") == pytest.approx(0.0)
        assert format_compliance(["Microcephaly"], "q7_recall") == pytest.approx(1.0)

    def test_an_unknown_key_falls_back_to_plain(self):
        """Cached cells outlive prompt renames. The metric must degrade, not raise."""
        assert format_compliance(["Microcephaly"], "q99_gone") == pytest.approx(1.0)


class TestNotApplicableMetricsSurviveAggregation:
    """A None diagnostic must reach the ranking table as a None.

    ``diagnose_records`` already refuses to call an inapplicable metric zero. The aggregation then
    averaged the cells unconditionally and died on ``int + NoneType``, after 64 GPU array tasks
    had already run, on the cheap CPU re-run at the end. The screen's whole output is a ranking, so
    the failure was maximally expensive and maximally late.
    """

    def test_all_none_cells_average_to_none_not_zero(self):
        rows = [{"label_exactness": None} for _ in range(8)]
        assert _mean_over_cells(rows, "label_exactness") is None

    def test_a_partial_column_averages_over_what_applies(self):
        """One model lost to a wall-clock kill must not drag the mean toward zero."""
        rows = [{"echo_rate": 0.2}, {"echo_rate": 0.4}, {"echo_rate": None}]
        assert _mean_over_cells(rows, "echo_rate") == pytest.approx(0.3)

    def test_no_cells_at_all_is_none_rather_than_a_zero_division(self):
        assert _mean_over_cells([], "echo_rate") is None

    def test_the_log_line_renders_a_missing_metric(self):
        assert _fmt_opt(None) == "n/a"
        assert _fmt_opt(0.6344) == "0.634"

    def test_the_csv_round_trip_nan_becomes_json_null(self):
        """pandas turns the None into NaN, and ``json.dump`` writes NaN as a bare token no other
        parser accepts, the verdict file would be readable only by the tool that wrote it."""
        assert _json_none(float("nan")) is None
        assert _json_none(None) is None
        assert _json_none(0.0) == 0.0
        assert json.loads(json.dumps({"x": _json_none(float("nan"))})) == {"x": None}
