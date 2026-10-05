"""Per-sentence attribution of PhenoBERT detections, with and without a patched install.

The Free Listing generation run ran eight models over two cohorts and logged `N sentences → 0 detections` every time:
``parse_phenobert_sentences`` required a ``sentence_count`` column that this PhenoBERT has never
written, so every one of ~800–2500 real detection rows per model was dropped. The parser now falls
back to the detection's character offsets against the input the pipeline itself generated.

What these tests fix down:

* the patched route (``sentence_count`` column) still wins when present;
* the offset route reproduces it on stock 5-column output;
* an offset that does *not* land on its reported phrase attributes **nothing**, a wrong sentence
  would be worse than a missing one, since the plurality rule votes per sentence;
* detections whose phrase contains a newline (an SLM answering with a numbered list) survive,
  because records are reassembled rather than read line by line;
* negation and non-HPO rows are still skipped.
"""

from __future__ import annotations

import logging
import os

import pytest

from hpo_extraction.phenojury import phenobert as phenobert_runner

from hpo_extraction.phenojury.phenobert import (
    build_phenobert_input,
    diagnose_phenobert_output,
    iter_phenobert_rows,
    parse_phenobert_sentences,
    phenobert_output_is_usable,
    phenobert_output_matches_input,
    write_phenobert_input,
)

pytestmark = pytest.mark.unit


RECORDS = [
    {"patient_id": "P1", "sentence_number": 0, "llm_output": "Seizures and ataxia"},
    {"patient_id": "P1", "sentence_number": 1, "llm_output": "Elevated plasma homocystine"},
    {"patient_id": "P2", "sentence_number": 0, "llm_output": "no phenotype"},
]


def _row(text, phrase, hpo, score="1.00", extra=()):
    """A stock 5-column PhenoBERT row for `phrase`, with true offsets into `text`."""
    start = text.index(phrase)
    return "\t".join([str(start), str(start + len(phrase)), phrase, hpo, score, *extra])


def _write_output(tmp_path, patient_id, rows):
    d = tmp_path / "phenobert_output_m"
    d.mkdir(exist_ok=True)
    (d / f"{patient_id}.txt").write_text("\n".join(rows) + "\n")
    return str(d)


# ── the input layout is one function, shared by writer and parser ─────────────

def test_build_and_write_agree(tmp_path):
    text, spans = build_phenobert_input(RECORDS, "P1")
    write_phenobert_input(RECORDS, ["P1"], str(tmp_path))
    assert (tmp_path / "P1.txt").read_text() == text
    assert [s[2] for s in spans] == [0, 1]
    # every span covers its own block and nothing else
    for start, end, sent_num in spans:
        assert text[start:end].startswith(f"sentence_number: {sent_num}")


def test_no_line_ends_on_a_word_character():
    """The property the layout exists to guarantee.

    ``process_text2phrases`` rewrites a newline that follows a word character as a bare period with
    no space after it, fusing the last token of a line into the first of the next, the fused phrase
    is never generated as a candidate and cannot be normalised (9.4 % of terms recovered against
    82.1 %. Experiments/findings/exp13_phenobert_input_format.md). Every line, the
    ``sentence_number`` marker included, must therefore end in punctuation.
    """
    records = [
        {"patient_id": "P1", "sentence_number": 0, "llm_output": "Microcephaly\nHypotonia"},
        {"patient_id": "P1", "sentence_number": 1, "llm_output": "- Seizures  \nAtaxia."},
    ]
    text, _ = build_phenobert_input(records, "P1")
    for line in text.split("\n"):
        if line:
            assert not line[-1].isalnum() and not line.endswith(" "), f"unterminated line: {line!r}"
    # and the terms themselves are still intact
    for term in ("Microcephaly", "Hypotonia", "Seizures", "Ataxia"):
        assert term in text


def test_stale_cache_from_another_layout_is_not_reused(tmp_path):
    """A cache whose offsets miss their phrases must be re-run, not parsed to nothing.

    ``phenobert_output_is_usable`` only asks whether detections exist, so on its own it green-lights
    a cache built from a different input layout, which then attributes nothing and reports zero
    detections, indistinguishable from "PhenoBERT found nothing".
    """
    text, _ = build_phenobert_input(RECORDS, "P1")
    out = _write_output(tmp_path, "P1", [_row(text, "Seizures", "HP:0001250")])
    inputs = {"P1": build_phenobert_input(RECORDS, "P1")}
    assert phenobert_output_matches_input(out, inputs) is True

    shifted = [{**r, "llm_output": "prefix " + r["llm_output"]} for r in RECORDS]
    assert phenobert_output_matches_input(out, {"P1": build_phenobert_input(shifted, "P1")}) is False



def test_partial_cache_from_a_crashed_run_is_not_reused(tmp_path):
    """Coverage, not just consistency, the failure that scored a prompt over 2 of 20 patients.

    A run killed part-way through annotation leaves a cache that is internally perfect: every row it
    did write lands on its own phrase. Checking content alone therefore green-lights it, and the
    cohort silently shrinks instead of the cache being rebuilt. An earlier exploratory run's medgemma cells died in
    stanza after two patients and the backfill scored p2_canonical over those two.
    """
    text, _ = build_phenobert_input(RECORDS, "P1")
    out = _write_output(tmp_path, "P1", [_row(text, "Seizures", "HP:0001250")])

    # P1 alone: complete, and reusable.
    assert phenobert_output_matches_input(out, {"P1": build_phenobert_input(RECORDS, "P1")}) is True

    # The same cache against a two-patient cohort is a *truncated* cache, and must be rejected even
    # though every row in it still verifies.
    two = {
        "P1": build_phenobert_input(RECORDS, "P1"),
        "P2": build_phenobert_input([{**r, "patient_id": "P2"} for r in RECORDS], "P2"),
    }
    assert phenobert_output_matches_input(out, two) is False



class TestCudaOomFallsBackToCpu:
    """A pathological line must cost speed, not the cell.

    stanza's dependency parser sizes its allocation from the longest sentence, so one unbroken line
    of a few thousand tokens asked for 22.6 GiB on a 23.5 GiB card and killed an earlier exploratory run's
    p3_two_column x medgemma cell. PhenoBERT and stanza both choose their device from
    torch.cuda.is_available(), so hiding the GPUs reruns the identical models at the identical
    thresholds on the CPU, the same annotation, produced slowly.
    """

    OOM = "RuntimeError: CUDA out of memory. Tried to allocate 22.59 GiB"

    def _spy(self, monkeypatch, returncodes):
        """Patch subprocess.run to fail with an OOM first, then succeed. Records each env."""
        calls = []

        class R:
            def __init__(self, rc, err):
                self.returncode, self.stdout, self.stderr = rc, "", err

        def fake_run(cmd, cwd=None, env=None, **kw):
            calls.append({"cuda": (env or {}).get("CUDA_VISIBLE_DEVICES", "<unset>")})
            rc = returncodes[len(calls) - 1]
            return R(rc, self.OOM if rc else "")

        monkeypatch.setattr(phenobert_runner.subprocess, "run", fake_run)
        return calls

    def _invoke(self, tmp_path):
        (tmp_path / "annotate.py").write_text("")
        phenobert_runner.run_phenobert(
            phenobert_dir=str(tmp_path),
            input_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
        )

    def test_an_oom_is_retried_with_the_gpus_hidden(self, tmp_path, monkeypatch, caplog):
        calls = self._spy(monkeypatch, [1, 0])
        with caplog.at_level(logging.WARNING):
            self._invoke(tmp_path)
        assert len(calls) == 2, "expected exactly one retry"
        assert calls[0]["cuda"] == "<unset>", "first attempt must use the GPU"
        assert calls[1]["cuda"] == "", "retry must hide the GPUs"
        assert "retrying on CPU" in caplog.text
        # The operator must know the numbers are unaffected, or they will distrust the cell.
        assert "unchanged" in caplog.text

    def test_a_successful_run_is_never_retried(self, tmp_path, monkeypatch):
        calls = self._spy(monkeypatch, [0])
        self._invoke(tmp_path)
        assert len(calls) == 1
        assert calls[0]["cuda"] == "<unset>"

    def test_a_non_oom_failure_is_not_retried_but_raised(self, tmp_path, monkeypatch):
        """Retrying an unrelated crash on CPU just wastes an allocation failing again."""
        class R:
            returncode, stdout, stderr = 1, "", "ValueError: something else entirely"

        calls = []
        monkeypatch.setattr(phenobert_runner.subprocess, "run",
                            lambda *a, **k: (calls.append(1), R())[1])
        (tmp_path / "annotate.py").write_text("")
        with pytest.raises(RuntimeError, match="exited with code 1"):
            self._invoke(tmp_path)
        assert len(calls) == 1

    def test_an_oom_that_fails_again_on_cpu_still_raises(self, tmp_path, monkeypatch):
        calls = self._spy(monkeypatch, [1, 1])
        with pytest.raises(RuntimeError, match="exited with code 1"):
            self._invoke(tmp_path)
        assert len(calls) == 2


class TestHostOomSplitsTheInputDirectory:
    """An earlier exploratory run D1: a verbose prompt makes PhenoBERT exceed the node's RAM and the cgroup OOM
    killer sends SIGKILL, exit -9, empty stderr, no traceback the CUDA markers can see.

    The answer is fewer files per subprocess, not a bigger allocation: annotate.py annotates each
    input file independently into its own output file, so the union of chunk runs is what one pass
    would have written. These tests fix that the retry fires only on an OOM kill, that it covers
    every input file once, and that a chunk which still dies names its members.
    """

    def _inputs(self, tmp_path, n):
        (tmp_path / "annotate.py").write_text("")
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        for i in range(n):
            (in_dir / f"P{i:03d}.txt").write_text(f"report {i}")
        return in_dir

    def _spy(self, monkeypatch, returncodes, stderrs=None):
        calls = []

        class R:
            def __init__(self, rc, err):
                self.returncode, self.stdout, self.stderr = rc, "", err

        def fake_run(cmd, cwd=None, env=None, **kw):
            in_dir = cmd[cmd.index("-i") + 1]
            calls.append(sorted(os.listdir(in_dir)))
            i = len(calls) - 1
            return R(returncodes[i], (stderrs or [""] * len(returncodes))[i])

        monkeypatch.setattr(phenobert_runner.subprocess, "run", fake_run)
        return calls

    def test_an_oom_kill_is_retried_in_chunks_covering_every_file(
        self, tmp_path, monkeypatch, caplog
    ):
        in_dir = self._inputs(tmp_path, 10)
        calls = self._spy(monkeypatch, [-9, 0, 0, 0, 0])
        with caplog.at_level(logging.WARNING):
            phenobert_runner.run_phenobert(
                phenobert_dir=str(tmp_path), input_dir=str(in_dir),
                output_dir=str(tmp_path / "out"), chunk_size=3,
            )
        assert len(calls) == 5, "one failed pass plus ceil(10/3) chunks"
        assert calls[0] == sorted(os.listdir(in_dir)), "the first attempt sees the whole directory"
        chunked = calls[1:]
        assert [len(c) for c in chunked] == [3, 3, 3, 1]
        # Every report annotated once, a dropped or duplicated file is a silent ground truth loss.
        assert sorted(f for c in chunked for f in c) == sorted(os.listdir(in_dir))
        assert "OOM-killed" in caplog.text

    def test_the_chunks_write_into_the_same_output_directory(self, tmp_path, monkeypatch):
        """Chunking must be a memory transform only: the detections still land in one place."""
        in_dir = self._inputs(tmp_path, 4)
        out_dirs = []

        class R:
            def __init__(self, rc):
                self.returncode, self.stdout, self.stderr = rc, "", ""

        codes = iter([-9, 0, 0])
        monkeypatch.setattr(
            phenobert_runner.subprocess, "run",
            lambda cmd, **kw: (out_dirs.append(cmd[cmd.index("-o") + 1]), R(next(codes)))[1],
        )
        phenobert_runner.run_phenobert(
            phenobert_dir=str(tmp_path), input_dir=str(in_dir),
            output_dir=str(tmp_path / "out"), chunk_size=2,
        )
        assert len(set(out_dirs)) == 1 and out_dirs[0] == str(tmp_path / "out")

    def test_a_chunk_that_still_dies_names_its_members(self, tmp_path, monkeypatch):
        """At this size the culprit is one report's text, the operator needs to know which."""
        in_dir = self._inputs(tmp_path, 4)
        self._spy(monkeypatch, [-9, 0, -9])
        with pytest.raises(RuntimeError, match=r"P002\.txt"):
            phenobert_runner.run_phenobert(
                phenobert_dir=str(tmp_path), input_dir=str(in_dir),
                output_dir=str(tmp_path / "out"), chunk_size=2,
            )

    def test_an_ordinary_failure_is_not_chunked(self, tmp_path, monkeypatch):
        """Splitting a ValueError just fails N more times and hides the original stderr."""
        in_dir = self._inputs(tmp_path, 4)
        calls = self._spy(monkeypatch, [1], ["ValueError: something else entirely"])
        with pytest.raises(RuntimeError, match="exited with code 1"):
            phenobert_runner.run_phenobert(
                phenobert_dir=str(tmp_path), input_dir=str(in_dir),
                output_dir=str(tmp_path / "out"), chunk_size=2,
            )
        assert len(calls) == 1

    def test_chunk_size_zero_keeps_the_old_behaviour(self, tmp_path, monkeypatch):
        in_dir = self._inputs(tmp_path, 4)
        calls = self._spy(monkeypatch, [-9])
        with pytest.raises(RuntimeError, match="exited with code -9"):
            phenobert_runner.run_phenobert(
                phenobert_dir=str(tmp_path), input_dir=str(in_dir),
                output_dir=str(tmp_path / "out"), chunk_size=0,
            )
        assert len(calls) == 1


def test_build_matches_ids_across_types():
    """Report ids arrive as ints from some loaders and strings after a JSON round-trip."""
    records = [{"patient_id": 42, "sentence_number": 0, "llm_output": "Ataxia"}]
    text, spans = build_phenobert_input(records, "42")
    assert "Ataxia" in text and len(spans) == 1


# ── attribution routes ───────────────────────────────────────────────────────

def test_offsets_attribute_stock_five_column_output(tmp_path):
    text, _ = build_phenobert_input(RECORDS, "P1")
    rows = [
        _row(text, "Seizures", "HP:0001250"),
        _row(text, "ataxia", "HP:0001251"),
        _row(text, "Elevated plasma homocystine", "HP:0002160"),
    ]
    out = _write_output(tmp_path, "P1", rows)

    got = parse_phenobert_sentences(out, inputs={"P1": build_phenobert_input(RECORDS, "P1")})
    assert got == {"P1": {0: {"HP:0001250": 1, "HP:0001251": 1}, 1: {"HP:0002160": 1}}}


def test_sentence_count_column_wins_when_present(tmp_path):
    """A patched install keeps working, and its column is trusted over the offsets."""
    text, _ = build_phenobert_input(RECORDS, "P1")
    # disagree: column 6 says sentence 1, the offset says sentence 0
    rows = [_row(text, "Seizures", "HP:0001250", extra=("1",))]
    out = _write_output(tmp_path, "P1", rows)

    got = parse_phenobert_sentences(out, inputs={"P1": build_phenobert_input(RECORDS, "P1")})
    assert got == {"P1": {1: {"HP:0001250": 1}}}


def test_offsets_from_a_different_cache_attribute_nothing(tmp_path):
    """Offsets that don't land on their phrase must be refused, not guessed."""
    text, _ = build_phenobert_input(RECORDS, "P1")
    rows = [_row(text, "Seizures", "HP:0001250")]
    out = _write_output(tmp_path, "P1", rows)

    other = [{"patient_id": "P1", "sentence_number": 0,
              "llm_output": "completely different extraction text"}]
    got = parse_phenobert_sentences(out, inputs={"P1": build_phenobert_input(other, "P1")})
    assert got == {"P1": {}}


def test_no_inputs_means_no_offset_route(tmp_path):
    """Callers that pass nothing (scripts/compute_ensemble_metrics.py) keep the old behaviour."""
    text, _ = build_phenobert_input(RECORDS, "P1")
    out = _write_output(tmp_path, "P1", [_row(text, "Seizures", "HP:0001250")])
    assert parse_phenobert_sentences(out) == {"P1": {}}


def test_whitespace_normalised_phrase_still_attributes(tmp_path):
    """PhenoBERT prints a normalised phrase. The span in the text may differ in whitespace."""
    records = [{"patient_id": "P1", "sentence_number": 0, "llm_output": "Psychomotor  retardation"}]
    text, _ = build_phenobert_input(records, "P1")
    start = text.index("Psychomotor")
    end = start + len("Psychomotor  retardation")
    rows = ["\t".join([str(start), str(end), "Psychomotor retardation", "HP:0025356", "1.00"])]
    out = _write_output(tmp_path, "P1", rows)

    got = parse_phenobert_sentences(out, inputs={"P1": build_phenobert_input(records, "P1")})
    assert got == {"P1": {0: {"HP:0025356": 1}}}


# ── row reassembly, negation, junk ───────────────────────────────────────────

def test_phrase_containing_a_newline_survives(tmp_path):
    """A numbered-list answer makes PhenoBERT emit a phrase with a newline in it."""
    records = [{"patient_id": "P1", "sentence_number": 0, "llm_output": "1\nSeizures\n2\nAtaxia"}]
    text, _ = build_phenobert_input(records, "P1")
    rows = [_row(text, "1;\nSeizures", "HP:0001250"), _row(text, "Ataxia", "HP:0001251")]
    out = _write_output(tmp_path, "P1", rows)

    got = parse_phenobert_sentences(out, inputs={"P1": build_phenobert_input(records, "P1")})
    assert got == {"P1": {0: {"HP:0001250": 1, "HP:0001251": 1}}}

    # and the raw iterator sees two records, not three lines
    raw = (tmp_path / "phenobert_output_m" / "P1.txt").read_text()
    assert len(list(iter_phenobert_rows(raw))) == 2


def test_negated_and_non_hpo_rows_are_skipped(tmp_path):
    text, _ = build_phenobert_input(RECORDS, "P1")
    rows = [
        _row(text, "Seizures", "HP:0001250", extra=("Neg",)),
        "\t".join(["0", "5", "junk", "NOT_AN_ID", "1.00"]),
        _row(text, "ataxia", "HP:0001251"),
    ]
    out = _write_output(tmp_path, "P1", rows)

    got = parse_phenobert_sentences(out, inputs={"P1": build_phenobert_input(RECORDS, "P1")})
    assert got == {"P1": {0: {"HP:0001251": 1}}}


def test_repeated_phrase_counts_per_sentence(tmp_path):
    records = [{"patient_id": "P1", "sentence_number": 0, "llm_output": "Ataxia, ataxia"}]
    text, _ = build_phenobert_input(records, "P1")
    first = text.index("Ataxia")
    second = text.index("ataxia", first + 1)
    rows = [
        "\t".join([str(first), str(first + 6), "Ataxia", "HP:0001251", "1.00"]),
        "\t".join([str(second), str(second + 6), "ataxia", "HP:0001251", "1.00"]),
    ]
    out = _write_output(tmp_path, "P1", rows)

    got = parse_phenobert_sentences(out, inputs={"P1": build_phenobert_input(records, "P1")})
    assert got == {"P1": {0: {"HP:0001251": 2}}}


# ── cache-usability check ────────────────────────────────────────────────────

def test_stock_output_counts_as_a_usable_cache(tmp_path):
    """Stock 5-column output must not force a PhenoBERT re-run: offsets can attribute it."""
    text, _ = build_phenobert_input(RECORDS, "P1")
    out = _write_output(tmp_path, "P1", [_row(text, "Seizures", "HP:0001250")])
    assert phenobert_output_is_usable(out) is True
    diag = diagnose_phenobert_output(out)
    assert diag["n_hp_rows"] == 1 and diag["n_indexed_rows"] == 0


def test_empty_output_is_not_a_usable_cache(tmp_path):
    out = _write_output(tmp_path, "P1", [""])
    assert phenobert_output_is_usable(out) is False
