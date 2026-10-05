"""The AutoPCR 8B baseline / the AutoPCR 70B baseline, the parts of the AutoPCR wrapper that are ours to get wrong.

Everything between the staged corpus and the output TSV is upstream's `run_gsc_test_ner`, called
unmodified, and it needs faiss + a GPU. What is testable, and what would silently produce a wrong
number rather than a crash, is the boundary on either side of it:

**Offsets survive staging.** AutoPCR's corpus format is one line per document, so a multi-line HCY
report cannot be staged verbatim the way `phenobert_experiment.stage_reports` stages one. Every
character offset in `autopcr_detections.jsonl`, and therefore an earlier exploratory run's whole sentence
attribution, rests on that substitution being length-preserving. A `"\r\n" -> " "` would shorten
the text by one character per line and shift every offset after it, which no assertion downstream
would catch.

**The summary line exists for a report that predicted nothing.** `result_tables/loaders.load_predictions`
reads only the summary line, so a report omitted because it had no predictions simply vanishes from
the denominator, precision inflated by the reports that said nothing.

**`linked_by` is the measurement.** AutoPCR's contribution over a retrieval baseline is the subset
of mentions its language model chose, marked by a score of `-1.0`. If that sentinel were
read as a similarity the entire finding would be miscounted, silently.

The ontology-adapter tests need the full NLTK corpora (punkt_tab / averaged_perceptron_tagger_eng /
wordnet), which this repo does not vendor, `resources/nltk_data/` carries stopwords only. They skip
with that reason, not passing vacuously.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]

from hpo_extraction.baselines.autopcr_experiment import (  # noqa: E402
    collect_detections,
    predicted_sets,
    write_predictions,
    write_segments,
)
from hpo_extraction.baselines.autopcr_runner import (  # noqa: E402
    LLM_CHOSEN_SCORE,
    LocalPrompter,
    diagnose_autopcr_output,
    flatten_for_corpus,
    invalidate_stale_phrase_cache,
    link_source,
    normalise_code,
    parse_autopcr_tsv,
    stage_corpus,
)

pytestmark = pytest.mark.unit


def _nltk_ready() -> bool:
    """Whether the three NLTK calls the adapter makes actually work.

    Checked by calling them, not by looking up resource names. On ``nltk==3.9.1``, this project's
    fix, ``word_tokenize`` needs ``punkt_tab`` and ``pos_tag`` needs
    ``averaged_perceptron_tagger_eng``. A machine carrying only the older ``punkt`` /
    ``averaged_perceptron_tagger`` passes a name lookup and then raises at the first call. An
    earlier version of this helper did that, and skipped on a machine that was fine while
    claiming readiness on one that was not.
    """
    import nltk

    import hpo_extraction.ontology.nltk_data  # noqa: F401

    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    try:
        word_tokenize("short stature")
        nltk.pos_tag(["short", "stature"])
        WordNetLemmatizer().lemmatize("seizures")
    except LookupError:
        return False
    return True


class _FakeLLM:
    """Records what it was asked and re-runs canned answers in order."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.seen: list[tuple[str, str, dict]] = []

    def generate(self, prompt, system_prompt, **kwargs):
        self.seen.append((prompt, system_prompt, kwargs))
        return self.answers.pop(0)


# ── Staging: the offsets have to survive ─────────────────────────────────────
class TestStaging:
    def test_flatten_preserves_length_and_therefore_offsets(self):
        text = "Krampfanfälle seit Geburt.\nKleinwuchs.\r\nMikrozephalie.\rEnde."
        flat = flatten_for_corpus(text)
        assert len(flat) == len(text)
        assert "\n" not in flat and "\r" not in flat
        # The character at every offset is either unchanged or was a line terminator.
        for i, (a, b) in enumerate(zip(text, flat)):
            assert a == b or (a in "\r\n" and b == " "), i
        # And the substring an offset pair names still reads the same word.
        start = text.index("Kleinwuchs")
        assert flat[start:start + len("Kleinwuchs")] == "Kleinwuchs"

    def test_corpus_round_trips_ids_and_text(self, tmp_path):
        reports = {"P:01": "line one\nline two", "P:02": "single line"}
        path = tmp_path / "corpus_test.tsv"
        pmid_to_id = stage_corpus(reports, ["P:01", "P:02"], str(path))

        assert pmid_to_id == {"P:01": "P:01", "P:02": "P:02"}
        blocks = path.read_text(encoding="utf-8").strip().split("\n\n")
        assert [b.split("\n")[0] for b in blocks] == ["P:01", "P:02"]
        # two lines per block: upstream reads lines[0] as the pmid and lines[1] as the
        # whole document. A report that kept its line breaks would be truncated at the first one.
        assert all(len(b.split("\n")) == 2 for b in blocks)
        assert blocks[0].split("\n")[1] == "line one line two"

    def test_id_with_a_newline_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="newline or tab"):
            stage_corpus({"a\nb": "x"}, ["a\nb"], str(tmp_path / "c.tsv"))

    def test_duplicate_id_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate"):
            stage_corpus({"P:01": "x"}, ["P:01", "P:01"], str(tmp_path / "c.tsv"))

    def test_a_pilot_run_does_not_poison_the_full_runs_phrase_cache(self, tmp_path):
        """The failure this guards is a `max_patients=2` pilot followed by the real run.

        `run_gsc_test_ner` reuses `phrases.json` whenever the file exists, without checking which
        documents it holds, so the full run would load a two-document cache and KeyError on the
        third report. A matching cache is still reused: phrase extraction is the Stanza pass.
        """
        corpus_dir = tmp_path / "autopcr_corpus"
        corpus_dir.mkdir()
        cache = corpus_dir / "phrases.json"
        cache.write_text(json.dumps({"P:01": {"phrases": []}}), encoding="utf-8")

        assert invalidate_stale_phrase_cache(str(corpus_dir), {"P:01"}) == []
        assert cache.is_file(), "a cache covering exactly the staged corpus must be kept"

        assert invalidate_stale_phrase_cache(str(corpus_dir), {"P:01", "P:02"}) == ["phrases.json"]
        assert not cache.exists()

    def test_an_unreadable_phrase_cache_is_stale(self, tmp_path):
        corpus_dir = tmp_path / "autopcr_corpus"
        corpus_dir.mkdir()
        (corpus_dir / "phrases.json").write_text("{not json", encoding="utf-8")
        assert invalidate_stale_phrase_cache(str(corpus_dir), {"P:01"}) == ["phrases.json"]


# ── Reading AutoPCR's output ─────────────────────────────────────────────────
_TSV = (
    "P:01\n"
    "seizures and short stature\n"
    "0\t8\tseizures\tHP:0001250\t0.97\n"
    "13\t26\tshort stature\tHP_0004322\t-1.0\n"
    "\n"
    "P:02\n"
    "nothing here\n"
    "\n"
    "P:03\n"
    "obsolete mention\n"
    "0\t8\tobsolete\tHP:0001275\t0.99\n"
)


class TestParsing:
    def test_normalise_code_accepts_both_forms(self):
        assert normalise_code("HP_0001250") == "HP:0001250"
        assert normalise_code(" HP:0001250 ") == "HP:0001250"

    def test_parse_yields_every_annotation_row(self):
        rows = list(parse_autopcr_tsv(_TSV))
        assert [(r[0], r[4]) for r in rows] == [
            ("P:01", "HP:0001250"), ("P:01", "HP:0004322"), ("P:03", "HP:0001275"),
        ]
        # start/end/phrase are carried through untouched, they are the only tie back to the text.
        assert rows[1][1:4] == (13, 26, "short stature")

    def test_malformed_rows_are_skipped_not_crashed(self):
        raw = "P:01\ntext\nnot\ta\trow\nx\ty\tz\tHP:0001250\n"
        rows = list(parse_autopcr_tsv(raw))
        assert rows == []

    def test_link_source_reads_the_sentinel_not_a_similarity(self):
        assert link_source(LLM_CHOSEN_SCORE) == "llm"
        assert link_source(-1.0) == "llm"
        assert link_source(0.97) == "retrieval"
        assert link_source(1.0) == "dictionary"   # abbreviation / first-word shortcut
        assert link_source(None) == "unknown"

    def test_diagnose_counts_hp_rows(self, tmp_path):
        p = tmp_path / "out.tsv"
        p.write_text(_TSV, encoding="utf-8")
        diag = diagnose_autopcr_output(str(p))
        assert diag["n_blocks"] == 3
        assert diag["n_hp_rows"] == 3
        assert diag["n_unparsed"] == 0

    def test_diagnose_reports_zero_for_an_output_with_no_codes(self, tmp_path):
        p = tmp_path / "out.tsv"
        p.write_text("P:01\ntext only\n\nP:02\nalso text\n", encoding="utf-8")
        # This is the shape that must become SystemExit(3) in the driver: files present, blocks
        # parsed, nothing grounded.
        assert diagnose_autopcr_output(str(p))["n_hp_rows"] == 0


# ── Detections and the predictions contract ──────────────────────────────────
@pytest.fixture(scope="module")
def hpo_tree():
    from hpo_extraction.ontology.hpo_tree import HPOTree

    return HPOTree()


class TestDetectionsAndPredictions:
    def test_collect_marks_the_llm_linked_row_and_remaps_alt_ids(self, tmp_path, hpo_tree):
        p = tmp_path / "out.tsv"
        p.write_text(_TSV, encoding="utf-8")
        records, counts = collect_detections(
            str(p), {"P:01": "P:01", "P:02": "P:02", "P:03": "P:03"}, hpo_tree)

        assert counts["n_rows"] == 3
        assert counts["n_llm_linked"] == 1
        by_phrase = {r["phrase"]: r for r in records}
        assert by_phrase["seizures"]["linked_by"] == "retrieval"
        assert by_phrase["short stature"]["linked_by"] == "llm"
        # HP:0001275 is an alt_id of HP:0001250 (Seizure) in the fixed release: the raw code is
        # kept alongside the resolved one, not being overwritten.
        obsolete = by_phrase["obsolete"]
        assert obsolete["raw_hpo_id"] == "HP:0001275"
        assert obsolete["hpo_id"] == "HP:0001250"
        assert obsolete["resolved"] is True
        assert counts["n_alt_id"] == 1

    def test_unknown_pmid_is_counted_not_attributed(self, tmp_path, hpo_tree):
        p = tmp_path / "out.tsv"
        p.write_text(_TSV, encoding="utf-8")
        records, counts = collect_detections(str(p), {"P:01": "P:01"}, hpo_tree)
        assert counts["n_unknown_pmid"] == 1          # P:03's row
        assert {r["report_id"] for r in records} == {"P:01"}

    def test_unresolved_codes_are_kept_but_never_predicted(self, tmp_path, hpo_tree):
        p = tmp_path / "out.tsv"
        p.write_text("P:01\ntext\n0\t4\tterm\tHP:9999999\t-1.0\n", encoding="utf-8")
        records, counts = collect_detections(str(p), {"P:01": "P:01"}, hpo_tree)
        assert counts["n_unmapped"] == 1
        assert records[0]["resolved"] is False
        # A linker that answered with an id outside the menu is a finding in the detections file,
        # but it must not reach the predicted set.
        assert predicted_sets(records, ["P:01"]) == {"P:01": []}

    def test_summary_line_is_written_for_a_report_that_predicted_nothing(self, tmp_path, hpo_tree):
        path = tmp_path / "autopcr_predictions.jsonl"
        n = write_predictions(
            str(path), ["P:01", "P:02"],
            {"P:01": ["HP:0001250"]}, {"P:01": ["HP:0001250"], "P:02": ["HP:0004322"]}, hpo_tree)

        recs = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
        summaries = {r["report_id"]: r for r in recs if r.get("summary")}
        assert set(summaries) == {"P:01", "P:02"}, "a silent report must still reach the scorer"
        assert summaries["P:02"]["predicted_set"] == []
        assert summaries["P:02"]["gold_set"] == ["HP:0004322"]
        assert n == 1
        term_rows = [r for r in recs if not r.get("summary")]
        assert term_rows[0]["prediction"] == 1 and term_rows[0]["ground_truth"] == 1


class TestSegments:
    def test_only_accepted_phrases_are_written_and_rank_1_is_the_verdict(self, tmp_path, hpo_tree):
        calls = [(
            ["seizures", "the"],
            [
                [("HP:0001250", 0.97), ("HP:0002373", 0.91)],   # accepted: top1 >= tau_1
                [("HP:0004322", 0.40), ("HP:0001250", 0.31)],   # rejected: below tau_1
            ],
        )]
        path = tmp_path / "seg.jsonl"
        n = write_segments(str(path), ["P:01"], calls, 0.95, hpo_tree)

        rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
        assert n == 2 and {r["text"] for r in rows} == {"seizures"}
        assert [r["rank"] for r in rows] == [1, 2]
        assert [r["slm_verdict"] for r in rows] == ["Yes", "No"]

    def test_llm_linked_phrase_is_accepted_on_the_negative_sentinel(self, tmp_path, hpo_tree):
        calls = [(["short stature"], [[("HP_0004322", -1.0), ("HP:0004325", 0.88)]])]
        path = tmp_path / "seg.jsonl"
        assert write_segments(str(path), ["P:01"], calls, 0.95, hpo_tree) == 2
        rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
        # abs(-1.0) >= tau_1, which is how upstream accepts it, and the code is normalised.
        assert rows[0]["hpo_id"] == "HP:0004322" and rows[0]["slm_verdict"] == "Yes"


# ── The linking client ───────────────────────────────────────────────────────
class TestLocalPrompter:
    def test_reproduces_upstreams_call_signature(self):
        llm = _FakeLLM(["answer: HP_0001250\nconfidence: HIGH"])
        out = LocalPrompter(llm, max_new_tokens=32)(
            {7: {"system": "SYS", "user": "seizures"}}, "some-model", None, None, 0)
        assert out == {7: "answer: HP_0001250\nconfidence: HIGH"}
        prompt, system, kwargs = llm.seen[0]
        # nn_model2 puts the candidate menu in the SYSTEM prompt and the entity in the user turn.
        assert (prompt, system) == ("seizures", "SYS")
        assert kwargs == {"max_new_tokens": 32}

    def test_counts_calls_for_the_timing_artifact(self):
        llm = _FakeLLM(["a", "b"])
        p = LocalPrompter(llm)
        p({0: {"system": "s", "user": "u"}, 1: {"system": "s", "user": "v"}})
        assert p.n_calls == 2

    def test_refuses_a_hosted_provider(self):
        p = LocalPrompter(_FakeLLM([]))
        with pytest.raises(ValueError, match="patient-data boundary"):
            p({0: {"system": "s", "user": "u"}}, "gpt-4", "openai", "sk-...", 0)


# ── The ontology adapter ─────────────────────────────────────────────────────
@pytest.mark.skipif(not _nltk_ready(),
                    reason="needs NLTK punkt_tab / averaged_perceptron_tagger_eng / wordnet — "
                           "resources/nltk_data/ vendors stopwords only; see "
                           "src/AutoPCR/README.md, 'Staging the assets'")
class TestOntologyAdapter:
    """The dictionary must be the fixed ontology, not AutoPCR's own HPO release."""

    def test_subtree_matches_the_repos_own_reading(self):
        from hpo_extraction.ontology.hpo_tree import HPOTree

        sys.path.insert(0, str(_REPO / "third_party" / "AutoPCR"))
        from build_dict_from_hpo_json import PHENOTYPIC_ABNORMALITY, subtree_ids
        from hpo_extraction.retrieval.surface_index import build_surface_index

        tree = HPOTree()
        _, hpo2surfaces = build_surface_index(tree)
        # build_surface_index counts HP:0000118 itself. Upstream's get_all_child only recurses into
        # The root's children. Compared as sets modulo that one id.
        assert set(subtree_ids(tree)) == set(hpo2surfaces) - {PHENOTYPIC_ABNORMALITY}

    def test_a_known_term_round_trips(self, tmp_path):
        sys.path.insert(0, str(_REPO / "third_party" / "AutoPCR"))
        from build_dict_from_hpo_json import build, cross_check
        from hpo_extraction.ontology.hpo_tree import DEFAULT_HPO_JSON

        out = tmp_path / "HPO"
        hpo_obo = build(str(DEFAULT_HPO_JSON), str(out))
        cross_check(str(out), str(DEFAULT_HPO_JSON))

        seizure = hpo_obo["HP:0001250"]
        assert seizure["name"][0] == "seizure"
        assert seizure["is_a"] == ["HP:0012638"]
        synonyms = {s[0] for s in seizure["synonym"]}
        assert {"epilepsy", "epileptic seizure", "seizures"} <= synonyms
        # The label must not be repeated as its own synonym: hpo.json sometimes lists it, an .obo
        # synonym: line never does, and a duplicate would be a duplicate vector in the index.
        assert "intellectual disability" not in {
            s[0] for s in hpo_obo["HP:0001249"]["synonym"]}

        vocab = (out / "lable.vocab").read_text(encoding="utf-8").split()
        assert vocab[-1] == "HP:None"                # upstream's neg label
        for name in ("obo.json", "abbr.json", "noabb_lemma.dic", "word_id_map.json",
                     "firstword_id_map.json", "id_word_map.json", "alt_hpoid.json"):
            assert (out / name).is_file(), name

    def test_abbreviation_rule_is_upstreams(self, tmp_path):
        sys.path.insert(0, str(_REPO / "third_party" / "AutoPCR"))
        import json as _json

        from build_dict_from_hpo_json import build
        from hpo_extraction.ontology.hpo_tree import DEFAULT_HPO_JSON

        out = tmp_path / "HPO"
        build(str(DEFAULT_HPO_JSON), str(out))
        abbr = _json.loads((out / "abbr.json").read_text(encoding="utf-8"))
        assert abbr, "no all-caps short synonyms were collected at all"
        assert all(k == k.lower() and len(k) < 10 for k in abbr)
        assert all(v.startswith("HP:") for v in abbr.values())


# ── The constituency-parse path (ee="neural++"), split across two environments ──
@pytest.fixture
def autopcr_modules():
    """Import AutoPCR's flat modules (``HPO_evaluation``, ``ee``, ``evaluate``, ``utils``) cleanly.

    Their names are generic: HuggingFace ships an ``evaluate`` package, and another test may already
    have imported something called ``utils``. Anything under those names not from src/AutoPCR is
    set aside for the test and put back afterwards, so the import below cannot resolve to it.
    """
    autopcr_dir = str(_REPO / "third_party" / "AutoPCR")
    names = ("HPO_evaluation", "evaluate", "ee", "utils")
    saved = {k: v for k, v in sys.modules.items()
             if k.split(".")[0] in names and autopcr_dir not in str(getattr(v, "__file__", ""))}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, autopcr_dir)
    try:
        import HPO_evaluation

        yield HPO_evaluation
    finally:
        sys.path.remove(autopcr_dir)
        sys.modules.update(saved)


def _stub_benepar(text):
    """Stands in for ee.benepar.process_text2phrases_benepar: (text, (spans, conjunct), index_map).

    ``index_map`` shifts every offset by 100 so the test can see that offsets are MAPPED, and the
    constituent list repeats one span so it can see that duplicates are removed.
    """
    index_map = [i + 100 for i in range(len(text) + 1)]
    spans = [(0, 8, text[0:8], "seizur"), (0, 8, text[0:8], "seizur"), (13, 26, text[13:26], "short statur")]
    conjunct = [(0, 26, "seizures short stature", "seizur short statur")]
    return text, (spans, conjunct), index_map


class TestParserCache:
    def test_cache_builder_is_upstreams_loop(self, tmp_path, monkeypatch, autopcr_modules):
        """What extract_phrases.py writes is what run_gsc_test_ner's inline loop wrote."""
        monkeypatch.setattr(autopcr_modules, "process_text2phrases_benepar", _stub_benepar)
        all_test = ["P:01\nseizures and short stature", "P:02\nseizures and short stature"]

        benepar, conjunct = autopcr_modules.build_benepar_phrase_cache(all_test, str(tmp_path))

        on_disk = json.loads((tmp_path / "phrases_benepar.json").read_text())
        assert on_disk == benepar
        assert json.loads((tmp_path / "phrases_conjunct.json").read_text()) == conjunct
        assert set(benepar) == set(conjunct) == {"P:01", "P:02"}
        assert benepar["P:01"]["phrases"] == [
            {"start": 100, "end": 108, "phrase": "seizures", "phrase_nostopword": "seizur"},
            {"start": 113, "end": 126, "phrase": "short stature", "phrase_nostopword": "short statur"},
        ]
        assert conjunct["P:02"]["phrases"] == [
            {"start": 100, "end": 126, "phrase": "seizures short stature",
             "phrase_nostopword": "seizur short statur"}]
        assert benepar["P:01"]["text_new"] == "seizures and short stature"

    def test_require_parser_cache(self, tmp_path):
        from hpo_extraction.baselines.autopcr_runner import require_parser_cache

        require_parser_cache(str(tmp_path), "neural")          # nothing needed, nothing checked
        with pytest.raises(SystemExit, match=r"DATASET=gsc sbatch slurm/autopcr_parse\.sbatch"):
            require_parser_cache(str(tmp_path), "neural++", dataset="gsc")
        (tmp_path / "phrases_benepar.json").write_text("{}")
        with pytest.raises(SystemExit, match="phrases_conjunct.json"):
            require_parser_cache(str(tmp_path), "neural+")    # upstream opens both, even at neural+
        (tmp_path / "phrases_conjunct.json").write_text("{}")
        require_parser_cache(str(tmp_path), "neural++")

    def test_a_pilots_parser_cache_is_dropped_and_then_reported_missing(self, tmp_path):
        from hpo_extraction.baselines.autopcr_runner import require_parser_cache

        for name in ("phrases_benepar.json", "phrases_conjunct.json"):
            (tmp_path / name).write_text(json.dumps({"P:01": {"phrases": []}}))
        assert sorted(invalidate_stale_phrase_cache(str(tmp_path), {"P:01", "P:02"})) == [
            "phrases_benepar.json", "phrases_conjunct.json"]
        with pytest.raises(SystemExit):
            require_parser_cache(str(tmp_path), "neural++")

    def test_phrase_counts_per_source(self, tmp_path):
        from hpo_extraction.baselines.autopcr_runner import phrase_counts

        (tmp_path / "phrases.json").write_text(json.dumps({"P:01": {"phrases": [{}, {}]}}))
        (tmp_path / "phrases_benepar.json").write_text(json.dumps({"P:01": {"phrases": [{}]}}))
        assert phrase_counts(str(tmp_path)) == {"phrases": 2, "phrases_benepar": 1}


class TestExtractPhrases:
    """third_party/AutoPCR/extract_phrases.py, the part that runs in autopcr_ee_venv."""

    @staticmethod
    def _module():
        sys.path.insert(0, str(_REPO / "third_party" / "AutoPCR"))
        import extract_phrases

        return extract_phrases

    def _stage(self, root, name, text="seizures and short stature"):
        d = root / name / "autopcr_corpus"
        d.mkdir(parents=True)
        stage_corpus({"P:01": text}, ["P:01"], str(d / "corpus_test.tsv"))
        return d

    def _cache(self, d):
        for name in ("phrases_benepar.json", "phrases_conjunct.json"):
            (d / name).write_text(json.dumps({"P:01": {"phrases": [{"start": 0}]}}))

    def test_a_covering_cache_is_reused_and_copied_to_the_other_linker(self, tmp_path):
        ep = self._module()
        a, b = self._stage(tmp_path, "8b"), self._stage(tmp_path, "70b")
        self._cache(a)
        # No parse happens (HPO_evaluation is never imported), so this runs without benepar.
        assert ep.main(["--corpus", str(a / "corpus_test.tsv"), "--copy_to", str(b)]) == 0
        for name in ("phrases_benepar.json", "phrases_conjunct.json"):
            assert (b / name).read_bytes() == (a / name).read_bytes()

    def test_copy_refuses_a_different_corpus(self, tmp_path):
        ep = self._module()
        a = self._stage(tmp_path, "8b")
        b = self._stage(tmp_path, "70b", text="a different report")
        self._cache(a)
        with pytest.raises(SystemExit, match="differs"):
            ep.main(["--corpus", str(a / "corpus_test.tsv"), "--copy_to", str(b)])
        assert not (b / "phrases_benepar.json").exists()

    def test_cache_covers_requires_both_files_on_the_exact_pmids(self, tmp_path):
        ep = self._module()
        d = self._stage(tmp_path, "8b")
        assert not ep.cache_covers(str(d), {"P:01"})
        self._cache(d)
        assert ep.cache_covers(str(d), {"P:01"})
        assert not ep.cache_covers(str(d), {"P:01", "P:02"})


class TestStageOnly:
    def test_stage_only_stages_and_never_loads_the_linker(self, tmp_path, monkeypatch):
        from omegaconf import OmegaConf

        import hpo_extraction.baselines.autopcr_experiment as drv
        import hpo_extraction.data.loading

        monkeypatch.setattr(hpo_extraction.data.loading, "load_cohort",
                            lambda cfg, logger: ({"P:01": "seizures", "P:02": "short stature"}, {}))

        def _boom(*a, **k):
            raise AssertionError("stage_only must not load the linker")

        monkeypatch.setattr(drv, "load_local_linker", _boom)
        monkeypatch.setattr(drv, "check_dictionary", _boom)
        cfg = OmegaConf.create({"output_dir": str(tmp_path), "dataset": "gsc", "ee": "neural++",
                                "tau_1": 0.95, "tau_2": 0.85, "k": 5, "llama_dir": None,
                                "ccr_model": None, "stage_only": True})
        drv.execute(cfg, "baseline_autopcr_8b")
        corpus = tmp_path / "baseline_autopcr_8b" / "gsc" / "autopcr_corpus" / "corpus_test.tsv"
        assert corpus.read_text(encoding="utf-8").startswith("P:01\nseizures")

    def test_a_neural_pp_run_without_the_cache_stops_before_the_linker(self, tmp_path, monkeypatch):
        from omegaconf import OmegaConf

        import hpo_extraction.baselines.autopcr_experiment as drv
        import hpo_extraction.data.loading

        monkeypatch.setattr(hpo_extraction.data.loading, "load_cohort",
                            lambda cfg, logger: ({"P:01": "seizures"}, {}))
        monkeypatch.setattr(drv, "load_local_linker",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("linker loaded")))
        cfg = OmegaConf.create({"output_dir": str(tmp_path), "dataset": "hcy", "ee": "neural++",
                                "tau_1": 0.95, "tau_2": 0.85, "k": 5, "llama_dir": None,
                                "ccr_model": None})
        with pytest.raises(SystemExit, match=r"DATASET=hcy sbatch slurm/autopcr_parse\.sbatch"):
            drv.execute(cfg, "baseline_autopcr_8b")
