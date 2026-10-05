"""The appendix's side-by-side example rows (``apps/compare_ui/export_examples.py``).

Built on the compare_ui GSC+ fixture, where the localisation cases all occur on GSC001:
PhenoBERT and AutoPCR place Seizure by offset, RAG-HPO and TreePhenoRAG find it with no location,
PhenoJury alone finds the tremor (by juror sentence), and AutoPCR's spurious term sits in segment 1.
"""

from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (_REPO,):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from apps.compare_ui import bundles, fixture, roster  # noqa: E402
from apps.compare_ui import export_examples as E  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def gsc(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("qualitative"))
    written = fixture.write_gsc(root)
    _ctx, built = fixture.build_gsc(written)
    built.pop("_skipped")
    bundle_dir = os.path.join(root, "bundles", "gsc")
    for bundle in built.values():
        bundles.write(bundle_dir, bundle)
    return {"built": built, "bundle_dir": bundle_dir, "codes": fixture._codes()}


def _by_code(rows):
    return {r["hpo_id"]: r for r in rows}


class TestRowSet:
    def test_every_annotated_term_of_the_segment_is_a_row(self, gsc):
        bundle = gsc["built"]["GSC001"]
        rows = _by_code(E.example_rows(bundle, 0))
        gold_here = {g["hpo_id"] for g in bundle["gold"] if g["segment_idx"] == 0}
        assert gold_here and gold_here <= set(rows)
        assert all(rows[c]["row_kind"] == "gold" for c in gold_here)

    def test_an_unplaced_annotated_term_is_not_drawn_on_any_segment(self, gsc):
        bundle = gsc["built"]["GSC001"]
        unplaced = {g["hpo_id"] for g in bundle["gold"] if g["segment_idx"] is None}
        assert unplaced
        for seg in range(len(bundle["segments"])):
            assert not unplaced & set(_by_code(E.example_rows(bundle, seg)))

    def test_a_false_positive_appears_only_where_the_method_placed_it(self, gsc):
        bundle, codes = gsc["built"]["GSC001"], gsc["codes"]
        assert codes["M"] not in _by_code(E.example_rows(bundle, 0))
        row = _by_code(E.example_rows(bundle, 1))[codes["M"]]
        assert row["row_kind"] == "predicted"
        assert row["autopcr_70b"] == "fp"
        assert all(row[k] == "" for k in roster.ORDER if k != "autopcr_70b")

    def test_gold_rows_come_in_reading_order(self, gsc):
        bundle, codes = gsc["built"]["GSC001"], gsc["codes"]
        gold = [r["hpo_id"] for r in E.example_rows(bundle, 0) if r["row_kind"] == "gold"]
        assert gold == [codes["C"], codes["D"]]      # "seizures and a fine tremor"


class TestCells:
    def test_found_here_found_unplaced_and_missed(self, gsc):
        bundle, codes = gsc["built"]["GSC001"], gsc["codes"]
        seizure = _by_code(E.example_rows(bundle, 0))[codes["C"]]
        assert seizure["phenobert"] == "tp"
        assert seizure["autopcr_70b"] == "tp"
        assert seizure["raghpo_70b"] == "tp_unplaced"
        assert seizure["treephenorag"] == "tp_unplaced"
        tremor = _by_code(E.example_rows(bundle, 0))[codes["D"]]
        assert tremor["phenojury"] == "tp"
        assert tremor["phenobert"] == "fn"

    def test_found_in_another_segment_is_not_found_here(self, gsc):
        bundle, codes = gsc["built"]["GSC001"], gsc["codes"]
        moved = {k: dict(v) for k, v in bundle["methods"].items()}
        moved["phenobert"]["reasons"] = dict(moved["phenobert"]["reasons"])
        moved["phenobert"]["reasons"][codes["C"]] = {"segment_idx": 2}
        row = _by_code(E.example_rows(dict(bundle, methods=moved), 0))[codes["C"]]
        assert row["phenobert"] == "tp_elsewhere"

    def test_a_juror_sentence_localises_a_jury_term(self, gsc):
        bundle, codes = gsc["built"]["GSC001"], gsc["codes"]
        moved = {k: dict(v) for k, v in bundle["methods"].items()}
        reasons = dict(moved["phenojury"]["reasons"])
        reasons[codes["D"]] = dict(reasons[codes["D"]], by_sentence={"1": ["apertus", "llama"]})
        moved["phenojury"]["reasons"] = reasons
        rows = _by_code(E.example_rows(dict(bundle, methods=moved), 0))
        assert rows[codes["D"]]["phenojury"] == "tp_elsewhere"

    def test_a_column_that_could_not_be_built_says_so(self, gsc):
        bundle, codes = gsc["built"]["GSC001"], gsc["codes"]
        moved = {k: dict(v) for k, v in bundle["methods"].items()}
        moved["raghpo_70b"]["status"] = "missing"
        row = _by_code(E.example_rows(dict(bundle, methods=moved), 0))[codes["C"]]
        assert row["raghpo_70b"] == "na"

    def test_every_cell_is_in_the_vocabulary(self, gsc):
        for bundle in gsc["built"].values():
            for seg in range(len(bundle["segments"])):
                for row in E.example_rows(bundle, seg):
                    assert all(row[k] in E.CELLS for k in roster.ORDER)


class TestSpec:
    def _spec(self, **kwargs):
        row = {"block": "gsc", "cohort": "gsc", "doc_id": "GSC001", "segment_idx": "",
               "anchor": "", "line": 2}
        row.update(kwargs)
        return row

    def test_a_location_resolves_to_its_segment(self, gsc):
        bundle = gsc["built"]["GSC001"]
        assert E.resolve_segment(bundle, self._spec(anchor="spasticity of the lower")) == 1

    @pytest.mark.parametrize("kwargs, message", [
        ({"anchor": "not in the abstract"}, "no segment"),
        ({"anchor": "e"}, "lengthen"),
        ({"anchor": "spasticity", "segment_idx": "0"}, "anchor is in segment 1"),
        ({"segment_idx": "9"}, "has 3 segments"),
        ({}, "neither"),
    ])
    def test_a_row_that_names_no_single_segment_aborts(self, gsc, kwargs, message):
        with pytest.raises(E.SpecError, match=message):
            E.resolve_segment(gsc["built"]["GSC001"], self._spec(**kwargs))

    def test_a_missing_bundle_names_the_build_command(self, gsc):
        with pytest.raises(E.SpecError, match="build.py --cohort gsc --reports GSC999"):
            E.export_cohort([self._spec(doc_id="GSC999", segment_idx="0")],
                            gsc["bundle_dir"], "gsc")

    def test_the_export_round_trips_through_csv(self, gsc, tmp_path):
        rows, _prov, summary = E.export_cohort(
            [self._spec(segment_idx="0"), self._spec(doc_id="GSC002", anchor="recurrent")],
            gsc["bundle_dir"], "gsc")
        assert [s[0] for s in summary] == [1, 2]
        path = E.export_path(str(tmp_path), "gsc")
        E.write_csv(path, rows)
        import csv
        with open(path, encoding="utf-8") as handle:
            back = list(csv.DictReader(handle))
        assert len(back) == len(rows)
        assert back[0]["sentence"] == gsc["built"]["GSC001"]["segments"][0]["text"]



# ── the LaTeX generator (experiments/figures/tables_ch6.py:t8) ────────────────────

@pytest.fixture(scope="module")
def tables_ch6():
    sys.path.insert(0, os.path.join(_REPO, "experiments", "figures"))
    import tables_ch6 as T
    return T


def _fake_export(path, n_examples, sentence_len=240, n_terms=4):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=E.FIELDS)
        writer.writeheader()
        for n in range(1, n_examples + 1):
            for t in range(n_terms):
                writer.writerow({
                    "example_no": n, "block": "hcy_phenojury" if n < 4 else "hcy_treephenorag",
                    "cohort": "hcy", "doc_id": "HCY%03d" % n, "segment_idx": 1, "n_segments": 9,
                    "sentence": "x" * sentence_len, "hpo_id": "HP:%07d" % t, "label": "Term %d" % t,
                    "row_kind": "gold" if t == 0 else "predicted", "trigger": "t" if t == 0 else "",
                    **{k: ("tp" if t == 0 else "fp") for k in roster.ORDER}})


class TestGenerator:
    def test_a_long_table_continues_in_a_second_float_with_one_label(self, tables_ch6, tmp_path):
        results = tmp_path / "output" / "comparison"
        results.mkdir(parents=True)
        _fake_export(str(tmp_path / "output" / "compare_ui_bundles" / "qualitative"
                         / "examples_hcy.csv"), n_examples=10)
        tables_ch6.t8_qualitative_examples(str(results), str(tmp_path))
        tex = (tmp_path / "tab_ch6_qualitative_hcy.tex").read_text(encoding="utf-8")
        assert tex.count(r"\begin{table}") >= 2
        assert tex.count(r"\label{tab:app-qualitative-hcy}") == 1
        assert tex.count("(continued)") == tex.count(r"\begin{table}") - 1
        assert tex.count("GENERATED FILE") == 1
        assert all(r"\textbf{(%d)}" % n in tex for n in range(1, 11))
        # Both blocks are headed.
        assert "PhenoJury's errors" in tex and "TreePhenoRAG's errors" in tex

    def test_a_missing_export_writes_the_placeholder(self, tables_ch6, tmp_path):
        results = tmp_path / "output" / "comparison"
        results.mkdir(parents=True)
        tables_ch6.t8_qualitative_examples(str(results), str(tmp_path))
        for cohort in ("hcy", "gsc"):
            tex = (tmp_path / f"tab_ch6_qualitative_{cohort}.tex").read_text(encoding="utf-8")
            assert "not yet computed" in tex

    def test_an_unmappable_character_aborts_naming_only_its_code_point(self, tables_ch6):
        with pytest.raises(SystemExit) as info:
            tables_ch6._tex_text("secret text ☃", "SYN001 segment 3")
        assert "U+2603" in str(info.value) and "SYN001" in str(info.value)
        assert "secret" not in str(info.value)

    def test_report_text_is_escaped(self, tables_ch6):
        out = tables_ch6._tex_text("Hcy 185 µmol/L (n= 0-12) & 50% ≥ 5_x", "d")
        assert r"\textmu{}" in out and r"\&" in out and r"\%" in out and r"$\geq$" in out
        assert r"\_" in out


class TestSkipUnlocated:
    def test_unlocated_rows_are_skipped_only_when_asked(self, gsc):
        spec = [{"block": "gsc", "cohort": "gsc", "doc_id": "GSC001", "segment_idx": "",
                 "anchor": "", "line": 2},
                {"block": "gsc", "cohort": "gsc", "doc_id": "GSC002", "segment_idx": "0",
                 "anchor": "", "line": 3}]
        with pytest.raises(E.SpecError, match="neither"):
            E.export_cohort(spec, gsc["bundle_dir"], "gsc")
        _rows, prov, summary = E.export_cohort(spec, gsc["bundle_dir"], "gsc",
                                               skip_unlocated=True)
        assert [s[1] for s in summary] == ["GSC002"]
        assert prov["skipped_lines"] == [2]

    def test_a_wrong_segment_still_fails_when_skipping(self, gsc):
        spec = [{"block": "gsc", "cohort": "gsc", "doc_id": "GSC002", "segment_idx": "9",
                 "anchor": "", "line": 2}]
        with pytest.raises(E.SpecError, match="segments"):
            E.export_cohort(spec, gsc["bundle_dir"], "gsc", skip_unlocated=True)


class TestAnnotatedElsewhere:
    """A term the ground truth holds for the document but not for this segment, named here by a method."""

    def _with(self, bundle, gold_seg, located_at):
        codes = fixture._codes()
        gold = [dict(g) for g in bundle["gold"]]
        for g in gold:
            if g["hpo_id"] == codes["D"]:
                g["segment_idx"], g["span"] = gold_seg, None
        methods = {k: dict(v) for k, v in bundle["methods"].items()}
        reasons = dict(methods["phenojury"]["reasons"])
        reasons[codes["D"]] = dict(reasons[codes["D"]],
                                   by_sentence={str(located_at): ["x", "y"]})   # The fold's k=2
        methods["phenojury"]["reasons"] = reasons
        return dict(bundle, gold=gold, methods=methods), codes["D"]

    def test_located_on_another_segment_is_found_here_and_flagged(self, gsc):
        bundle, code = self._with(gsc["built"]["GSC001"], gold_seg=0, located_at=1)
        row = _by_code(E.example_rows(bundle, 1))[code]
        assert row["row_kind"] == "gold_elsewhere"
        assert row["phenojury"] == "tp"
        assert row["phenobert"] == ""                   # its miss belongs to segment 0

    def test_unplaced_in_the_gold_is_found_here(self, gsc):
        bundle, code = self._with(gsc["built"]["GSC001"], gold_seg=None, located_at=1)
        row = _by_code(E.example_rows(bundle, 1))[code]
        assert row["row_kind"] == "gold_unplaced"
        assert row["phenojury"] == "tp"


class TestJuryVote:
    """PhenoJury's term is placed only where its jurors reach the vote -- one juror is noise."""

    def _jury(self, bundle, by_code, predicted, known=False, k=None):
        methods = {key: dict(v) for key, v in bundle["methods"].items()}
        methods["phenojury"] = dict(
            methods["phenojury"], predicted=sorted(predicted),
            reasons={c: {"by_sentence": bs, "config_known": known, "k": k, "unit": "segment"}
                     for c, bs in by_code.items()})
        return dict(bundle, methods=methods)

    def test_a_lone_juror_is_not_the_methods_prediction(self, gsc):
        codes = gsc["codes"]
        bundle = self._jury(gsc["built"]["GSC001"],
                            {codes["M"]: {"0": ["a"], "1": ["a", "b", "c"]}}, {codes["M"]},
                            known=True, k=3)
        assert codes["M"] not in _by_code(E.example_rows(bundle, 0))       # 1 juror: outvoted
        assert _by_code(E.example_rows(bundle, 1))[codes["M"]]["phenojury"] == "fp"

    def test_the_threshold_is_inferred_from_the_prediction_set(self, gsc):
        codes = gsc["codes"]
        bundle = self._jury(gsc["built"]["GSC001"],
                            {codes["M"]: {"1": ["a", "b", "c"]},          # predicted, peak 3
                             codes["F"]: {"1": ["a", "b"]}},              # not predicted, peak 2
                            {codes["M"]})
        assert E.jury_threshold(bundle) == {"t": 3, "source": "inferred from the prediction set",
                                            "k_lo": 3, "k_hi": 3}

    def test_a_prediction_set_no_threshold_reproduces_places_nothing(self, gsc):
        codes = gsc["codes"]
        bundle = self._jury(gsc["built"]["GSC001"],
                            {codes["M"]: {"1": ["a"]},                    # predicted, peak 1
                             codes["F"]: {"1": ["a", "b"]}},              # not predicted, peak 2
                            {codes["M"]})
        jury = E.jury_threshold(bundle)
        assert jury["t"] is None and jury["source"].startswith("inconsistent")
        row = _by_code(E.example_rows(bundle, 1))[codes["M"]]        # AutoPCR's row, not ours
        assert row["phenojury"] == "" and row["autopcr_70b"] == "fp"


class TestMainResultK:
    def test_the_largest_k_the_folds_chose_is_read(self, tmp_path):
        path = tmp_path / "s7_headline.csv"
        path.write_text(
            'config,selection_modes\n'
            'full_pool,"[[[""exact"", ""window"", 3], 32], [[""exact"", ""segment"", 2], 15]]"\n'
            'fixed:core3,"[[[""exact"", ""report"", 5], 30]]"\n', encoding="utf-8")
        assert E.main_result_k(str(path)) == 3
        assert E.main_result_k(str(tmp_path / "absent.csv")) is None

    def test_it_settles_placement_when_the_fold_is_unknown(self, gsc):
        codes = gsc["codes"]
        bundle = TestJuryVote()._jury(gsc["built"]["GSC001"],
                                      {codes["M"]: {"0": ["a", "b"], "1": ["a", "b", "c"]}},
                                      {codes["M"]})
        jury = E.jury_threshold(bundle, cohort_k=3)
        assert jury["t"] == 3 and jury["source"].startswith("largest k")
        assert codes["M"] not in {r["hpo_id"] for r in E.example_rows(bundle, 0, jury)
                                  if r["phenojury"]}
        assert _by_code(E.example_rows(bundle, 1, jury))[codes["M"]]["phenojury"] == "fp"
