"""PhenoJury on new reports: from clinical text to HPO terms with scores.

PhenoJury reads every segment of a report with each of several small language models, the jurors.
Each juror names the phenotypic findings it sees, in free text. A normaliser maps every name to HPO
identifiers, and a term is predicted when at least ``k`` jurors support it within one unit of text
(the report, a window of three segments, or a single segment).

The defaults are the configuration the full pool of eight jurors selected most often by nested
cross-validation on the HCY cohort: the HPO-Guided prompt (``q2_sentence_last``), the PhenoBERT
normaliser applied to the parsed candidate strings, the exact matching rule, the segment as the
voting unit and ``k = 3``. Prompt rendering, candidate parsing, normalisation and voting call the
functions the thesis experiments used (``prompts``, ``normalisers``, ``vote``).

Example::

    from hpo_extraction.phenojury import PhenoJury

    method = PhenoJury.from_config("configs/apps/phenojury.yaml")
    result = method.extract("At 11 months his development was mildly delayed.", "report_1")
    for term in result.terms:
        print(term.hpo_id, term.label, term.score)
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from hpo_extraction.data.report_segments import Segmenter
from hpo_extraction.evaluation.metrics import OntologyView
from hpo_extraction.ontology.hpo_tree import HPOTree, hpo_label
from hpo_extraction.phenojury import vote
from hpo_extraction.phenojury.generation import MODEL_KEYS
from hpo_extraction.phenojury.prompt_diagnostics import _shape_of
from hpo_extraction.phenojury.prompts import get_prompt
from hpo_extraction.results import Evidence, ReportResult, TermScore

logger = logging.getLogger(__name__)

#: Generation budgets of the thesis runs: 512 new tokens, 1536 for the three reasoning models,
#: whose reasoning block is removed before parsing.
REASONING_MODELS = {"deepseek": 1536, "intelligent_internet": 1536, "medpsy": 1536}


#: ``(records, report_ids, prompt_key) -> {report_id: {segment_index: {hpo_id: count}}}``.
#: ``records`` are generation records in the layout of ``llm_extractions_<model>.jsonl``.
Normaliser = Callable[[list[dict], list[str], str], dict]


@dataclass
class PhenoJurySettings:
    """Configuration of one PhenoJury run.

    Attributes:
        prompt: prompt key: ``p0_baseline`` (Free Listing), ``q2_sentence_last`` (HPO-Guided),
            ``q4_span_json`` (Evidence Spans) or ``q7_recall`` (Recall-First).
        jurors: model keys of the jurors, in voting order.
        rule: ``exact`` (a juror supports a term it produced) or ``closure_reduced`` (a juror also
            supports every ancestor of a term it produced, and the output is reduced to its most
            specific terms).
        unit: voting unit, ``segment``, ``window`` (the segment and its two neighbours) or
            ``report``.
        k: minimum number of jurors that must support a term within one unit. Between 1 and the
            number of jurors.
        max_new_tokens: generation budget per segment.
        max_new_tokens_overrides: larger budgets for reasoning models.
        batch_size: segments generated together per call.
    """

    prompt: str = "q2_sentence_last"
    jurors: list[str] = field(default_factory=lambda: list(MODEL_KEYS))
    rule: str = "exact"
    unit: str = "segment"
    k: int = 3
    max_new_tokens: int = 512
    max_new_tokens_overrides: dict = field(default_factory=lambda: dict(REASONING_MODELS))
    batch_size: int = 8

    def __post_init__(self) -> None:
        get_prompt(self.prompt)
        if self.unit not in vote.UNITS:
            raise ValueError(f"unit must be one of {vote.UNITS}")
        if self.rule not in ("exact", "closure_reduced"):
            raise ValueError("rule must be 'exact' or 'closure_reduced'")
        if not 1 <= self.k <= len(self.jurors):
            raise ValueError("k must lie between 1 and the number of jurors")


def dictionary_normaliser(tree: HPOTree, hpo_json: str | None = None) -> Normaliser:
    """The dictionary normaliser of the thesis: exact, then lexically normalised label match.

    Needs no model. Matches against the labels and synonyms of the ontology file, with the settings
    of the thesis runs (``dictionary_mode = lexical``, accepted sources exact and synonym).
    """
    from hpo_extraction.phenojury.normalisers import normalise_cell
    from hpo_extraction.retrieval.ontology_index import load_index
    from hpo_extraction.retrieval.surface_index import build_surface_index

    surface2hpo, _ = build_surface_index(tree)
    index = load_index(hpo_json) if hpo_json else load_index()

    def normalise(records: list[dict], report_ids: list[str], prompt_key: str) -> dict:
        detections, _ = normalise_cell(
            records, report_ids, "dictionary", prompt_key, shape=_shape_of(prompt_key),
            surface2hpo=surface2hpo, ontology_index=index, dictionary_mode="lexical",
            dictionary_accept_sources=("exact", "synonym"))
        return detections

    return normalise


def phenobert_normaliser(phenobert_dir: str, phenobert_python: str | None = None,
                         stanza_dir: str | None = None, work_dir: str | None = None,
                         p1: float = 0.8, p2: float = 0.6, p3: float = 0.9,
                         chunk_size: int = 16) -> Normaliser:
    """The PhenoBERT normaliser of the thesis, applied to the parsed candidate strings.

    PhenoBERT runs as a subprocess in its own Python environment (``third_party/README.md``).
    ``p1``, ``p2`` and ``p3`` are its confidence thresholds (0 to 1), at the published defaults.
    """
    from types import SimpleNamespace

    from hpo_extraction.phenojury.normalisers import normalise_cell, run_phenobert_candidates

    cfg = SimpleNamespace(phenobert_dir=phenobert_dir, phenobert_python=phenobert_python,
                          stanza_dir=stanza_dir, p1=p1, p2=p2, p3=p3,
                          phenobert_chunk_size=chunk_size)

    def normalise(records: list[dict], report_ids: list[str], prompt_key: str) -> dict:
        shape = _shape_of(prompt_key)
        model_key = records[0]["model"] if records else "juror"
        base = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="phenojury_"))
        out_dir, inputs, line_spans = run_phenobert_candidates(
            cfg, {"model_key": model_key}, records, report_ids, shape, base / model_key)
        detections, _ = normalise_cell(
            records, report_ids, "phenobert_candidates", prompt_key, shape=shape,
            phenobert_output_dir=out_dir, phenobert_inputs=inputs,
            phenobert_line_spans=line_spans)
        return detections

    return normalise


class PhenoJury:
    """Extract HPO terms from reports with PhenoJury.

    Args:
        hpo_tree: the ontology (``resources/util/hpo.json``, release 2024-08-13).
        jurors: ``{model_key: local model folder}``. The jurors are loaded one after another,
            each released before the next, as in the thesis runs.
        normaliser: maps generations to HPO identifiers (:func:`phenobert_normaliser` or
            :func:`dictionary_normaliser`).
        settings: prompt, jury, rule, unit and ``k``.
        segmenter: splits reports into segments.
    """

    def __init__(self, hpo_tree: HPOTree, jurors: Mapping[str, str], normaliser: Normaliser,
                 settings: PhenoJurySettings | None = None, segmenter: Segmenter | None = None):
        self.tree = hpo_tree
        self.jurors = dict(jurors)
        self.normaliser = normaliser
        self.settings = settings or PhenoJurySettings(jurors=list(self.jurors))
        missing = [m for m in self.settings.jurors if m not in self.jurors]
        if missing:
            raise ValueError(f"no model given for juror(s) {missing}")
        self.segmenter = segmenter or Segmenter()
        self.view = OntologyView(hpo_tree)

    @classmethod
    def from_config(cls, path: str) -> "PhenoJury":
        """Build the method from a YAML file (see ``configs/apps/phenojury.yaml``)."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        tree = HPOTree(cfg["hpo_json"])
        tree.buildHPOTree()
        settings = PhenoJurySettings(**(cfg.get("settings") or {}))
        juror_dirs = {key: cfg["juror_dirs"][key] for key in settings.jurors}
        norm_cfg = cfg.get("normaliser") or {}
        if norm_cfg.get("name", "phenobert_candidates") == "dictionary":
            normaliser = dictionary_normaliser(tree, cfg["hpo_json"])
        else:
            normaliser = phenobert_normaliser(
                norm_cfg["phenobert_dir"], norm_cfg.get("phenobert_python"),
                norm_cfg.get("stanza_dir"), norm_cfg.get("work_dir"))
        return cls(tree, juror_dirs, normaliser, settings, Segmenter(cfg.get("stanza_dir")))

    def extract(self, text: str, report_id: str = "report") -> ReportResult:
        """Predict the HPO terms of one report.

        Returns:
            A :class:`ReportResult`. Each term's ``score`` is the share of jurors supporting it in
            its best unit (between ``k / J`` and 1), and its evidence lists the segments where
            jurors named it, with the number of jurors per segment.
        """
        return self.extract_many({report_id: text})[0]

    def extract_many(self, texts: dict[str, str]) -> list[ReportResult]:
        """Predict the HPO terms of several reports (``{report_id: text}``), in input order."""
        s = self.settings
        segments = self.segmenter.segment(texts)
        report_ids = list(texts)
        spec = get_prompt(s.prompt)
        per_juror: dict[vote.Juror, dict] = {}
        for model_key in s.jurors:
            records = self._generate(model_key, spec, segments, report_ids)
            detections = self.normaliser(records, report_ids, s.prompt)
            juror = vote.Juror(model=model_key, prompt=s.prompt, normaliser="")
            per_juror[juror] = {rid: {int(sent): set(hpos) for sent, hpos in by_sent.items()}
                                for rid, by_sent in detections.items()}
        jurors = list(per_juror)
        packed = vote.pack(jurors, per_juror, report_ids, s.unit, s.rule, self.view)
        counts = vote.max_counts(packed, report_ids, frozenset(range(len(jurors))))
        predicted = vote.sets_at_k(counts, s.k)
        if s.rule == "closure_reduced":
            predicted = {rid: vote.reduce_specific(t, self.view) for rid, t in predicted.items()}
        results = []
        for rid in report_ids:
            sents = segments.get(rid, [])
            terms = [TermScore(h, hpo_label(self.tree, h), counts[rid][h] / len(jurors),
                               self._evidence(h, rid, per_juror, sents))
                     for h in predicted.get(rid, set())]
            terms.sort(key=lambda t: (-t.score, t.hpo_id))
            results.append(ReportResult(rid, terms, len(sents), len(jurors) * len(sents),
                                        "phenojury", asdict(s)))
        return results

    def _generate(self, model_key: str, spec, segments: dict, report_ids: list[str]) -> list[dict]:
        """One juror's generation records, produced by the extraction loop of the thesis runs."""
        from omegaconf import OmegaConf

        from hpo_extraction.phenojury.generation import _run_extraction

        s = self.settings
        cfg = OmegaConf.create({
            "reuse_extractions": False, "gen_batch_size": s.batch_size,
            "max_new_tokens": s.max_new_tokens,
            "max_new_tokens_overrides": dict(s.max_new_tokens_overrides),
        })
        sent_dict = {rid: segments.get(rid, []) for rid in report_ids}
        with tempfile.TemporaryDirectory(prefix="phenojury_") as run_dir:
            records, _ = _run_extraction(cfg, model_key, self.jurors[model_key], sent_dict,
                                         report_ids, run_dir, logger,
                                         system_prompt=spec.system,
                                         user_template=spec.user_template)
        return records

    def _evidence(self, hpo_id, rid, per_juror, sents) -> list[Evidence]:
        votes: dict[int, int] = {}
        for by_sent in per_juror.values():
            for sent, terms in by_sent.get(rid, {}).items():
                if hpo_id in {self.view.resolve(t) for t in terms}:
                    votes[sent] = votes.get(sent, 0) + 1
        return [Evidence(i, sents[i] if i < len(sents) else "", float(n))
                for i, n in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))]
