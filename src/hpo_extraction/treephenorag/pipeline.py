"""TreePhenoRAG on new reports: from clinical text to HPO terms with scores.

TreePhenoRAG walks the ontology from the organ-system terms below *Phenotypic abnormality*
(HP:0000118) downwards. For every term it visits it

1. retrieves the ``S`` report segments most similar to the term or any of its descendants
   (descendant closure),
2. asks the verifier, Llama-3.1-8B-Instruct, whether each segment confirms the term, and turns the
   answer into a segment score ``sigma = P(Yes) / (P(Yes) + P(No))`` computed from the first-token
   logits,
3. pools the ``S`` segment scores twice: an expansion score decides whether the term's children are
   visited (expansion threshold), and an acceptance score decides whether the term is output
   (acceptance threshold).

The defaults are the configuration the thesis selected by nested cross-validation on the HCY
cohort: term-information index, ``S = 10``, mean pooling (P4) for expansion at a threshold of
5.58e-5, log-sum-exp pooling (LSE) for acceptance at 0.99. Every step calls the functions the
thesis experiments used (``score_store``, ``traversal``, ``pooling``, ``verifier_prompt``).

Example::

    from hpo_extraction.treephenorag import TreePhenoRAG

    method = TreePhenoRAG.from_config("configs/apps/treephenorag.yaml")
    result = method.extract("He has had recurrent seizures since the age of two.", "report_1")
    for term in result.terms:
        print(term.hpo_id, term.label, round(term.score, 3))
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from hpo_extraction.data.report_segments import Segmenter
from hpo_extraction.evaluation.retrieval_analysis import descendants_with_hops
from hpo_extraction.ontology.hpo_tree import HPOTree, hpo_label
from hpo_extraction.results import Evidence, ReportResult, TermScore
from hpo_extraction.retrieval.descendant_closure import top_sentences
from hpo_extraction.treephenorag import pooling
from hpo_extraction.treephenorag.traversal import NodeEval, traverse
from hpo_extraction.treephenorag.verifier_prompt import BaselinePrompter

logger = logging.getLogger(__name__)

#: Online pooling operators by the identifiers the thesis tables use (thesis, Table 4.1). Each one
#: maps the verifier margins of a term's retrieved segments to a score between 0 and 1. They are
#: The functions in :mod:`hpo_extraction.treephenorag.pooling`. A test checks that they give the
#: same values as the vectorised versions in :mod:`hpo_extraction.treephenorag.stored_scores`,
#: which produced the thesis numbers.
POOLINGS: dict[str, Callable[[Sequence[float]], float]] = {
    "P0": lambda m: float(pooling.accept_confidence(m) > 0.5),
    "P1": pooling.accept_confidence,
    "P2": pooling.second_largest_confidence,
    "P3_1": lambda m: pooling.noisy_or_kappa(m, 1.0),
    "P3_2": lambda m: pooling.noisy_or_kappa(m, 2.0),
    "P3_S": lambda m: pooling.noisy_or_kappa(m, float(max(len(m), 1))),
    "P4": pooling.mean_confidence,
    "lse_beta1": lambda m: pooling.lse_beta_prob(m, 1.0),
}


class Encoder(Protocol):
    """Anything with a SentenceTransformer-style ``encode`` (texts to an ``(n, d)`` array)."""

    def encode(self, texts: list[str], **kwargs) -> np.ndarray: ...


class Verifier(Protocol):
    """Anything with :meth:`hpo_extraction.models.llama.LlamaLogitsLLM.generate_batch`.

    ``generate_batch(prompts, system_prompt)`` returns one dict per prompt with at least the key
    ``margin``, the log-odds ``log P(Yes) - log P(No)`` of the first answer token.
    """

    def generate_batch(self, prompts: list[str], system_prompt: str) -> list[dict]: ...


@dataclass
class TreePhenoRAGSettings:
    """Configuration of one TreePhenoRAG run.

    Attributes:
        segments_per_term: ``S``, segments retrieved and verified per visited term (1 to 10 were
            examined in the thesis, 10 is the selected value, 3 keeps 97 % of the F1 score at
            35 % of the verifier calls).
        expansion_pooling: pooling operator for the expansion score, a key of :data:`POOLINGS`.
        expansion_threshold: a term's children are visited when its expansion score is at least
            this value. Range 0 to 1. The default was chosen by conformal risk control at a risk
            tolerance of 0.10 (expected share of annotated terms never visited).
        acceptance_pooling: pooling operator for the acceptance score.
        acceptance_threshold: a visited term is output when its acceptance score is at least this
            value. Range 0 to 1. Scores are not calibrated probabilities.
        retrieval_index: ``"ontology_r3"`` (term information: label, definition and synonyms in
            one string per term) or ``"exemplar"`` (synthetic sentences, needs
            ``synthetic_sentences_dir``).
        synthetic_sentences_dir: folder with one ``HP_xxxxxxx.txt`` file of synthetic sentences
            per term. Used only with ``retrieval_index = "exemplar"``.
        subtree_root: limit the traversal to the subtree below this term (for trials). ``None``
            uses the whole ontology below HP:0000118.
        max_terms: stop after scoring this many terms per report (for trials). ``None`` means no
            limit.
    """

    segments_per_term: int = 10
    expansion_pooling: str = "P4"
    expansion_threshold: float = 5.581644480753024e-05
    acceptance_pooling: str = "lse_beta1"
    acceptance_threshold: float = 0.99
    retrieval_index: str = "ontology_r3"
    synthetic_sentences_dir: str | None = None
    subtree_root: str | None = None
    max_terms: int | None = None

    def __post_init__(self) -> None:
        for name in (self.expansion_pooling, self.acceptance_pooling):
            if name not in POOLINGS:
                raise ValueError(f"unknown pooling operator {name!r}; expected one of "
                                 f"{sorted(POOLINGS)}")
        if self.retrieval_index not in ("ontology_r3", "exemplar"):
            raise ValueError("retrieval_index must be 'ontology_r3' or 'exemplar'")
        if self.retrieval_index == "exemplar" and not self.synthetic_sentences_dir:
            raise ValueError("retrieval_index 'exemplar' needs synthetic_sentences_dir")


class TreePhenoRAG:
    """Extract HPO terms from reports with TreePhenoRAG.

    Args:
        hpo_tree: the ontology (``HPOTree`` over the release the thesis used,
            ``resources/util/hpo.json``).
        encoder: sentence encoder for segments and term texts (all-mpnet-base-v2 in the thesis).
        verifier: the Yes/No verifier (Llama-3.1-8B-Instruct in 8-bit precision in the thesis).
        settings: thresholds, pooling operators and retrieval index.
        segmenter: splits reports into segments. Default: rule-based (see
            :class:`hpo_extraction.data.report_segments.Segmenter`).
    """

    def __init__(self, hpo_tree: HPOTree, encoder: Encoder, verifier: Verifier,
                 settings: TreePhenoRAGSettings | None = None,
                 segmenter: Segmenter | None = None):
        from omegaconf import OmegaConf

        from hpo_extraction.treephenorag.score_store import SYSTEM_PROMPT, _build_universe

        self.tree = hpo_tree
        self.encoder = encoder
        self.verifier = verifier
        self.settings = settings or TreePhenoRAGSettings()
        self.segmenter = segmenter or Segmenter()
        self.system_prompt = SYSTEM_PROMPT
        self.prompter = BaselinePrompter()
        # The retrieval index and the traversal graph are built by the function the thesis runs
        # used, from the same keys their configs set.
        cfg = OmegaConf.create({
            "retrieval_index": self.settings.retrieval_index,
            "context_dir": self.settings.synthetic_sentences_dir,
            "restrict_to_subtree": self.settings.subtree_root,
        })
        (self.scorer, _, self.children_map, self.roots,
         self.subtree) = _build_universe(cfg, hpo_tree, encoder, logger)
        self._desc_cols: dict[str, np.ndarray] = {}

    @classmethod
    def from_config(cls, path: str) -> "TreePhenoRAG":
        """Build the method from a YAML file (see ``configs/apps/treephenorag.yaml``).

        The file names the ontology file, the local model folders and the settings. Models are
        always read from local folders, never downloaded.
        """
        from omegaconf import OmegaConf
        from sentence_transformers import SentenceTransformer

        from hpo_extraction.models.llama import LlamaLogitsLLM, load_llama

        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        tree = HPOTree(cfg["hpo_json"])
        tree.buildHPOTree()
        encoder = SentenceTransformer(cfg["encoder_dir"])
        tokenizer, model = load_llama(cfg["verifier_dir"],
                                      load_in_4bit=bool(cfg.get("load_in_4bit", False)))
        model.eval()
        settings = TreePhenoRAGSettings(**(cfg.get("settings") or {}))
        return cls(tree, encoder, LlamaLogitsLLM(model, tokenizer), settings,
                   Segmenter(cfg.get("stanza_dir")))

    def _columns(self, hpo_id: str) -> np.ndarray:
        if hpo_id not in self._desc_cols:
            self._desc_cols[hpo_id] = self.scorer.columns_for(
                descendants_with_hops(self.tree, hpo_id))
        return self._desc_cols[hpo_id]

    def extract(self, text: str, report_id: str = "report") -> ReportResult:
        """Predict the HPO terms of one report.

        Args:
            text: the report text.
            report_id: identifier written into the result.

        Returns:
            A :class:`ReportResult`. Each term's ``score`` is its acceptance score (0 to 1) and its
            evidence lists the retrieved segments with their verifier probability of *Yes*.
        """
        return self.extract_many({report_id: text})[0]

    def extract_many(self, texts: dict[str, str]) -> list[ReportResult]:
        """Predict the HPO terms of several reports (``{report_id: text}``), in input order."""
        segments = self.segmenter.segment(texts)
        return [self._extract_segments(rid, segments.get(rid, [])) for rid in texts]

    def _extract_segments(self, report_id: str, sents: list[str]) -> ReportResult:
        s = self.settings
        if not sents:
            return ReportResult(report_id, [], 0, 0, "treephenorag", asdict(s))
        self.scorer.score_patient(np.asarray(self.encoder.encode(sents)))
        expand = POOLINGS[s.expansion_pooling]
        accept = POOLINGS[s.acceptance_pooling]
        retrieved: dict[str, tuple[list[int], list[float]]] = {}

        # The per-term evaluation of the thesis driver (score_store.execute, retrieval_ctx
        # "union_only"): retrieve over the term and its descendants, verify each segment, pool.
        def evaluate(hpo_id: str) -> NodeEval:
            sims = self.scorer.union_over_columns(self._columns(hpo_id))
            info = top_sentences(sims, sents, top_n=s.segments_per_term)
            details = self.verifier.generate_batch(
                self.prompter.build_prompts(info["top_sents"], hpo_id), self.system_prompt)
            margins = [d["margin"] for d in details]
            retrieved[hpo_id] = (info["top_indices"], margins)
            return NodeEval(expand(margins), accept(margins), len(details), [])

        trace = traverse(self.children_map, self.roots, evaluate, s.expansion_threshold,
                         s.acceptance_threshold, max_nodes=s.max_terms)
        terms = []
        for hpo_id in trace.accepted:
            indices, margins = retrieved[hpo_id]
            sigma = 1.0 / (1.0 + np.exp(-np.asarray(margins, dtype=float)))
            evidence = sorted((Evidence(int(i), sents[i], float(p))
                               for i, p in zip(indices, sigma)), key=lambda e: -e.score)
            terms.append(TermScore(hpo_id, hpo_label(self.tree, hpo_id),
                                   float(trace.visits[hpo_id].accept_score), evidence))
        terms.sort(key=lambda t: (-t.score, t.hpo_id))
        return ReportResult(report_id, terms, len(sents), trace.n_slm_calls, "treephenorag",
                            asdict(s))
