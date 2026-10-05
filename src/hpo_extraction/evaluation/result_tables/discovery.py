"""Which earlier runs exist on disk, and which skeleton metrics each one can support.

Everything downstream is driven from here. The results chapter must be able to distinguish three
situations that all look like "no number in the table":

1. **The run did not happen.** No output directory, or an empty one.
2. **The run happened but is unusable.** Shard files were never merged, so any metric computed
   from them would silently describe a fraction of the cohort.
3. **The metric does not apply to that method.** RAG-HPO retrieves phrase→HPO candidates rather
   than report segments, so eq. (2) at the segment unit is not a missing measurement, it is a
   category error, and printing a blank there without saying so invites the reader to read it as
   a failure.

:func:`discover` answers (1) and (2) by looking at the filesystem; :data:`METRIC_SUPPORT` answers
(3) from the method's design. ``availability.md`` prints both, and every placeholder anywhere in
the emitted results traces back to a row of one of these two tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Shard files are written as ``{variant}_s{i}of{N}_{kind}.jsonl`` by the SLURM array and collapsed
# by ``src/hpo_extraction/evaluation/merge_shards.py``. The lookahead is essential and copied from that
# module's own ``SHARD_RE``: consuming the trailing _ character would derive
# ``flat_topmpredictions.jsonl`` as the merged name, which never exists, so every merged run
# would be misreported as unmerged.
_SHARD_RE = re.compile(r"_s\d+of\d+(?=_)")

#: Cohorts that have their own run directories on disk.
BASE_COHORTS: tuple[str, ...] = ("hcy", "gsc")

#: Cohorts scored from another cohort's artifacts, ``derived -> the directory to read``.
#:
#: The RAG-HPO paper evaluates on 114 of the 228 GSC+ documents, against its own re-annotation
#: (``resources/data/GSC_RAGHPO/PROVENANCE.md``). Both differences are gold-side, and every the earlier runs
#: driver writes one summary line per report, so restricting report ids at scoring time gives
#: bit-identical numbers to re-running inference on the subset. There is no separate run to find.
DERIVED_COHORTS: dict[str, str] = {
    "gsc_raghpo": "gsc",      # their documents, original GSC+ ground truth, the document-selection effect
    "gsc_raghpo_ann": "gsc",  # their documents, their ground truth, comparable to their Table 5
    # AutoPCR's evaluation split of the same corpus: 206 of the 228 documents, corpus ground truth
    # unchanged. It exists so Tao et al.'s published *document-level* F1 has a frame it can be
    # read against. Their main is mention-level and still cannot be.
    "gsc_2024_eval_206": "gsc",
    # The same HCY predictions re-scored against the curation UI's repaired ground truth
    # (``hcy_ground_truth_curated.csv``). The original HCY ground truth is known to be incomplete, the
    # metabolic branch was never annotated, so this cohort exists to price that: every method,
    # both ground-truth sets, one table. Opt-in via `cohorts` + `hcy_curated_gt_path`. Without the export
    # on disk it is simply not requested.
    "hcy_curated": "hcy",
}

COHORTS: tuple[str, ...] = BASE_COHORTS + tuple(DERIVED_COHORTS)


def artifact_cohort(cohort: str) -> str:
    """The cohort whose run directory *cohort* is scored from, itself, unless derived."""
    return DERIVED_COHORTS.get(cohort, cohort)


@dataclass(frozen=True)
class MethodSpec:
    """One row of the earlier comparison, a method, not a (method, cohort) cell.

    ``variant`` is the filename prefix the driver writes (``tree_experiment`` derives it from
    ``build_scores``, ``flat_topm_experiment`` hardcodes it). ``kind`` selects the loader family;
    ``sweep`` names the subdirectory pattern holding configurations, or ``None`` when the run
    writes its files directly into the cohort directory.
    """

    key: str
    exp_id: str
    variant: str
    kind: str                      # "tree" | "raghpo" | "flat_topm" | "ensemble" | "phenobert"
                                   #   | "rerank" | "typed_span" | "constrained" | "autopcr"
                                   # Every value needs a branch in `_resolve_files` AND a key in
                                   # every METRIC_SUPPORT row, a new kind that has one but not
                                   # The other fails at discovery, before any metric is computed.
    label: str                     # how the method is named in the thesis tables
    sweep: str | None = None       # glob for operating-point subdirs, e.g. "tau_*"
    sweep_axis: str | None = None  # what the configuration varies, for table headers
    #: The driver's ``mlflow_experiment_name``, where its cost metrics were logged. Declared here
    #:, not left to config so the §Scalability columns populate without setup, several
    #: methods share one experiment, and the run within it is identified by name (below).
    mlflow_experiment: str = ""
    #: Extra path segment between the cohort directory and the configurations. Empty for every
    #: earlier method, which writes ``{exp_id}/{cohort}/``. earlier sweeps the extraction prompt, so
    #: each of its conditions lives in ``{exp_id}/{cohort}/{prompt_key}/`` and is a separate table row
    #: sharing one run directory, hence a field, not a new ``exp_id`` per prompt.
    #:
    #: Declared last on purpose: every existing row passes ``mlflow_experiment`` positionally, so
    #: inserting a field ahead of it would silently rebind those arguments.
    subdir: str = ""

    @property
    def run_name_prefix(self) -> str:
        """How this method's MLflow runs are named, minus the cohort.

        Every earlier driver opens its run as ``{exp_id}_{dataset}`` with an optional shard or
        model suffix (``tree_experiment.py:253`` and its three siblings), so
        ``f"{prefix}_{cohort}"`` selects this (method, cohort) cell's runs, including
        every shard of an array job, which must be aggregated, not picked from.

        earlier appends the prompt, because two prompt rows share one ``exp_id``: without it both
        would match the same runs and be attributed the same GPU cost. ``prompt_screen`` names its
        runs ``{exp_id}_{prompt_key}_{dataset}_…`` to match.
        """
        return f"{self.exp_id}_{self.subdir}" if self.subdir else self.exp_id


_TREE_MLFLOW = "exp13_treephenorag"
_COMPARISON_MLFLOW = "exp13_comparison"
_EXP17_MLFLOW = "exp17_typed_span"

#: The prompts the generation run of the other prompts was submitted with, one table row each. This is a **manual mirror** of what
#: was actually run, a cell whose subdir has no run directory is reported as `missing` with the
#: path it looked for, so a mismatch is visible, not silent, but it is still a missing row
#: in the thesis table until somebody edits this tuple.
#:
#: The three q-prompts, not an earlier exploratory run's p-family. An earlier exploratory run's gate closed (best margin −0.0122, no
#: prompt beat the control) and an earlier exploratory run's closed too on micro-F1 (+0.0314 against a 0.05
#: threshold), but an earlier exploratory run also found the gate reads the wrong metric on a 20-report cohort: on
#: **macro**-F1 the same screen puts q4_span_json +0.117 over p0_baseline, more than twice the
#: threshold (findings/an earlier exploratory run.md §3, D2). These three were promoted by hand
#: on that reading, and they span the trade-off the screen exposed, not clustering at one
#: end of it:
#:   q2_sentence_last  the micro-F1 winner (0.6344), instruction placed after the sentence
#:   q4_span_json      the macro-F1 and label-exactness winner (0.655 / 0.392), structured output
#:   q7_recall         the highest recall measured anywhere in the screen (0.612 at k4, 0.806 at k1)
EXP14_PROMPT_ROWS: tuple[str, ...] = ("q2_sentence_last", "q4_span_json", "q7_recall")

METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("tree_gate_lr", "exp13_00_tree_gate_lr", "tree_gate_lr", "tree",
               "TreePhenoRAG (LR gate)", "tau_*", "tau_prune", _TREE_MLFLOW),
    MethodSpec("tree_noisyor", "exp13_01_tree_noisyor", "tree_noisyor", "tree",
               "TreePhenoRAG (noisyOR)", "tau_*", "tau_prune", _TREE_MLFLOW),
    MethodSpec("tree_no_prune", "exp13_02_tree_no_prune", "tree_no_prune", "tree",
               "TreePhenoRAG (single threshold)", "tau_*", "tau_prune", _TREE_MLFLOW),
    MethodSpec("tree_lse_beta1", "exp13_09_tree_lse_beta1", "tree_lse_beta1", "tree",
               "TreePhenoRAG (lse_beta1)", "tau_*", "tau_prune", _TREE_MLFLOW),
    MethodSpec("raghpo_8b", "baseline_raghpo_8b", "rag_hpo", "raghpo",
               "RAG-HPO (LLaMA-8B)", None, None, _COMPARISON_MLFLOW),
    MethodSpec("raghpo_70b", "baseline_raghpo_70b", "rag_hpo", "raghpo",
               "RAG-HPO (LLaMA-70B)", None, None, _COMPARISON_MLFLOW),
    # The faithful replication: upstream's published revision, prompts and retrieval stack, on
    # LLaMA-3-70B, the rows comparable to Garcia et al.'s own table. Two of them, because the
    # paper leaves "LLaMA-3-70B" ambiguous: upstream's code names Groq's tool-use finetune, their
    # workbook's GSC columns name the plain Meta instruct model. Same pipeline, same vector DB,
    # same decoding, only the model differs, so the pair also measures what that is worth.
    MethodSpec("raghpo_paper_70b", "raghpo_reproduction_published_code", "rag_hpo", "raghpo",
               "RAG-HPO as published (Groq tool-use 70B)", None, None, _COMPARISON_MLFLOW),
    MethodSpec("raghpo_paper_70b_meta", "raghpo_reproduction_llama3_70b", "rag_hpo", "raghpo",
               "RAG-HPO as published (Meta LLaMA-3-70B)", None, None, _COMPARISON_MLFLOW),
    # The same pipeline on the extraction prompt the *paper* prints (Additional file 2, Fig. S1),
    # which is not the one upstream's repo ships at 25c1ea7. Only `prompts_file` differs from
    # The RAG-HPO reproduction with Llama-3 70B, so this row minus that one measures what the published-prompt disagreement is worth.
    MethodSpec("raghpo_figs1", "raghpo_reproduction_published_prompt", "rag_hpo", "raghpo",
               "RAG-HPO on the paper's Fig. S1 prompt", None, None, _COMPARISON_MLFLOW),
    MethodSpec("flat_topm", "exp13_05_topm_retrieval_slm", "flat_topm", "flat_topm",
               "Flat top-M + SLM", None, None, _COMPARISON_MLFLOW),
    MethodSpec("slm_ensemble", "phenojury_generation_free_listing", "slm_ensemble", "ensemble",
               "SLM ensemble (8 models)", "vote_k*", "vote_k", _COMPARISON_MLFLOW),
    MethodSpec("phenobert", "baseline_phenobert", "phenobert", "phenobert",
               "PhenoBERT", None, None, _COMPARISON_MLFLOW),
    # baseline_autopcr_8b/baseline_autopcr_70b: AutoPCR, the published method that puts an LLM at the *linking* decision
    #, not at extraction, the family this thesis argues for, and the one the comparison
    # otherwise has no external member of. The pair differs only in the linker model, so
    # 13_20 minus 13_19 prices what 9x the linker is worth with retrieval held fixed.
    #
    # `kind="autopcr"`, not reusing "raghpo", which it otherwise resembles: two of raghpo's
    # eleven METRIC_SUPPORT answers are false here. AutoPCR *does* persist a per-term score (so the
    # calibration reason has to say why that score is still not a confidence) and its driver *does*
    # count LLM invocations. Claiming raghpo's reasons would print two statements that are not true
    # of this method.
    MethodSpec("autopcr_8b", "baseline_autopcr_8b", "autopcr", "autopcr",
               "AutoPCR (LLaMA-8B linker)", None, None, _COMPARISON_MLFLOW),
    MethodSpec("autopcr_70b", "baseline_autopcr_70b", "autopcr", "autopcr",
               "AutoPCR (LLaMA-70B linker)", None, None, _COMPARISON_MLFLOW),
    # An earlier exploratory run/an earlier exploratory run: a learned selector over candidates the other methods already produced.
    # They re-score existing artifacts, so their cells appear without any new inference, and
    # their configuration is a probability, not a vote count or a tau.
    MethodSpec("ensemble_rerank", "exp13_13_ensemble_rerank", "rerank", "rerank",
               "SLM ensemble + learned rerank", "p_*", "p_accept", _COMPARISON_MLFLOW),
    MethodSpec("pool_rerank", "exp13_14_pool_rerank", "pool_rerank", "rerank",
               "Cross-method pool + learned rerank", "p_*", "p_accept", _COMPARISON_MLFLOW),
    # An earlier exploratory run: the same pool widened with all four tree variants, each at its own best operating
    # point. A separate row, not a replacement for an earlier exploratory run's, because the two differ only
    # in pool membership and the pair is the measurement, see an earlier exploratory run/experiment.md.
    MethodSpec("pool_rerank_tree", "exp13_16_pool_rerank_tree", "pool_rerank_tree", "rerank",
               "Cross-method + tree pool + learned rerank", "p_*", "p_accept",
               _COMPARISON_MLFLOW),
    # The earlier runs: the typed-span pipeline. `kind="typed_span"` because it is the first method here that
    # emits mention spans, assertion status and an audit trail, not a bare set of codes - so
    # it supports metrics none of the others can (mention-level, polarity) and lacks some they have
    # (no retrieval ranking, no traversal). Each sub-experiment is its own row: they differ by one
    # stage each, and collapsing them would hide the comparison the group exists to make.
    *(
        MethodSpec(key, exp_id, variant, "typed_span", label, None, None, _EXP17_MLFLOW)
        for key, exp_id, variant, label in (
            ("typed_span_detect", "exp17_00_span_detection", "exp17_00",
             "Typed-span: detection"),
            ("typed_span_type", "exp17_01_span_typing", "exp17_01",
             "Typed-span: + span typing"),
            ("typed_span_retrieve", "exp17_02_candidate_retrieval", "exp17_02",
             "Typed-span: + deterministic candidates"),
            ("typed_span_select", "exp17_03_constrained_selection", "exp17_03",
             "Typed-span: + constrained selection"),
            ("typed_span_adjudicate", "exp17_04_numeric_adjudication", "exp17_04",
             "Typed-span: + numeric adjudication"),
            ("typed_span_validate", "exp17_05_record_validation", "exp17_05",
             "Typed-span: + record validation"),
            ("typed_span_escalate", "exp17_06_escalation", "exp17_06",
             "Typed-span: + escalation"),
        )
    ),
    # The earlier runs: constrained decoding. The first method here whose *output space* is the ontology
    #, not free text, the model scores HPO label strings retrieved for each sentence and
    # never emits an unlinkable string, so there is no PhenoBERT stage to lose recall in. The pair
    # differs by one flag (`show_menu`), which prices what showing the model its candidates buys.
    MethodSpec("constrained_decode", "exp18_00_constrained_decode", "constrained", "constrained",
               "Constrained decoding (retrieved top-M)", "thr*", "score_threshold x vote_k",
               _COMPARISON_MLFLOW),
    MethodSpec("constrained_menu", "exp18_01_constrained_menu", "constrained", "constrained",
               "Constrained decoding + candidate menu", "thr*", "score_threshold x vote_k",
               _COMPARISON_MLFLOW),
    # The earlier runs: the Free Listing generation run's own generations, grounded by the ontology instead of by PhenoBERT. The
    # generations are byte-identical to the Free Listing generation run row above, so the difference between the two
    # rows is the linker and nothing else, which is why it reuses `kind="ensemble"`, not
    # declaring a kind of its own: it writes the three files that kind resolves, and every
    # METRIC_SUPPORT answer for the ensemble ("no retrieval stage", "no ontology traversal") is
    # still true of it.
    MethodSpec("agent_lookup", "exp19_00_lookup_only", "slm_ensemble", "ensemble",
               "Ontology-linked ensemble (lookup only)", "vote_k*", "vote_k", _COMPARISON_MLFLOW),
    # The earlier runs: the same ensemble with a restating extraction prompt. Two rows over one run
    # directory, separated by `subdir`, see EXP14_PROMPT_ROWS above.
    *(
        MethodSpec(f"slm_ensemble_{p}", "phenojury_generation_other_prompts", "slm_ensemble", "ensemble",
                   f"SLM ensemble ({p})", "vote_k*", "vote_k", _COMPARISON_MLFLOW, subdir=p)
        for p in EXP14_PROMPT_ROWS
    ),
)

METHODS_BY_KEY: dict[str, MethodSpec] = {m.key: m for m in METHODS}


# ── What each method family can measure ──────────────────────────────────────

# Group names are constants, not literals: they are used as dict keys in three modules,
# and a typo would otherwise surface as a KeyError deep inside a stage, not at import.
FLAT = "flat P/R/F1 (eq. 1)"
HIERARCHY = "hierarchy hP/hR/hF, CoPHE (eqs. 3-4)"
NEAR_MISS = "near-miss + error taxonomy (eq. 5)"
SEGMENT_RETRIEVAL = "segment P@S / R@S (eq. 2)"
TERM_RETRIEVAL = "term P@M / R@M (eq. 2)"
REACHABILITY = "reachability recall (eq. 6)"
BLOCKING_DEPTH = "blocking depth"
DEPTH_COST = "traversal cost by depth"
CALIBRATION = "calibration (eq. 7)"
RISK_COVERAGE = "risk-coverage (eq. 8)"
SLM_CALLS = "SLM calls per report"

#: Metric group → {method kind → "yes" | "no" | a short reason it does not apply}.
#: A reason string is printed verbatim in ``metric_availability`` so the results chapter never
#: shows an unexplained blank. Keys mirror the skeleton's own section names.
METRIC_SUPPORT: dict[str, dict[str, str]] = {
    FLAT: {
        "tree": "yes", "raghpo": "yes", "flat_topm": "yes", "ensemble": "yes",
        "phenobert": "yes", "rerank": "yes", "typed_span": "yes", "constrained": "yes",
        "autopcr": "yes"},
    HIERARCHY: {
        "tree": "yes", "raghpo": "yes", "flat_topm": "yes", "ensemble": "yes",
        "phenobert": "yes", "rerank": "yes", "typed_span": "yes", "constrained": "yes",
        "autopcr": "yes"},
    NEAR_MISS: {
        "tree": "yes", "raghpo": "yes", "flat_topm": "yes", "ensemble": "yes",
        "phenobert": "yes", "rerank": "yes", "typed_span": "yes", "constrained": "yes",
        "autopcr": "yes"},
    SEGMENT_RETRIEVAL: {
        "tree": "yes",
        "raghpo": "retrieval unit is phrase->HPO, not report-segment",
        "flat_topm": "yes",
        "ensemble": "no retrieval stage — generation is ungrounded in segments",
        "phenobert": "no retrieval stage — mentions are matched in place, not retrieved",
        # Same unit as raghpo, plus a second reason of its own: the emitted candidate list covers
        # accepted phrases only, because the full n-gram enumeration is millions of rows per
        # cohort. Scoring it would be scoring a truncation.
        "autopcr": "retrieval unit is phrase->HPO, not report-segment",
        "rerank": "inherits its candidates' provenance; it retrieves nothing itself",
        "typed_span": "spans are detected in place, not retrieved as ranked segments",
        # The retrieval unit is inverted: segment -> ranked terms, not term -> ranked segments.
        # The underlying own_max matrix could answer this, but the driver does not emit the
        # term-major ranking the metric scores, so claiming support would be claiming a file.
        "constrained": "retrieval unit is segment->term; the term->segment ranking is not emitted",
    },
    TERM_RETRIEVAL: {
        "tree": "yes",
        "raghpo": "no ranked ontology-wide candidate list is emitted",
        "flat_topm": "yes",
        "ensemble": "no candidate ranking — terms are generated, not retrieved",
        "phenobert": "no ranked candidate list — the CNN hierarchy emits one code per mention",
        "autopcr": "candidates are the top-k for one phrase, not a ranked ontology-wide list per "
                   "report",
        "rerank": "the pool is a set, not a ranked ontology-wide list",
        "typed_span": "Stage C emits a candidate list per span, not per report - the unit differs",
        # The one method whose candidate list *is* the measurement: recall@M is the hard ceiling
        # The constrained decoder selects within, and both halves are reported separately.
        "constrained": "yes",
    },
    REACHABILITY: {
        "tree": "yes",
        "raghpo": "no ontology traversal", "flat_topm": "no ontology traversal",
        "ensemble": "no ontology traversal",
        "phenobert": "no ontology traversal",
        "autopcr": "no ontology traversal",
        "rerank": "no ontology traversal",
        "typed_span": "no ontology traversal",
        "constrained": "no ontology traversal",
    },
    BLOCKING_DEPTH: {
        "tree": "yes",
        "raghpo": "no pruning decisions", "flat_topm": "no pruning decisions",
        "ensemble": "no pruning decisions",
        "phenobert": "no pruning decisions",
        "autopcr": "no pruning decisions — the tau_1/tau_2 band gates linking, not traversal",
        "rerank": "no pruning decisions",
        "typed_span": "no pruning decisions",
        "constrained": "no pruning decisions — retrieval truncates at M, it does not prune",
    },
    DEPTH_COST: {
        "tree": "yes",
        "raghpo": "no depth-structured cost", "flat_topm": "no depth-structured cost",
        "ensemble": "no depth-structured cost",
        "phenobert": "no depth-structured cost",
        "autopcr": "no depth-structured cost",
        "rerank": "no depth-structured cost",
        "typed_span": "no depth-structured cost",
        "constrained": "no depth-structured cost",
    },
    CALIBRATION: {
        "tree": "yes",
        "raghpo": "no per-term confidence is persisted",
        "flat_topm": "yes",
        "ensemble": "votes/n_models only — 9 atoms, uncalibrated by construction",
        "phenobert": "the match score is a fixed filter threshold (p1/p2/p3), not a per-term "
                     "confidence",
        # A score IS persisted, unlike raghpo, but on two incommensurable scales in one column:
        # a SapBERT cosine for a retrieval hit, and the flat -1.0 sentinel nn_model2 writes for
        # anything the linker chose. The linker's own HIGH/MEDIUM/LOW is consumed as a filter and
        # never reaches the artifact.
        "autopcr": "the score is a cosine or the -1.0 linker sentinel — two scales, neither a "
                   "per-term confidence",
        "rerank": "yes",
        # Stage D emits a softmax over the candidate options, so a presence probability exists.
        "typed_span": "yes",
        # Every candidate carries a sequence log-probability whether accepted or not, so the whole
        # distribution is observable, not only its accepted tail.
        "constrained": "yes",
    },
    RISK_COVERAGE: {
        "tree": "yes",
        "raghpo": "no per-term confidence is persisted",
        "flat_topm": "yes",
        "ensemble": "votes/n_models only — 9 atoms, uncalibrated by construction",
        "phenobert": "the match score is a fixed filter threshold (p1/p2/p3), not a per-term "
                     "confidence",
        # A score IS persisted, unlike raghpo, but on two incommensurable scales in one column:
        # a SapBERT cosine for a retrieval hit, and the flat -1.0 sentinel nn_model2 writes for
        # anything the linker chose. The linker's own HIGH/MEDIUM/LOW is consumed as a filter and
        # never reaches the artifact.
        "autopcr": "the score is a cosine or the -1.0 linker sentinel — two scales, neither a "
                   "per-term confidence",
        "rerank": "yes",
        # Abstention is an explicit output, so coverage is a real axis, not a sweep.
        "typed_span": "yes",
        "constrained": "yes",
    },
    SLM_CALLS: {
        "tree": "yes",
        "raghpo": "LLM invocations are not counted by the driver",
        "flat_topm": "yes",
        "ensemble": "per-sentence generation is not counted by the driver",
        "phenobert": "no LLM stage — zero calls by construction",
        # Counted, unlike raghpo's: LocalPrompter tallies every invocation and the driver writes it
        # to autopcr_timing.jsonl. It is also the one method whose call count is *data*, the
        # linker fires only inside the [tau_2, tau_1) band, so calls per report measure how often
        # retrieval was unsure.
        "autopcr": "yes",
        "rerank": "zero additional calls — the cost is the candidate sources' own",
        "typed_span": "yes",
        "constrained": "yes",
    },
}


def _verdict(spec: MethodSpec, metric_group: str) -> str:
    try:
        by_kind = METRIC_SUPPORT[metric_group]
    except KeyError:
        raise KeyError(
            f"unknown metric group {metric_group!r}; expected one of "
            f"{sorted(METRIC_SUPPORT)}") from None
    return by_kind.get(spec.kind, "no")


def supports(spec: MethodSpec, metric_group: str) -> bool:
    """Whether *metric_group* is measurable for this method at all."""
    return _verdict(spec, metric_group) == "yes"


def support_reason(spec: MethodSpec, metric_group: str) -> str:
    """Why *metric_group* does not apply, ``""`` when it does."""
    reason = _verdict(spec, metric_group)
    return "" if reason == "yes" else reason


# ── Filesystem discovery ─────────────────────────────────────────────────────

@dataclass
class Cell:
    """One (method, cohort) cell of the earlier grid, as found on disk."""

    method: str
    cohort: str
    run_dir: Path
    status: str                                  # "present" | "missing" | "partial"
    detail: str = ""                             # why, when not "present"
    operating_points: list[str] = field(default_factory=list)   # sweep subdir names, "" = flat
    #: ``{operating_point: {artifact_kind: path}}``. Only points listed in
    #: :attr:`operating_points` appear, and only their predictions file is known to exist.
    files: dict[str, dict[str, Path]] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """True when the run's files are complete and can be scored."""
        return self.status == "present"


@dataclass
class Availability:
    """Which (method, cohort) runs exist on disk, and in what state."""
    cells: dict[tuple[str, str], Cell]

    def get(self, method: str, cohort: str) -> Cell:
        """The cell for *cohort*, resolving a derived cohort to the run it is scored from.

        A derived cohort has no directory of its own, so its availability *is* that of its source:
        if the ``gsc`` run is missing or unmerged, so is every view of it.
        """
        return self.cells[(method, artifact_cohort(cohort))]

    def usable_cells(self) -> list[Cell]:
        """The cells whose runs are complete."""
        return [c for c in self.cells.values() if c.usable]

    def counts(self) -> dict[str, int]:
        """Number of cells per status: present, missing and partial."""
        out = {"present": 0, "missing": 0, "partial": 0}
        for c in self.cells.values():
            out[c.status] += 1
        return out


def _shard_problems(run_dir: Path) -> tuple[list[str], list[str]]:
    """``(unmerged, stale)`` shard files anywhere under *run_dir*, including ``tau_*/``.

    The presence of shard files is **not** itself a problem: ``src/hpo_extraction/evaluation/merge_shards.py``
    writes the consolidated artifact alongside its inputs and leaves them in place,
    so a correctly merged run still has them. Treating any shard as unmerged would permanently
    refuse every array job in the repo.

    Two things are genuine problems, and they are reported separately because the fix differs:

    * **unmerged**, a shard whose consolidated counterpart does not exist. Nothing has been
      merged. The metrics would silently describe one shard's slice of the cohort.
    * **stale**, the consolidated file exists but predates a shard, so a later session's reports
      are on disk yet absent from the merged artifact. Re-running the merge is a no-op if the
      timestamps mislead, and correct if they do not.
    """
    unmerged: list[str] = []
    stale: list[str] = []
    for path in sorted(run_dir.rglob("*.jsonl")):
        if not _SHARD_RE.search(path.name):
            continue
        merged = path.with_name(_SHARD_RE.sub("", path.name))
        name = str(path.relative_to(run_dir))
        if not merged.is_file():
            unmerged.append(name)
        elif merged.stat().st_mtime < path.stat().st_mtime:
            stale.append(name)
    return unmerged, stale


def _tree_files(run_dir: Path, spec: MethodSpec, point: str) -> dict[str, Path]:
    v = spec.variant
    return {
        "predictions": run_dir / point / f"{v}_predictions.jsonl",
        "nodes": run_dir / point / f"{v}_nodes.jsonl",
        "traversal": run_dir / point / f"{v}_traversal.jsonl",
        "calls": run_dir / f"{v}_calls.jsonl",
        "node_metadata": run_dir / f"{v}_node_metadata.jsonl",
    }


#: Tree operating-point directory names. Two layouts coexist by design: runs that fix a single
#: accept threshold keep the original 1-D ``tau_{prune}`` (an earlier exploratory run, and every dump written before
#: The accept sweep existed), while a real 2-D sweep adds ``_acc_{accept}``. Parsing both is what
#: lets old and new dumps be scored side by side, see ``tree_experiment._tau_dir_name``.
_TAU_DIR_RE = re.compile(r"^tau_(?P<prune>[0-9.eE+-]+)(?:_acc_(?P<accept>[0-9.eE+-]+))?$")


def parse_tau_point(name: str) -> tuple[float, float | None]:
    """``tau_0.02`` -> ``(0.02, None)``; ``tau_0.02_acc_0.9`` -> ``(0.02, 0.9)``.

    Raises ``ValueError`` on anything else, not guessing, a directory that looks like an
    configuration but does not parse is a bug in the driver or a stray path, and silently
    dropping it would quietly shrink a sweep.
    """
    m = _TAU_DIR_RE.match(name)
    if not m:
        raise ValueError(f"not a tree operating-point directory: {name!r}")
    accept = m.group("accept")
    return float(m.group("prune")), (float(accept) if accept is not None else None)


def _sweep_points(run_dir: Path, spec: MethodSpec) -> list[str]:
    """Operating-point subdirectory names, sorted so tables read in a sensible order."""
    if spec.sweep is None:
        return [""]
    points = sorted(p.name for p in run_dir.glob(spec.sweep) if p.is_dir())
    if spec.kind == "tree":
        # Sort on (tau_prune, tau_accept) so a 2-D sweep reads as a grid. ``None`` sorts before any
        # real accept threshold, keeping 1-D points ahead of the 2-D ones for the same tau_prune.
        points = [n for n in points if _TAU_DIR_RE.match(n)]
        points.sort(key=lambda n: (lambda p, a: (p, -1.0 if a is None else a))(*parse_tau_point(n)))
    elif spec.kind == "ensemble":
        points.sort(key=lambda n: int(n.removeprefix("vote_k")))
        if (run_dir / "agg_plurality").is_dir():
            points.append("agg_plurality")
    return points


def _resolve_files(run_dir: Path, spec: MethodSpec, point: str) -> dict[str, Path]:
    """Artifact paths for one configuration. Existence is checked by the caller."""
    if spec.kind == "tree":
        return _tree_files(run_dir, spec, point)
    if spec.kind == "raghpo":
        return {
            "predictions": run_dir / f"{spec.variant}_predictions.jsonl",
            "segments": run_dir / f"{spec.variant}_retrieved_segments.jsonl",
        }
    if spec.kind == "flat_topm":
        return {
            "predictions": run_dir / f"{spec.variant}_predictions.jsonl",
            "calls": run_dir / f"{spec.variant}_calls.jsonl",
            "timing": run_dir / f"{spec.variant}_timing.jsonl",
        }
    if spec.kind == "ensemble":
        return {
            "predictions": run_dir / point / f"{spec.variant}_predictions.jsonl",
            "detections": run_dir / f"{spec.variant}_detections.jsonl",
            "agg_summary": run_dir / f"{spec.variant}_agg_summary.csv",
        }
    if spec.kind == "phenobert":
        return {
            "predictions": run_dir / f"{spec.variant}_predictions.jsonl",
            "detections": run_dir / f"{spec.variant}_detections.jsonl",
            "timing": run_dir / f"{spec.variant}_timing.jsonl",
        }
    if spec.kind == "autopcr":
        return {
            "predictions": run_dir / f"{spec.variant}_predictions.jsonl",
            # One row per accepted mention, carrying `linked_by`, whether SapBERT's index or the
            # language model produced it. That split is the reason this method is in the table.
            "detections": run_dir / f"{spec.variant}_detections.jsonl",
            # Ranked candidates for accepted phrases only. See write_segments' docstring for why
            # The full enumeration is not emitted and why segment retrieval is unsupported below.
            "segments": run_dir / f"{spec.variant}_retrieved_segments.jsonl",
            "timing": run_dir / f"{spec.variant}_timing.jsonl",
        }
    if spec.kind == "rerank":
        return {
            # One directory per probability threshold, mirroring the ensemble's vote_k* layout.
            "predictions": run_dir / point / f"{spec.variant}_predictions.jsonl",
            # Every pooled candidate's probability, accepted or not, what §Calibration and
            # §Risk--coverage need, and what no other method in the comparison persists.
            "scores": run_dir / f"{spec.variant}_scores.jsonl",
        }
    if spec.kind == "constrained":
        return {
            # One directory per (acceptance threshold, vote) configuration, the an earlier exploratory run
            # `tau_*_acc_*` layout, two axes in one name.
            "predictions": run_dir / point / f"{spec.variant}_predictions.jsonl",
            # Every candidate label string's sequence log-probability, accepted or not. This is
            # what §Calibration and §Risk--coverage read, and it is also the substrate the whole
            # re-run grid (M, scoring rule, surface aggregation, gate) is derived from.
            "scores": run_dir / f"{spec.variant}_scores.jsonl",
            "agg_summary": run_dir / f"{spec.variant}_agg_summary.csv",
        }
    if spec.kind == "typed_span":
        return {
            # earlier writes straight into {exp_id}/{cohort}/, no operating-point sweep, so `point`
            # is "" and plays no part here. `predictions` is the shared contract every the earlier runs
            # metric reads. The rest is the earlier runs'own output, which nothing in this comparison
            # consumes yet but which names the file to open when it does.
            "predictions": run_dir / f"{spec.variant}_predictions.jsonl",
            # One row per finding: term, assertion status, char-offset span, evidence, stage_trace.
            "records": run_dir / f"{spec.variant}_records.jsonl",
            # Stage G only (an earlier exploratory run). Absent for every earlier row, hence `.get()` at the reader.
            "escalations": run_dir / f"{spec.variant}_escalations.jsonl",
        }
    raise ValueError(f"unknown method kind: {spec.kind}")


def discover_cell(results_dir: Path, spec: MethodSpec, cohort: str) -> Cell:
    """Classify one (method, cohort) cell as present / missing / partial."""
    run_dir = Path(results_dir) / spec.exp_id / cohort
    if spec.subdir:
        run_dir = run_dir / spec.subdir

    if not run_dir.is_dir():
        return Cell(spec.key, cohort, run_dir, "missing", f"no run directory at {run_dir}")

    unmerged, stale = _shard_problems(run_dir)
    if unmerged:
        return Cell(
            spec.key, cohort, run_dir, "partial",
            f"{len(unmerged)} shard file(s) have no merged counterpart — run "
            f"`python src/evaluation/merge_shards.py {run_dir}` first "
            f"(e.g. {unmerged[0]})",
        )
    if stale:
        return Cell(
            spec.key, cohort, run_dir, "partial",
            f"{len(stale)} shard file(s) are newer than the merged artifact, so a later "
            f"session's reports are on disk but not in it — re-run "
            f"`python src/evaluation/merge_shards.py {run_dir}` (e.g. {stale[0]})",
        )

    points = _sweep_points(run_dir, spec)
    if not points:
        return Cell(spec.key, cohort, run_dir, "missing",
                    f"no operating-point subdirectory matching {spec.sweep!r} under {run_dir}")

    # A point counts only when its predictions file exists, that is the one artifact every
    # metric family needs. Points that ran out of wall-clock mid-sweep are dropped, not averaged in.
    usable, files = [], {}
    for point in points:
        resolved = _resolve_files(run_dir, spec, point)
        if resolved["predictions"].is_file():
            usable.append(point)
            files[point] = resolved

    if not usable:
        return Cell(spec.key, cohort, run_dir, "missing",
                    f"no predictions file under {run_dir} "
                    f"(expected e.g. {_resolve_files(run_dir, spec, points[0])['predictions']})")

    detail = ""
    status = "present"
    if len(usable) < len(points):
        status = "partial"
        detail = (f"{len(usable)}/{len(points)} operating points have predictions; "
                  f"missing: {', '.join(p for p in points if p not in usable)}")

    return Cell(spec.key, cohort, run_dir, status, detail, usable, files)


def discover(results_dir: str | Path) -> Availability:
    """Scan *results_dir* for every (method, cohort) cell of the earlier grid.

    Only :data:`BASE_COHORTS` are scanned. A derived cohort reads the same directory as its source,
    so giving it its own row would double ``availability.md`` without adding a fact;
    :meth:`Availability.get` resolves it instead.
    """
    results_dir = Path(results_dir)
    return Availability({
        (spec.key, cohort): discover_cell(results_dir, spec, cohort)
        for spec in METHODS
        for cohort in BASE_COHORTS
    })


# ── Tables ───────────────────────────────────────────────────────────────────

def availability_rows(avail: Availability) -> list[dict]:
    """One row per (method, cohort) cell, the input to ``availability.md``."""
    rows = []
    for spec in METHODS:
        for cohort in BASE_COHORTS:
            cell = avail.get(spec.key, cohort)
            rows.append({
                "method": spec.key,
                "label": spec.label,
                "experiment": spec.exp_id,
                "cohort": cohort,
                "status": cell.status,
                "n_operating_points": len(cell.operating_points),
                "operating_points": " ".join(cell.operating_points),
                "detail": cell.detail,
            })
    return rows


def metric_support_rows() -> list[dict]:
    """One row per (metric group, method), why a blank in the results is blank."""
    rows = []
    for group, by_kind in METRIC_SUPPORT.items():
        for spec in METHODS:
            reason = by_kind.get(spec.kind, "no")
            rows.append({
                "metric_group": group,
                "method": spec.key,
                "label": spec.label,
                "supported": reason == "yes",
                "reason": "" if reason == "yes" else reason,
            })
    return rows


def rerun_command(spec: MethodSpec, cohort: str) -> str:
    """The cluster script that would fill a missing cell, for ``availability.md``.

    The names follow the cluster scripts of the original repository, ``run_<ds>_<exp_id>.sh``
    (their replacements are the templates in ``slurm/``, see ``docs/thesis_map.md``). The
    ensemble is the one two-stage member, so it needs both jobs. A derived cohort resolves to its
    source's script, there is no ``run_gsc_raghpo_*.sh`` to point anyone at.
    """
    cohort = artifact_cohort(cohort)
    if spec.kind == "ensemble":
        return (f"sbatch cluster/run_{cohort}_{spec.exp_id}_extract.sh"
                f"  # then run_{cohort}_{spec.exp_id}_aggregate.sh")
    return f"sbatch cluster/run_{cohort}_{spec.exp_id}.sh"
