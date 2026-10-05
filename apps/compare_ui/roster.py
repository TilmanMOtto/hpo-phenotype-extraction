"""The five method columns: which experiment each one is, and where its evidence lives.

This is a *narrower* roster than ``experiments/06_comparison/comparison/roster.py``,
and it is narrower in a way that has to be stated. The comparison carries seven rows because a results
table wants the 8B and 70B conditions of the two external systems side by side. A reading surface wants
five columns that fit on a screen, so the 8B rows are dropped and the 70B ones kept -- the same
choice the thesis text makes when it names "AutoPCR" without a size.

What is **not** re-decided here is which artifact is authoritative. ``comparison``'s ``manifest.json``
names the prediction file behind every number in Table 1, and :meth:`Method.predictions_for`
resolves to those, per cohort. A column in this app that disagreed with the table it is
meant to explain would be worse than no column.

The per-cohort split is not cosmetic for the two **protocol** methods, and neither case is derivable
from the other:

``phenojury``
    HCY and GSC+ come from two different runs of ``phenojury_protocol``, because the folds have to be built
    on the documents the prediction set is then pooled over. Deriving one path from the other would
    quietly read the 228-document GSC+ run and report it under the 114-document heading.
``treephenorag``
    On HCY the configuration is chosen inside nested CV (``nested_cv_pooled.csv``). On GSC+
    **nothing is fitted at all**: the configuration selected on all of HCY is carried over
    unchanged (``gsc_raghpo_ann_transferred.csv``). That is a different estimator and a different
    file, and the column says so rather than presenting a transfer as an out-of-fold estimate.

``evidence`` names the per-term records the adapter in :mod:`apps.compare_ui.methods` reads. Where a
method records less than the others that is written down in ``evidence_note``, not papered
over: AutoPCR and RAG-HPO both call an LLM and neither kept its answer, and a panel that quietly
showed a reconstructed prompt without saying so would be claiming to show a model's reasoning while
showing our own.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Method:
    """One column."""

    key: str
    label: str
    #: Two or three characters for the gutter header. The gutters are ~64 px wide.
    short: str
    #: Which family the reader should hold it in -- ``external`` published work, ``ours`` a
    #: contribution of this thesis. Matches ``comparison.roster``'s own ``family``.
    family: str
    #: The experiment directory the *evidence* comes from, relative to ``output_base``.
    exp_id: str
    #: The ``{variant}`` prefix its artifacts carry, or ``""`` for the protocol methods whose
    #: predictions are a two-column CSV, not a predictions JSONL.
    variant: str = ""
    #: ``{cohort key: path}`` for a protocol method's pooled prediction set, ``{output_base}``-
    #: relative. Declared per cohort, not templated, because the two protocol drivers do
    #: not lay their output out the same way and pretending they do would read the wrong file.
    predictions: dict = field(default_factory=dict)
    #: What the adapter can show, in the reader's words.
    evidence: str = ""
    #: A note that must travel with every panel this method draws.
    evidence_note: str = ""
    #: A further note that applies only in the named cohort, ``{cohort key: text}``.
    cohort_note: dict = field(default_factory=dict)
    #: Extra experiment directories the adapter needs, keyed by a name it knows.
    extra: dict = field(default_factory=dict)

    def predictions_for(self, cohort) -> str:
        """The prediction set for *cohort*, ``{output_base}``-relative, or ``""``."""
        return self.predictions.get(getattr(cohort, "key", str(cohort)), "")

    def note_for(self, cohort) -> str:
        """``evidence_note`` with this cohort's extra note appended, if it has one."""
        extra = self.cohort_note.get(getattr(cohort, "key", str(cohort)), "")
        return " ".join(part for part in (self.evidence_note, extra) if part)


ROSTER: tuple[Method, ...] = (
    Method(
        key="phenobert",
        label="PhenoBERT",
        short="PB",
        family="external",
        exp_id="baseline_phenobert",
        variant="phenobert",
        evidence="the matched phrase with its character offsets, the tagger's own score, and "
                 "whether the span was read as negated",
    ),
    Method(
        key="autopcr_70b",
        label="AutoPCR (LLaMA-3 70B linker)",
        short="APC",
        family="external",
        exp_id="baseline_autopcr_70b",
        variant="autopcr",
        evidence="the extracted phrase, which of the three linking routes decided it "
                 "(dictionary / retrieval / LLM), and the ranked candidate menu the linker saw",
        evidence_note="The linker's prompt is reconstructed from the candidate menu, not read "
                      "from disk, and its answer was never recorded -- "
                      "hpo_extraction.baselines.autopcr_runner.LocalPrompter keeps a call counter and drops both. "
                      "Runs AutoPCR's published extraction (ee=neural++, with the benepar "
                      "constituency parser) and a local LLaMA linker.",
    ),
    Method(
        key="raghpo_70b",
        label="RAG-HPO (LLaMA-3 70B)",
        short="RAG",
        family="external",
        exp_id="baseline_raghpo_70b",
        variant="rag_hpo",
        evidence="the phrase it extracted, the sentence it came from, and the ranked HPO "
                 "candidates with the one its verifier said Yes to",
        evidence_note="RAG-HPO computes a rationale per selection (`raw_llm_resp`, "
                      "`llm_parse_reason`) and hpo_extraction.baselines.rag_hpo_runner drops both before writing, so "
                      "what is shown is the candidate menu and the verdict, not the reasoning.",
        cohort_note={"gsc": "On this cohort the ground truth is RAG-HPO's own re-annotation, so this "
                            "column is the one method being scored against its authors' "
                            "annotation policy rather than somebody else's."},
    ),
    Method(
        key="phenojury",
        label="PhenoJury (8 jurors)",
        short="PJ",
        family="ours",
        exp_id="phenojury_generation_other_prompts",
        # Chapter 6's "PhenoJury (full pool, 8)" -- the comparison's `protocol_predictions`: all eight
        # jurors, the prompt chosen inside each outer training split. NOT `prompt_<p>.csv`, the
        # forced-prompt pools beside it that only t1_sensitivity.csv reads. Drawing one of those
        # here put a different system than the chapter's under its name (on GSC+ the full pool
        # never chooses q2 -- it is p0_baseline in 50/50 folds).
        predictions={
            "hcy": "comparison_inputs/phenojury_protocol/hcy/predictions/full_pool.csv",
            "gsc": "comparison_inputs/phenojury_protocol/gsc/predictions/full_pool.csv",
        },
        evidence="every juror's verbatim generation for the sentence, which of them voted, and "
                 "the vote threshold the outer fold selected",
        extra={"protocol": "phenojury_protocol", "grid": "phenojury_normalisation"},
    ),
    Method(
        key="treephenorag",
        label="TreePhenoRAG",
        short="TPR",
        family="ours",
        exp_id="treephenorag_protocol",
        predictions={
            "hcy": "treephenorag_protocol/predictions/nested_cv_pooled.csv",
            "gsc": "treephenorag_protocol/predictions/gsc_raghpo_ann_transferred.csv",
        },
        evidence="the node's pruning and acceptance scores against the fold's own thresholds, the "
                 "verifier calls behind them, and -- for a miss -- the ancestor that blocked it",
        evidence_note="The pooled prediction set this column is scored on has no writer in tracked "
                      "code; its digest is recorded in the bundle so a change is visible.",
        cohort_note={"gsc": "On GSC+ the configuration selected on all of HCY is applied "
                            "unchanged -- no folds, no risk control, nothing fitted on this "
                            "cohort -- so there are no per-fold thresholds to read a score "
                            "against, and the panel shows the verdict alone."},
        extra={"cache": "treephenorag_protocol", "calls": "treephenorag_scores_terminfo",
               "calls_exemplar": "treephenorag_scores_synthetic"},
    ),
)

ROSTER_BY_KEY: dict[str, Method] = {m.key: m for m in ROSTER}

#: Column order on screen, left to right. External systems first, then ours -- so the reader walks
#: from the published baselines toward the contribution, not being shown the answer first.
ORDER: tuple[str, ...] = tuple(m.key for m in ROSTER)

#: The three columns the GSC+ deep-dive frame is drawn on. Naming them here, not in the
#: selection script is deliberate: the cells a document was drawn into are a claim about *these*
#: columns, and a roster change that renamed one of them must not leave the frame silently
#: describing a method the app no longer has.
FRAME_METHODS: tuple[str, ...] = ("phenojury", "raghpo_70b", "autopcr_70b")


def get(key: str) -> Method:
    """The method *key* of the roster."""
    return ROSTER_BY_KEY[key]


def labels() -> dict[str, str]:
    """``{method key: display label}``."""
    return {m.key: m.label for m in ROSTER}
