"""The five systems the discussion compares, and where each one's predictions come from.

This is **not** ``result_tables/discovery.py``. That module enumerates every run on disk so
the results chapter can say which measurements exist. This one names a *closed* roster, the
comparison the discussion actually makes, and refuses to grow by accident. A row here is a claim
that the system belongs in the main table, which is an editorial decision, not a filesystem
fact.

Two provenances, and the difference is the single most important note in the output:

``fixed``      The method has nothing to tune. Its ``*_predictions.jsonl`` is the whole cohort at
               its one configuration, so the number is a plain cohort estimate.
``protocol``   The method's configuration is chosen **inside nested cross-validation**, so the
               prediction for a report comes from the fold that report was held out of. The file
               is the pooled out-of-fold set written by ``hpo_extraction.evaluation.prediction_sets``. The number
               is an out-of-fold estimate over the same reports, which is a different estimator and
               is marked as such in every table.

Mixing the two in one column is defensible only because the alternative is worse: quoting a
same-data-tuned number for the methods that *can* be tuned, against untuned baselines, is the bias
``an earlier exploratory run`` and ``treephenorag_protocol`` exist to remove. The ``estimator`` column carries the distinction
into the CSV so no reader has to remember it.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The two cohorts the discussion reports on.
#:
#: ``hcy`` is the curated ground truth (``curated_gold_<date>/hcy_ground_truth_curated.csv``); ``hcy_dir``
#: never leaves the cluster. ``gsc_raghpo_ann`` is RAG-HPO's own 114 documents under RAG-HPO's own
#: re-annotation, the only GSC+ view a published number in ``LITERATURE`` may be read against.
COHORTS: tuple[str, ...] = ("hcy", "gsc_raghpo_ann", "gsc_2024_eval_206")

#: ``reported cohort -> the run directory its artifacts actually live in``.
#:
#: ``gsc_raghpo_ann`` is a **derived** cohort: no driver was ever run against it. It is the `gsc`
#: run restricted to RAG-HPO's 114 documents and scored against RAG-HPO's annotation. Every driver
#: writes one summary line per report, so restricting at scoring time gives bit-identical numbers
#: to re-running inference on the subset, see ``result_tables/discovery.py``'s ``DERIVED_COHORTS``,
#: which this mirrors. Getting it wrong does not produce a wrong number, it produces a missing
#: file, which is why the table reports the path it looked for.
ARTIFACT_COHORT: dict[str, str] = {"gsc_raghpo_ann": "gsc", "gsc_2024_eval_206": "gsc"}


def artifact_cohort(cohort: str) -> str:
    """The run directory *cohort* is scored from, itself, unless derived."""
    return ARTIFACT_COHORT.get(cohort, cohort)


@dataclass(frozen=True)
class Row:
    """One system in the comparison."""

    key: str
    label: str
    #: ``external`` (a published method we re-ran) or ``ours``.
    family: str
    #: ``fixed`` | ``protocol``, see the module docstring.
    provenance: str
    exp_id: str
    #: Filename prefix inside ``{results_dir}/{exp_id}/{cohort}/`` for ``fixed`` rows.
    variant: str = ""
    #: Default path **relative to** ``{results_dir}/{exp_id}`` for ``protocol`` rows, with
    #: ``{cohort}`` substituted. Declared per row rather than derived, because the two protocol
    #: drivers do not lay their output directories out the same way and pretending they do would
    #: silently read the wrong cohort. Overridable per (method, cohort) from config, see
    #: :func:`predictions_path`.
    predictions: str = ""
    note: str = ""
    #: Per-cohort estimator overrides, as ``((cohort, label), ...)``. A tuple, not a dict
    #: because the dataclass is fixed.
    #:
    #: Only TreePhenoRAG needs one, and the reason is worth stating: on HCY its configuration is
    #: chosen inside nested CV, but on GSC+ **nothing is fitted at all**, the configuration
    #: selected on all of HCY is carried over unchanged, no folds, no risk control. That is a
    #: third estimator and the strictest of the three, not a variant of either other one, and
    #: collapsing it into "pooled out-of-fold" would understate the claim, not overstate it.
    estimators: tuple = ()

    @property
    def estimator(self) -> str:
        """How this row's predictions were produced: a single fixed run or pooled out-of-fold predictions."""
        return ("cohort (single operating point, nothing tuned)" if self.provenance == "fixed"
                else "pooled out-of-fold (nested CV)")

    def estimator_for(self, cohort: str) -> str:
        """The estimator description for one cohort, if the row declares one, else :attr:`estimator`."""
        for name, label in self.estimators:
            if name == cohort:
                return label
        return self.estimator


#: How PhenoJury's prompt is chosen: inside every outer training split, as chapter 5's
#: protocol does it (the PhenoJury protocol's in-fold pair selection, 2026-09-27). Which prompts the folds chose
#: is not a constant of this module -- it is read per cohort from the preparation run's
#: ``s7_full_pool_pair_frequency.csv`` and reported in ``t1_sensitivity.csv``.
#:
#: This used to be a per-corpus constant (``q2_sentence_last`` on HCY, ``p0_baseline`` on GSC+)
#: taken from the PhenoJury protocol's development-split grid. That split holds ~80 % of the reports the number is
#: then scored on, so the choice was partly in-sample. Before that it was one prompt, ``q2``, forced
#: on every cohort, which made chapter 6's PhenoJury rows a different system from chapter 5's.
#:
#: The **reader** stays fixed, and is the less obvious half: left to choose, the grid picks
#: ``phenobert_candidates`` on HCY, on the 228 and on the 206 abstracts, but ``sapbert`` on the
#: 114-document cohort, so an unpinned reader there would be a different system from chapter 5's.
PHENOJURY_PROMPT_RULE = "prompt selected in-fold"
PHENOJURY_NORMALISER = "phenobert_candidates"

#: The cost-constrained jury, and the file the PhenoJury protocol writes it to.
#:
#: These three are the jurors the PhenoJury protocol's backward elimination keeps most often (`s5_selection_
#: frequency.csv`: medgemma 50/50, medpsy 48/50, phi4 47/50 on HCY). Naming them is therefore a
#: reading of a result, which is why the row is marked exploratory everywhere it appears and why
#: it is absent from the PhenoJury protocol's fixed in advance test family.
#:
#: The slug must match ``phenojury_protocol``'s own ``_slug("fixed:core3")``. It is spelled out here rather
#: than derived because getting it wrong does not produce a wrong number -- it produces a missing
#: file, and the table then reports the path it looked for.
PHENOJURY_CORE3 = ("medgemma", "medpsy", "phi4")
PHENOJURY_CORE3_SLUG = "fixed_core3"

ROSTER: tuple[Row, ...] = (
    Row("raghpo_8b", "RAG-HPO (Llama-3.1 8B)", "external", "fixed",
        "baseline_raghpo_8b", variant="rag_hpo"),
    Row("raghpo_70b", "RAG-HPO (Llama-3.3 70B)", "external", "fixed",
        "baseline_raghpo_70b", variant="rag_hpo"),
    # AutoPCR at both linker sizes. The pair differs only in the linking model, so the
    # difference prices what 9x the linker buys with retrieval held fixed. Both run the PUBLISHED
    # extraction, ee="neural++" (Stanza + benepar constituents + coordination split. One parse
    # shared by both), since 2026-09-26. The ee="neural" rows were a lower bound and are archived
    # as output/_archive/*_ee-neural. What still differs from the paper is the linker (a local
    # LLaMA, not Qwen3-Next-80B), which the row label names, see third_party/AutoPCR/PATCHES.md.
    Row("autopcr_8b", "AutoPCR (Llama-3.1 8B linker)", "external", "fixed",
        "baseline_autopcr_8b", variant="autopcr",
        note="ee=neural++ (published extraction); local LLaMA linker, not the paper's Qwen3-Next-80B"),
    Row("autopcr_70b", "AutoPCR (Llama-3.3 70B linker)", "external", "fixed",
        "baseline_autopcr_70b", variant="autopcr",
        note="ee=neural++ (published extraction); local LLaMA linker, not the paper's Qwen3-Next-80B"),
    Row("phenobert", "PhenoBERT", "external", "fixed",
        "baseline_phenobert", variant="phenobert"),
    # PhenoJury twice: the whole pool, and the cost-constrained jury.
    #
    # The two answer different questions and neither substitutes for the other. The full pool is
    # The ceiling the ensemble can reach; `core3` is what it costs to reach most of it. Three of
    # The eight jurors carry essentially the whole effect -- on GSC+ they are within 0.006 micro-F1
    # of all eight (p = 0.63) at three eighths of the generation budget, and on HCY they give up
    # 0.022 (p = 0.0086). Reporting only the pool hides the deployment result. Reporting only
    # `core3` hides what was given up for it.
    #
    # `core3` is **exploratory and says so**: the three models were named after reading the
    # selection frequencies in the PhenoJury protocol's S5, so it is not in that experiment's
    # fixed in advance test family and carries no Holm-corrected p-value. A comparator chosen after
    # reading a result and then folded into the family would be arithmetic dressed as inference.
    Row("phenojury", "PhenoJury (full pool, 8)", "ours", "protocol",
        "phenojury_protocol",
        # The full pool with the prompt selected in-fold. The config's `protocol_predictions`
        # names the exact file per cohort. This is only the fallback.
        predictions="{cohort}/predictions/full_pool.csv",
        note=(f"full pool, {PHENOJURY_PROMPT_RULE}, + {PHENOJURY_NORMALISER}; "
              f"rule/unit/k by inner CV")),
    Row("phenojury_core3", "PhenoJury (core-3)", "ours", "protocol",
        "phenojury_protocol",
        predictions="{cohort}/predictions/" + PHENOJURY_CORE3_SLUG + ".csv",
        note=("fixed jury " + " + ".join(PHENOJURY_CORE3) + f", {PHENOJURY_PROMPT_RULE}, + "
              f"{PHENOJURY_NORMALISER}; rule/unit/k by inner CV; jury named after reading "
              f"exp14_04's selection frequencies, so exploratory")),
    # TreePhenoRAG's two cohorts are scored under DIFFERENT protocols, by the TreePhenoRAG protocol's design, and
    # The `estimators` override is where that is recorded, not glossed:
    #
    #   HCY   `predictions/nested_cv_pooled.csv`, the configuration (retrieval index, both
    #         poolings, tau_accept) chosen inside nested CV, with tau_prune set by conformal risk
    #         control on the training folds. Pooled out-of-fold, like PhenoJury.
    #   GSC+  `predictions/gsc_raghpo_ann_transferred.csv`, the configuration selected on ALL of
    #         HCY, applied to the abstracts unchanged. No folds, no CRC, nothing fitted on GSC+ at
    #         all: the TreePhenoRAG protocol treats it as a genre-transfer test, not a second training set
    #         (`treephenorag_protocol/README.md` §2.1).
    #
    # The GSC+ file appears when the queued `treephenorag_protocol` job runs, which is gated on the
    # `treephenorag_scores_terminfo` cache array. Until then the cell reports `missing` with the path it looked
    # for, and fills in with no code change.
    Row("treephenorag", "TreePhenoRAG", "ours", "protocol",
        "treephenorag_protocol",
        predictions="predictions/nested_cv_pooled.csv",
        note="operating point by nested CV + conformal risk control",
        estimators=(("gsc_raghpo_ann",
                     "transfer (HCY-selected configuration, nothing fitted on this cohort)"),)),
)

ROSTER_BY_KEY: dict[str, Row] = {r.key: r for r in ROSTER}


def predictions_path(row: Row, results_dir: str, cohort: str, override: str = "") -> str:
    """Where *row*'s prediction sets for *cohort* are, as a string path.

    *override* is the config's ``protocol_predictions[key][cohort]`` entry, and it exists because a
    protocol driver's output directory is named after **the cohort it was run on**, which is not
    always the cohort this table reports. ``phenojury_protocol``'s GSC+ condition is the live case: its committed
    run is over the full 228-document corpus ground truth, so PhenoJury on ``gsc_raghpo_ann`` has to come
    from a separate run whose folds were built on RAG-HPO's 114 documents. Deriving the path would
    quietly read the 228-document one and report it under the 114-document heading.

    ``{results_dir}`` and ``{cohort}`` are substituted into an override before it is used.
    """
    from pathlib import Path

    if override:
        return str(Path(override.format(results_dir=results_dir, cohort=cohort)))
    base = Path(results_dir) / row.exp_id
    if row.provenance == "protocol":
        return str(base / row.predictions.format(cohort=cohort))
    return str(base / artifact_cohort(cohort) / f"{row.variant}_predictions.jsonl")


# ── Published numbers ────────────────────────────────────────────────────────

#: What a published row must carry beyond its numbers. ``compare_against`` names the column of
#: ``t1_overall.csv`` the figure is defined the same way as, a per-case average is not a micro
#: average and printing them in one column would be a category error. ``comparable`` says whether
#: The row shares this table's cohort AND ground truth. When it is false, ``why_not`` must say why, and the
#: row is written to its own file, not into the main table.
LITERATURE_FIELDS: tuple[str, ...] = (
    "label", "system_of", "source", "cohort", "n_documents", "n_gold_terms", "unit",
    "averaging", "compare_against", "comparable", "why_not", "precision", "recall", "f1",
    "tp", "fp", "fn", "note",
)


# ── The configuration, read from the driver's own artifacts ────────────────

def _tree_operating_point(exp_dir) -> str:
    """TreePhenoRAG's selected configuration, plus how uniformly it was selected.

    The protocol chooses the retrieval index, both poolings, ``tau_accept`` and ``S`` **inside each
    fold**, so "the" configuration is a modal description of five selections, not a setting the
    run was configured with. Printing it flat would overclaim: on HCY, repetition 0 splits 4
    ``ontology_r3`` to 1 ``exemplar``, so one fold's reports were predicted from a different
    retrieval index than the label suggests. The unanimity fractions are appended for that
    reason, and ``tau_prune`` is shown as the modal value because conformal risk control sets it
    per fold (it ranges 2.03e-4 to 2.74e-4 across HCY's five).

    Returns ``""`` when the artifacts are absent, so a missing cell degrades to a blank column,
    not an exception.
    """
    import csv
    import json
    from collections import Counter
    from pathlib import Path

    exp_dir = Path(exp_dir)
    chosen_path = exp_dir / "selected_configuration.json"
    if not chosen_path.exists():
        return ""
    chosen = json.loads(chosen_path.read_text(encoding="utf-8"))
    parts = [
        str(chosen.get("retrieval_index", "")),
        f"{chosen.get('pool_pr', '')} expansion",
        f"tau_prune {float(chosen.get('tau_prune', 0)):.2e}",
        f"{chosen.get('pool_acc', '')} acceptance",
        f"tau_accept {chosen.get('tau_accept', '')}",
        f"S={chosen.get('S', '')}",
    ]
    point = " / ".join(p for p in parts if p.strip(" /"))

    # How uniformly each axis was selected, over the repetition the pooled set comes from.
    choices_path = exp_dir / "tables" / "selection_choices.csv"
    if choices_path.exists():
        with open(choices_path, newline="", encoding="utf-8") as fh:
            folds = [r for r in csv.DictReader(fh) if str(r.get("repetition")) == "0"]
        if folds:
            agreed = []
            for axis, key in (("index", "retrieval_index"), ("poolings", "pool_pr"),
                              ("tau_accept", "tau_accept")):
                modal, n = Counter(r.get(key) for r in folds).most_common(1)[0]
                agreed.append(f"{axis} {n}/{len(folds)}")
            point += f" (modal over {len(folds)} folds: " + ", ".join(agreed) + ")"

    # Verifier calls per report, from the ladder rung that IS this configuration.
    quality_path = exp_dir / "tables" / "core_quality.csv"
    if quality_path.exists():
        with open(quality_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("config", "").startswith("V3") and row.get("calls_per_report"):
                    point += f"; {float(row['calls_per_report']):,.0f} verifier calls/report"
                    break
    return point


def operating_point(row: Row, results_dir: str, cohort: str) -> str:
    """A one-line description of what the row was actually run at.

    The user-facing rule this serves: a number the *protocol* chose must be quoted with its
    configuration attached, because unlike a swept number there is no single configuration the
    reader can look up in a config file.
    """
    from pathlib import Path

    if row.provenance == "fixed":
        return "single operating point (nothing to tune)"
    if row.key == "phenojury":
        return (f"{PHENOJURY_PROMPT_RULE} / {PHENOJURY_NORMALISER} / all 8 jurors; "
                f"rule, unit and k by inner CV")
    if row.key == "treephenorag":
        return _tree_operating_point(Path(results_dir) / row.exp_id)
    return ""
