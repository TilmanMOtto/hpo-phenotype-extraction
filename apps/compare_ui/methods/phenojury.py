"""PhenoJury: eight models' verbatim generations, and the vote that was taken over them.

This is the only column that can show a model's actual reasoning, because it is the only method
whose generations were kept. ``llm_extractions_{model}.jsonl`` holds, per (model, report, sentence),
the sentence the juror was shown and the text it wrote back -- so a term this column got right can
be traced to the words a particular model produced, and a term it got wrong to the model that
produced those instead.

**The vote is recomputed, not read.** ``phenojury_protocol`` selects ``(rule, unit, k, subset)`` inside nested
cross-validation and writes out the resulting prediction *sets*. The per-term vote counts are built
in memory by ``protocol.max_counts`` and discarded. Recomputing them needs the fold's own choice,
and that is where the subtlety is: the pooled prediction set comes from **repetition 0 only**
(``tree_selection.nested_evaluation`` pools there so every report appears once), so the
configuration that produced a given report's prediction is the one chosen at
``(repetition=0, outer_fold=f)`` for the fold that report was *evaluated* in. Using the modal
configuration instead would be a different method that happens to agree most of the time.

``protocol.pack`` and ``protocol.max_counts`` are imported rather than reimplemented. The unit
semantics are the whole method -- a vote is *k* jurors agreeing **in some one unit**, a maximum and
never a sum, or two jurors in two different sentences manufacture agreement neither expressed -- and
a second copy of that rule would be wrong eventually and silently.

Where the import is not available the column still draws: generations and per-sentence detections
are shown, the recomputed vote count is not, and the panel says which. Degrading loudly beats a
vote count computed by a rule nobody checked.
"""

from __future__ import annotations

import csv
import json
import logging
import os

from apps.compare_ui.methods import base

logger = logging.getLogger(__name__)

#: The eight jurors, in the order every phenojury_generation_free_listing/exp14 surface lists them.
MODEL_ORDER = ("apertus", "deepseek", "intelligent_internet", "openbiollm",
               "llama", "medpsy", "medgemma", "phi4")

#: Only repetition 0 contributes to the pooled prediction set, so only its choices explain it.
POOLED_REPETITION = 0


class PhenoJuryAdapter(base.Adapter):
    """Reads PhenoJury predictions and the jurors' generations, and places each term on the segments its jurors named."""
    key = "phenojury"

    def load(self, report_ids) -> None:
        """Read the PhenoJury predictions and generations of *report_ids*. Sets ``status`` to ``missing`` when absent."""
        paths = self.ctx.paths

        predictions = self.method.predictions_for(self.ctx.cohort)
        if not predictions:
            self.missing("no prediction set is declared for cohort " + self.ctx.cohort.key)
            return
        predictions = os.path.join(paths.output_base, predictions)
        self._predictions = base.load_prediction_sets(predictions)
        if not self._predictions:
            self.missing(predictions)
            return
        protocol_dir = _protocol_dir(paths, self.method, predictions)
        self.ctx.note_input("phenojury_predictions", predictions)

        manifest = _read_json(os.path.join(protocol_dir, "manifest.json")) or {}
        self.prompt = str(manifest.get("selected_prompt") or "q2_sentence_last")
        self.normaliser = str(manifest.get("selected_normaliser") or "phenobert_candidates")
        self.ctx.note_input("phenojury_manifest", os.path.join(protocol_dir, "manifest.json"))

        self._folds = _fold_of_report(paths, protocol_dir, predictions, report_ids)
        self._choices = _selection_choices(os.path.join(protocol_dir, "tables",
                                                        "s5_selection.csv"))
        self.ctx.note_input("phenojury_selection",
                            os.path.join(protocol_dir, "tables", "s5_selection.csv"))

        det_dir, gen_dir = self._cells()
        self._detections = {}
        self._generations = {}
        for model in MODEL_ORDER:
            det = _first_existing(
                os.path.join(det_dir, "detections_{}__{}.jsonl".format(model, self.normaliser)),
                os.path.join(det_dir, "detections_{}.jsonl".format(model)),
                os.path.join(gen_dir, "detections_{}__{}.jsonl".format(model, self.normaliser)),
                os.path.join(gen_dir, "detections_{}.jsonl".format(model)))
            gen = _first_existing(
                os.path.join(gen_dir, "llm_extractions_{}.jsonl".format(model)),
                os.path.join(det_dir, "llm_extractions_{}.jsonl".format(model)))
            if det:
                self._detections[model] = _detections_by_report(det, report_ids)
                self.ctx.note_input("phenojury_detections_" + model, det)
            if gen:
                self._generations[model] = _generations_by_report(gen, report_ids)
                self.ctx.note_input("phenojury_generations_" + model, gen)

        if not self._detections:
            self.note = ("no per-juror detections under {} -- votes cannot be recomputed and only "
                         "the prediction set is shown".format(det_dir))
        elif not self._generations:
            self.note = ("no llm_extractions_*.jsonl under {} -- the votes are shown but not the "
                         "words behind them".format(gen_dir))
        logger.info("phenojury: detections from %s, generations from %s (%d/%d models)",
                    det_dir, gen_dir, len(self._detections), len(MODEL_ORDER))

    def _cells(self):
        """``(detections_dir, generations_dir)`` -- and they are not the same directory.

        This caught us on the first real run. ``phenojury_normalisation`` re-reads the Free Listing generation run's
        generations through four different readers and writes ``detections_{model}__{normaliser}``. It does **not** copy the generations themselves. ``phenojury_generation_other_prompts`` holds
        ``llm_extractions_{model}.jsonl`` -- the juror's verbatim output -- and a
        ``detections_{model}`` that is one particular reader's.

        So the two are resolved independently: the detections from wherever this run's normaliser
        was actually applied, the generations from wherever the model's words were written down.
        Deriving both from one directory silently produced a jury column with votes and no prose,
        which is the half of it that is worth reading.
        """
        paths = self.ctx.paths
        grid = paths.cohort_exp(self.method.extra["grid"], self.prompt)
        full = paths.cohort_exp(self.method.exp_id, self.prompt)
        detections = grid if os.path.isdir(grid) else full
        generations = full if os.path.isdir(full) else grid
        return detections, generations

    def evidence(self, rv, outcomes, gold_rows):
        """``(marks, reasons)`` of one report, with the jurors that voted for each term."""
        rid = rv.report_id
        config = self._config_for(rid)
        per_model = {m: (self._detections.get(m) or {}).get(rid) or {}
                     for m in MODEL_ORDER}
        counts, best_units = _vote(per_model, config, self.ctx.view)

        placed = base.place_by_sentence(rv, {
            code: _first_sentence(best_units.get(code)) for code in outcomes
        })

        reasons = {}
        wanted: set = set()
        for code, outcome in outcomes.items():
            reason = _reason(code, outcome, counts.get(code, 0), config, per_model,
                             best_units.get(code))
            reasons[code] = reason
            wanted.update(tuple(ref) for ref in reason.get("refs") or ())

        self._pending_generations = _collect_generations(self._generations, rid, wanted)
        return self.marks_from(outcomes, placed, gold_rows), reasons

    def block_extras(self) -> dict:
        """Per-report payload shared across this column's reasons.

        Generations are stored once per (model, sentence) at the block level, not inside
        each term's reason. One sentence commonly explains several terms, and eight jurors on a
        long report is the single largest thing this bundle could carry -- duplicating it per term
        is how a 300 KB bundle becomes a 3 MB one.
        """
        out = {"generations": getattr(self, "_pending_generations", {})}
        self._pending_generations = {}
        return out

    def _config_for(self, report_id: str) -> dict:
        """The ``(rule, unit, k, subset)`` the outer fold this report was evaluated in chose."""
        fold = self._folds.get(report_id)
        chosen = self._choices.get(fold) if fold is not None else None
        if chosen is None:
            return {"fold": fold, "known": False, "rule": "exact", "unit": "segment",
                    "k": None, "subset": list(MODEL_ORDER)}
        return dict(chosen, fold=fold, known=True)


# -- the fold a report was scored in -----------------------------------------

def _fold_of_report(paths, protocol_dir: str, predictions: str, report_ids) -> dict:
    """``{report_id: outer_fold}`` at repetition 0, from whichever folds file describes this run.

    The eval rows are the ones with ``inner_fold == -1``: a report is *evaluated* in one
    outer fold per repetition, and it is that fold's selection that produced its prediction.

    Three candidate locations, tried in the order of how tightly each is bound to the prediction
    set being explained -- beside the predictions first, then the protocol run, then (HCY only)
    the curated ground truth the folds were stratified on. The order counts: the folds have to be the
    ones the pooled set was pooled over, and two runs of ``phenojury_protocol`` on two document sets both
    write a file called ``folds_<cohort>.csv``.
    """
    from apps.treephenorag_ui import curated as curated_mod

    cohort = paths.cohort_obj
    names = ["folds_{}.csv".format(cohort.artifact_dir), "folds_{}.csv".format(cohort.key)]
    directories = [os.path.dirname(os.path.dirname(predictions)), protocol_dir]
    if cohort.key == "hcy":
        directories.append(paths.gold_dir or curated_mod.newest(paths.hcy_dir) or "")

    path = _first_existing(*[os.path.join(directory, name)
                             for directory in directories if directory for name in names])
    out: dict = {}
    if not path:
        logger.warning("no folds file beside %s or under %s; PhenoJury's per-fold configuration "
                       "is unavailable", predictions, protocol_dir)
        return out
    wanted = set(report_ids)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if _int(row.get("repetition")) != POOLED_REPETITION:
                continue
            if _int(row.get("inner_fold")) != -1:
                continue
            if str(row.get("role") or "").strip() != "eval":
                continue
            rid = str(row.get("report_id") or "").strip()
            if rid in wanted:
                out[rid] = _int(row.get("outer_fold"))
    return out


def _protocol_dir(paths, method, predictions: str) -> str:
    """The ``phenojury_protocol`` run directory whose selections explain *predictions*.

    The run that **wrote** the prediction set wins, because its ``s5_selection.csv`` is the one
    that produced these predictions. The committed run of the same experiment is the fallback for
    a layout that predates ``comparison_inputs``. Reading a different run's selections would explain
    this column with a configuration it was never at.
    """
    beside = os.path.dirname(os.path.dirname(predictions))
    if os.path.isfile(os.path.join(beside, "manifest.json")):
        return beside
    return paths.cohort_exp(method.extra["protocol"])


def _selection_choices(path: str) -> dict:
    """``{outer_fold: {rule, unit, k, subset}}`` at repetition 0."""
    out: dict = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if _int(row.get("repetition")) != POOLED_REPETITION:
                continue
            fold = _int(row.get("outer_fold"))
            if fold is None:
                continue
            out[fold] = {
                "rule": str(row.get("rule") or "exact"),
                "unit": str(row.get("unit") or "segment"),
                "k": _int(row.get("k")),
                "subset": [m for m in str(row.get("subset") or "").split(";") if m],
            }
    return out


# -- the vote ----------------------------------------------------------------

_PROTOCOL = []          # a one-slot memo: the module, or None, decided once per process


def _protocol():
    """The PhenoJury voting module (:mod:`hpo_extraction.phenojury.vote`), or ``None``.

    Failing to import degrades this column, not the build.
    """
    if _PROTOCOL:
        return _PROTOCOL[0]
    try:
        from hpo_extraction.phenojury import vote as protocol
    except Exception as exc:                                   # pragma: no cover - env-dependent
        logger.warning("the PhenoJury vote is not importable (%s); votes will not be recomputed",
                       exc)
        _PROTOCOL.append(None)
        return None
    _PROTOCOL.append(protocol)
    return protocol


def _vote(per_model: dict, config: dict, view):
    """``(counts, best_units)`` -- votes per term, and which units carried the winning tally.

    ``best_units`` is what lets a vote be drawn on a sentence, not only counted: under the
    segment unit the winning tally happened in one specific segment, and that is where the reader
    should be looking.
    """
    protocol = _protocol()
    if protocol is None or view is None:
        return _fallback_counts(per_model, config), _units_by_code(per_model, config)

    subset = [m for m in config.get("subset") or MODEL_ORDER if m in per_model]
    jurors = [protocol.Juror(model=m, prompt="", normaliser="") for m in subset]
    packed = protocol.pack(
        jurors=jurors,
        per_juror={j: {"r": per_model[j.model]} for j in jurors},
        report_ids=["r"],
        unit=config.get("unit") or "segment",
        rule=config.get("rule") or "exact",
        view=view,
    )
    counts = protocol.max_counts(packed, ["r"], frozenset(range(len(jurors)))).get("r", {})
    return dict(counts), _units_by_code(per_model, config, subset)


def _fallback_counts(per_model: dict, config: dict) -> dict:
    """Distinct jurors naming a term anywhere in the report -- the report unit,.

    Used only when ``protocol`` cannot be imported. It is right for ``unit="report"`` and an upper
    bound otherwise, which is why the panel labels it ``vote_recomputed: false``, not
    printing it as the method's own number.
    """
    subset = set(config.get("subset") or MODEL_ORDER)
    counts: dict = {}
    for model, sentences in per_model.items():
        if model not in subset:
            continue
        named = {code for terms in (sentences or {}).values() for code in terms}
        for code in named:
            counts[code] = counts.get(code, 0) + 1
    return counts


def _units_by_code(per_model: dict, config: dict, subset=None) -> dict:
    """``{hpo_id: {sentence: [model, ...]}}`` -- who named what, where.

    Kept per sentence whatever the unit, because it is the evidence a reader wants regardless of
    how the vote was counted: the vote answers "was it accepted", this answers "who said so".
    """
    allowed = set(subset if subset is not None else (config.get("subset") or MODEL_ORDER))
    out: dict = {}
    for model, sentences in per_model.items():
        if model not in allowed:
            continue
        for sent, terms in (sentences or {}).items():
            for code in terms:
                out.setdefault(code, {}).setdefault(int(sent), []).append(model)
    for where in out.values():
        for models in where.values():
            models.sort(key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
    return out


def _first_sentence(units):
    """The sentence with the most jurors behind it -- earliest wins a tie."""
    if not units:
        return None
    return sorted(units.items(), key=lambda kv: (-len(kv[1]), kv[0]))[0][0]


# -- loading -----------------------------------------------------------------

def _detections_by_report(path: str, report_ids) -> dict:
    """``{report: {sentence: {hpo, ...}}}`` -- the same shape ``protocol.load_detections_file``
    produces, restricted to the reports asked for so the builder does not hold the cohort."""
    out: dict = {}
    for rid, row in base.read_jsonl(path, report_ids):
        code = row.get("hpo_id")
        sent = _int(row.get("sentence_number"))
        if not code or sent is None:
            continue
        out.setdefault(rid, {}).setdefault(sent, set()).add(str(code))
    return out


def _generations_by_report(path: str, report_ids) -> dict:
    out: dict = {}
    for rid, row in base.read_jsonl(path, report_ids):
        sent = _int(row.get("sentence_number"))
        if sent is None:
            continue
        out.setdefault(rid, {})[sent] = row
    return out


def _collect_generations(generations: dict, report_id: str, wanted) -> dict:
    """``{"model|sentence": record}`` for the (model, sentence) pairs some reason referenced."""
    out: dict = {}
    for model, sent in sorted(wanted):
        row = ((generations.get(model) or {}).get(report_id) or {}).get(sent)
        if not row:
            continue
        text, truncated = base.clip(row.get("llm_output"))
        out["{}|{}".format(model, sent)] = {
            "model": model,
            "sentence_number": sent,
            "sentence_text": base.clip(row.get("sentence_text"), 600)[0],
            "llm_output": text,
            "truncated": truncated,
        }
    return out


# -- the reason --------------------------------------------------------------

def _reason(code, outcome, votes, config, per_model, units) -> dict:
    units = units or {}
    subset = [m for m in (config.get("subset") or MODEL_ORDER)]
    voted = {m for models in units.values() for m in models}

    jurors = []
    refs = []
    for model in MODEL_ORDER:
        sentences = sorted(s for s, models in units.items() if model in models)
        jurors.append({
            "model": model,
            "in_subset": model in subset,
            "voted": model in voted,
            "sentences": sentences,
        })
        for sent in sentences:
            refs.append([model, sent])

    k = config.get("k")
    reason = {
        "kind": "jury",
        "source": "llm_extractions_*.jsonl + detections_*.jsonl",
        "votes": int(votes or 0),
        "k": k,
        "rule": config.get("rule"),
        "unit": config.get("unit"),
        "subset": subset,
        "outer_fold": config.get("fold"),
        "config_known": bool(config.get("known")),
        "vote_recomputed": _protocol() is not None,
        "jurors": jurors,
        "refs": refs,
        "by_sentence": {str(s): models for s, models in sorted(units.items())},
    }
    reason["why"] = _why(outcome, votes, k, voted, config)
    return reason


def _why(outcome, votes, k, voted, config) -> str:
    """One sentence naming the vote, the unit it was taken in, and the threshold it met or missed.

    The unit is spelled out, not printed as its key: "in one sentence" is the whole content
    of ``unit="segment"``, and a reader who has to remember what ``segment`` means here is being
    asked to carry the method's vocabulary instead of being told the answer.
    """
    unit = config.get("unit") or "segment"
    where = {"report": "anywhere in the report", "segment": "in one sentence",
             "window": "within a +/-1 sentence window"}.get(unit, "in one " + unit)
    threshold = "no threshold recorded" if k is None else "threshold k={}".format(k)

    if outcome == "fn":
        if not voted:
            return "no juror wrote anything that linked to this term"
        return "named by {} but not accepted: {} agreeing {}, {}".format(
            ", ".join(sorted(voted)), votes, where, threshold)
    if not voted:
        return "predicted, but no juror in the selected subset is recorded naming it"
    return "{} juror{} agreed {} ({}): {}".format(
        votes, "" if votes == 1 else "s", where, threshold, ", ".join(sorted(voted)))


# -- small helpers -----------------------------------------------------------

def _first_existing(*paths):
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return ""


def _read_json(path: str):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None
