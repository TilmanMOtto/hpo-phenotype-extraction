"""Pure IO for the Free Listing generation run artifact set. No caching, no Dash, no ontology.

Every function here returns ``None`` / ``[]`` / ``{}`` rather than raising, so the layer above can
branch on what a run actually has instead of guarding each call. A half-finished run, six of eight
models extracted, no aggregate stage yet, must open in the app and say so, not crash it.

The contract, all of it written by ``src/hpo_extraction/phenojury/generation.py`` into
``{output_base}/phenojury_generation_free_listing/{cohort}/``:

======================================  =========================================================
file                                    schema
======================================  =========================================================
``llm_extractions_{model}.jsonl``       ``model, patient_id, sentence_number, sentence_text,
                                        llm_output``, the raw generation, ``<think>`` *not*
                                        stripped (``:265-279``)
``detections_{model}.jsonl``            ``report_id, model, sentence_number, hpo_id, count``
                                        (``_write_detections :298-314``), negated detections are
                                        already gone at this point
``phenobert_output_{model}/{rid}.txt``  PhenoBERT TSV: ``start, end, phrase, HPO_ID, confidence``
                                        + optional ``sentence_count, source`` + optional ``Neg``.
                                        The only place negation and the matched phrase survive
``slm_ensemble_detections.jsonl``       the active models' slices concatenated (``:603-612``)
``{rule}/slm_ensemble_predictions.jsonl``  per-term lines then one ``summary: true`` line carrying
                                        ``predicted_set`` **and ``gold_set``** (``_write_rule
                                        :406-427``), which is why this app needs no ground-truth
                                        CSV, no target-symptom list and no dataset loader
``slm_ensemble_agg_summary.csv``        one row per rule (``:645``)
``slm_ensemble_slm_metrics.csv``        one row per model (``:651``)
``slm_ensemble_timing*.jsonl``          stage timings (``:515``, ``:656``)
======================================  =========================================================

The earlier prompt experiments (:data:`EXP_IDS`) write *the same files* one level deeper, under
``{output_base}/{exp_id}/{cohort}/{prompt_key}/``, because ``hpo_extraction.phenojury.generation_prompts`` calls
the Free Listing generation run's own ``_write_rule`` and ``write_slm_metrics``. Three deltas, all handled here:

======================================  =========================================================
``slm_metrics.csv``                     the per-model metrics under a second name, same writer,
                                        same columns (:data:`SLM_METRICS_FILES`)
``slm_ensemble_timing*.jsonl``          absent. Earlier logs timing to MLflow only, so
                                        ``load_timing`` returns its empty shape
``prompt_diagnostics.csv``              extra: echo rate, label exactness, format compliance
``prompt_screen_ranking.csv``           extra, one level *up*: where this prompt placed
======================================  =========================================================

stdlib-only (plus a lazy pandas for the two CSVs): importing the driver pulls in
torch and mlflow, and the modules that genuinely need the driver's own scoring functions
(:mod:`votes`, :mod:`verify`) import them there, where the cost buys something.
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

#: The experiment directories this app understands, in dropdown order.
#:
#: ``phenojury_generation_free_listing`` writes its per-model artifacts straight into ``<cohort>/``. The earlier runs
#: prompt experiments run the *same* driver stages through :mod:`hpo_extraction.phenojury.generation_prompts`, which writes
#: The Free Listing generation run's own artifact names one level deeper, into ``<cohort>/<prompt_key>/``,
#: so a promoted prompt enters the comparison table with no reader changes beyond the extra path
#: segment (``core/prompt_screen.py:428-431``). That makes a prompt cell a Free Listing generation run directory
#: in every respect this app cares about, so it is read by the same code, not a parallel one.
EXP_IDS: tuple[str, ...] = (
    "phenojury_generation_free_listing",
    "exp14_00_prompt_screen",
    "phenojury_generation_other_prompts",
    "exp14_02_curated_prompt_screen",
)

#: Kept as a name because the module docstring, the tests and ``registry.validate`` all speak of
#: "the Free Listing generation run contract", this is the experiment that defines it.
EXP_ID = EXP_IDS[0]

#: Ensemble membership, in ``slm_ensemble_experiment.MODEL_KEYS`` order. Copied, not
#: imported to keep this module free of the torch/mlflow import chain; ``test_exp13_06_ui`` pins
#: The two together, so a change upstream fails a test, not silently reordering the UI.
MODEL_ORDER: tuple[str, ...] = (
    "apertus", "deepseek", "intelligent_internet", "openbiollm",
    "llama", "medpsy", "medgemma", "phi4",
)

_EXTRACT_PREFIX = "llm_extractions_"
_DETECT_PREFIX = "detections_"
_PB_PREFIX = "phenobert_output_"
_PREDICTIONS = "slm_ensemble_predictions.jsonl"

#: PhenoBERT TSV column positions. Columns 5+ are install-dependent, so they are read by content
#: (``isdigit`` for the sentence index, ``"neg"`` for the negation flag), not by position.
_COL_START, _COL_END, _COL_PHRASE, _COL_HPO, _COL_CONF = 0, 1, 2, 3, 4

_RULE_RE = re.compile(r"^(vote_k(\d+)|agg_plurality)$")


# ──────────────────────────────────────────────────────────────────────────────
# discovery
# ──────────────────────────────────────────────────────────────────────────────
def is_run_dir(path: str) -> bool:
    """True if *path* holds at least one per-model artifact, the weakest usable signal.

    Extractions alone qualify: a cohort whose array tasks are still running has no detections yet,
    and refusing to open it would make the app useless when it is most wanted.
    """
    if not os.path.isdir(path):
        return False
    try:
        names = os.listdir(path)
    except OSError:
        return False
    return any(
        n.startswith((_EXTRACT_PREFIX, _DETECT_PREFIX, _PB_PREFIX)) for n in names
    )


def find_runs(output_base: str) -> list[dict]:
    """``[{run_id, run_dir, cohort, exp_id, prompt_key, models}]`` for every run under *output_base*.

    Four layouts are accepted, so the same flag works whether it points at the shared output root,
    at an experiment directory, or at one cohort:

    * ``<base>/phenojury_generation_free_listing/<cohort>/``, the canonical cluster layout;
    * ``<base>/phenojury_generation_other_prompts/<cohort>/<prompt_key>/``, the same artifacts one level deeper,
      one directory per prompt (:data:`EXP_IDS`);
    * ``<base>/<cohort>/`` (or ``<base>/<cohort>/<prompt_key>/``), *base* is already an
      experiment directory;
    * ``<base>/``, *base* is a single run directory.

    Every :data:`EXP_IDS` entry present is scanned and the results **accumulate**, so one dropdown
    holds the ensemble and each promoted prompt beside it, which is the comparison the app exists
    to make. Only the two fallback layouts stop at the first that matches, because there *base* names
    one thing and guessing further would invent cohorts.

    ``cohort`` stays the dataset (``hcy`` / ``gsc``) even for a prompt cell, and never becomes the
    prompt key: it is what ``pbstandalone.find_run`` resolves the PhenoBERT baseline baseline by, so a cell
    labelled with its prompt would go looking for ``baseline_phenobert/q4_span_json/``.

    ``run_id`` is the bare cohort for :data:`EXP_ID` and carries its experiment for everything else,
    so the ids this app has always used keep working while the earlier cells stay distinguishable.
    """
    runs: list[dict] = []
    if not output_base or not os.path.isdir(output_base):
        return runs

    def add(run_dir: str, run_id: str, cohort: str, exp_id: str = "",
            prompt_key: str = "") -> None:
        runs.append({
            "run_id": run_id, "run_dir": run_dir, "cohort": cohort,
            "exp_id": exp_id, "prompt_key": prompt_key,
            "models": discover_models(run_dir),
        })

    def scan_experiment(exp_dir: str, exp_id: str, prefix: str = "") -> None:
        """Add every cohort under *exp_dir*, descending one level into prompt cells."""
        if not os.path.isdir(exp_dir):
            return
        try:
            names = sorted(os.listdir(exp_dir))
        except OSError:
            return
        for cohort in names:
            child = os.path.join(exp_dir, cohort)
            if not os.path.isdir(child):
                continue
            if is_run_dir(child):
                add(child, f"{prefix}{cohort}", cohort, exp_id)
                continue
            # Not a run directory itself, it may still be a cohort holding one cell per prompt.
            try:
                cells = sorted(os.listdir(child))
            except OSError:
                continue
            for prompt_key in cells:
                cell = os.path.join(child, prompt_key)
                if is_run_dir(cell):
                    add(cell, f"{prefix}{cohort}/{prompt_key}", cohort, exp_id, prompt_key)

    for exp_id in EXP_IDS:
        # The canonical experiment keeps the bare cohort as its run id. That is a published
        # contract, ``drivers.py --cohort hcy``, the disk-cache filenames and the README all use
        # it, and prefixing it would rename every cached report to buy nothing: the earlier ids
        # already carry their experiment and prompt, so they cannot collide with it.
        prefix = "" if exp_id == EXP_ID else f"{exp_id}/"
        scan_experiment(os.path.join(output_base, exp_id), exp_id, prefix=prefix)
    if runs:
        return runs

    # *base* is already an experiment directory: label with its own name so two sources opened in
    # one session are still told apart.
    scan_experiment(output_base, os.path.basename(os.path.normpath(output_base)),
                    prefix=f"{os.path.basename(os.path.normpath(output_base))}/")
    if runs:
        return runs

    if is_run_dir(output_base):
        cohort = os.path.basename(os.path.normpath(output_base))
        add(output_base, cohort, cohort)
    return runs


def discover_models(run_dir: str) -> dict:
    """Which models got how far, in ``MODEL_ORDER``.

    ``extracted``
        wrote an ``llm_extractions_{model}.jsonl``.
    ``linked``
        has a non-empty ``phenobert_output_{model}/``.
    ``active``
        has a **non-empty** ``detections_{model}.jsonl``. This is the driver's own membership test
        (``slm_ensemble_experiment.py:576-592``): an empty detections file is a grounding failure,
        not a model that found nothing, and counting it as a voter would let a broken PhenoBERT
        pass as eight legitimate abstentions. Only these models vote.
    """
    out = {"extracted": [], "linked": [], "active": []}
    if not os.path.isdir(run_dir):
        return out
    try:
        names = set(os.listdir(run_dir))
    except OSError:
        return out

    def known(prefix: str, suffix: str) -> set[str]:
        return {n[len(prefix):-len(suffix)] for n in names
                if n.startswith(prefix) and n.endswith(suffix)}

    extracted = known(_EXTRACT_PREFIX, ".jsonl")
    detected = known(_DETECT_PREFIX, ".jsonl")
    linked = {n[len(_PB_PREFIX):] for n in names if n.startswith(_PB_PREFIX)
              and os.path.isdir(os.path.join(run_dir, n))
              and os.listdir(os.path.join(run_dir, n))}

    for model in _ordered(extracted | detected | linked):
        if model in extracted:
            out["extracted"].append(model)
        if model in linked:
            out["linked"].append(model)
        if model in detected and _size(os.path.join(run_dir, f"{_DETECT_PREFIX}{model}.jsonl")) > 0:
            out["active"].append(model)
    return out


def _ordered(models) -> list[str]:
    """``MODEL_ORDER`` first, then anything unrecognised alphabetically, never dropped."""
    known = [m for m in MODEL_ORDER if m in models]
    return known + sorted(set(models) - set(MODEL_ORDER))


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def find_rules(run_dir: str) -> list[str]:
    """Rule directories present, ``vote_k1 … vote_kN`` then ``agg_plurality``.

    Sorted numerically: a lexical sort puts ``vote_k10`` between ``vote_k1`` and ``vote_k2``, which
    would silently break the monotonicity check and the sweep curve on a >9-model ensemble.
    """
    if not os.path.isdir(run_dir):
        return []
    found = []
    for name in os.listdir(run_dir):
        m = _RULE_RE.match(name)
        if m and os.path.isfile(os.path.join(run_dir, name, _PREDICTIONS)):
            found.append((int(m.group(2)) if m.group(2) else 10**6, name))
    return [name for _, name in sorted(found)]


def artifacts_mtime(run_dir: str) -> float:
    """Newest mtime over the artifacts, used as the cache key.

    ``run.log`` is excluded on purpose: it is appended to for as long as a SLURM array is alive, so
    including it would invalidate the cache on every poll of a run that is still going.
    """
    newest = 0.0
    for root, dirs, files in os.walk(run_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.startswith("run") and name.endswith(".log"):
                continue
            if name.startswith(".checkpoint"):
                continue
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                continue
    return newest


# ──────────────────────────────────────────────────────────────────────────────
# per-model artifacts
# ──────────────────────────────────────────────────────────────────────────────
def _read_jsonl(path: str, what: str) -> list[dict]:
    """Every parseable line of a JSONL file. Torn lines are counted and skipped.

    A torn line is not necessarily the last one, ``slm_ensemble_experiment._load_cached_records``
    documents a resubmitted job appending valid records *after* the damage, so parsing stops at
    no line, it only skips.
    """
    records: list[dict] = []
    if not os.path.isfile(path):
        return records
    n_bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                n_bad += 1
    if n_bad:
        logger.warning("%s: skipped %d unparseable %s line(s)", os.path.basename(path), n_bad, what)
    return records


def load_extractions(run_dir: str, model: str) -> list[dict]:
    """One model's raw generations. ``llm_output`` still carries any ``<think>`` block."""
    return _read_jsonl(os.path.join(run_dir, f"{_EXTRACT_PREFIX}{model}.jsonl"), "extraction")


def load_detections(run_dir: str, model: str) -> list[dict]:
    """One model's grounded detections, negation already applied by the driver."""
    return _read_jsonl(os.path.join(run_dir, f"{_DETECT_PREFIX}{model}.jsonl"), "detection")


def load_merged_detections(run_dir: str) -> list[dict]:
    """The aggregate stage's re-run dump. Only present once ``stage=aggregate`` has run."""
    return _read_jsonl(os.path.join(run_dir, "slm_ensemble_detections.jsonl"), "detection")


def load_phenobert(run_dir: str, model: str) -> list[dict]:
    """One row per PhenoBERT detection, **including the negated ones**.

    ``detections_{model}.jsonl`` is the post-negation view, so it cannot answer "was this annotated term
    mentioned but negated?", the difference between a model that never saw a finding and one that
    saw it and said it was absent. That distinction is a whole fate in the recall autopsy, and this
    is the only artifact that carries it, along with the matched phrase and its character offsets.

    Rows: ``patient_id, sentence_number|None, hpo_id, phrase, confidence, negated, start, end``.
    """
    pb_dir = os.path.join(run_dir, f"{_PB_PREFIX}{model}")
    rows: list[dict] = []
    if not os.path.isdir(pb_dir):
        return rows
    for fname in sorted(os.listdir(pb_dir)):
        fpath = os.path.join(pb_dir, fname)
        if not os.path.isfile(fpath):
            continue
        patient_id = os.path.splitext(fname)[0]
        with open(fpath, "r", encoding="utf-8") as f:
            raw = f.read()
        for start, end, phrase, hpo_id, after in _iter_rows(raw):
            negated = bool(after) and after[-1].strip().lower() == "neg"
            sent_num = None
            if len(after) >= 2 and after[1].strip().isdigit():
                sent_num = int(after[1].strip())
            rows.append({
                "patient_id": patient_id, "sentence_number": sent_num, "hpo_id": hpo_id,
                "phrase": phrase, "confidence": _float(after[0] if after else ""),
                "negated": negated, "start": start, "end": end,
            })
    return rows


def _iter_rows(raw: str):
    """``iter_phenobert_rows`` without the driver's import chain.

    Kept byte-identical in behaviour, a matched phrase may contain newlines (models answer with
    numbered lists), so physical lines are folded back into records before parsing, otherwise every
    multi-line detection is silently lost. ``test_exp13_06_ui`` differentially tests this against
    ``hpo_extraction.phenojury.phenobert.iter_phenobert_rows``.
    """
    records: list[str] = []
    pending = ""
    for line in raw.split("\n"):
        if _ROW_RE.match(line):
            if pending:
                records.append(pending)
            pending = line
        elif pending:
            pending += "\n" + line
    if pending:
        records.append(pending)

    for rec in records:
        m = _ROW_RE.match(rec)
        if not m:
            continue
        fields = m.group(3).split("\t")
        hpo = next((f for f in fields if f.strip().startswith("HP:")), None)
        if hpo is None:
            continue
        idx = fields.index(hpo)
        yield int(m.group(1)), int(m.group(2)), "\t".join(fields[:idx]), hpo.strip(), fields[idx + 1:]


_ROW_RE = re.compile(r"^(\d+)\t(\d+)\t(.*)$", re.S)


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# aggregate-stage artifacts
# ──────────────────────────────────────────────────────────────────────────────
def load_predictions(run_dir: str, rule: str) -> dict:
    """One rule's predictions, split into the two things it carries.

    Returns ``{"terms": [...], "sets": {report_id: {"predicted": [...], "gold": [...]}}}``.

    The summary line is the app's ground truth: earlier scores open-set, so the ground-truth set is not
    recoverable from any per-term line (an annotated term nobody predicted has no line at all). Reports
    are kept in file order, which is the driver's ``report_ids`` order.
    """
    path = os.path.join(run_dir, rule, _PREDICTIONS)
    terms: list[dict] = []
    sets: dict[str, dict] = {}
    for rec in _read_jsonl(path, "prediction"):
        if rec.get("summary"):
            sets[str(rec["report_id"])] = {
                "predicted": list(rec.get("predicted_set") or []),
                "gold": list(rec.get("gold_set") or []),
            }
        else:
            terms.append(rec)
    return {"terms": terms, "sets": sets}


def load_gold(run_dir: str, rules: list[str] | None = None) -> dict:
    """``{report_id: [gold term]}``, read from whichever rule directory is available.

    Ground truth is a property of the cohort, not of the rule, so any rule's summary lines answer it. The
    rules are tried in order and the first one that parses wins; ``verify.gate_gold_consistency``
    is what checks they all agree.
    """
    for rule in (rules if rules is not None else find_rules(run_dir)):
        sets = load_predictions(run_dir, rule)["sets"]
        if sets:
            return {rid: list(v["gold"]) for rid, v in sets.items()}
    return {}


def load_agg_summary(run_dir: str):
    """``slm_ensemble_agg_summary.csv`` as a DataFrame, or ``None``."""
    return _read_csv(os.path.join(run_dir, "slm_ensemble_agg_summary.csv"))


#: The per-model metrics file, under both names it is written with. The Free Listing generation run writes the first,
#: :mod:`hpo_extraction.phenojury.generation_prompts` the second, but both come from ``ensemble_eval.write_slm_metrics``,
#: so the *columns are identical* and only the filename differs. Gate G3 therefore needs no
#: special case. It compares the same columns either way.
SLM_METRICS_FILES = ("slm_ensemble_slm_metrics.csv", "slm_metrics.csv")


def slm_metrics_path(run_dir: str) -> str | None:
    """Whichever per-model metrics file this run wrote, or ``None``."""
    for name in SLM_METRICS_FILES:
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    return None


def load_slm_metrics(run_dir: str):
    """The per-model metrics as a DataFrame, or ``None``. See :data:`SLM_METRICS_FILES`."""
    path = slm_metrics_path(run_dir)
    return _read_csv(path) if path else None


def load_prompt_diagnostics(run_dir: str):
    """``prompt_diagnostics.csv`` as a DataFrame, or ``None``, earlier only.

    One row per model: ``echo_rate``, ``label_exactness``, ``negation_trap_rate``,
    ``format_compliance``. ``label_exactness`` is **empty, not 0.0**, for a prompt whose output is a
    sentence, not a label list. Rendering it as a zero would report a real failure where the
    metric simply does not apply.
    """
    return _read_csv(os.path.join(run_dir, "prompt_diagnostics.csv"))


def load_prompt_ranking(run_dir: str):
    """``prompt_screen_ranking.csv`` as a DataFrame, or ``None``, earlier only.

    It sits one level **above** a prompt cell, in the cohort directory, because it ranks the
    prompts against each other. Read from there so a cell can show where it placed.
    """
    parent = os.path.dirname(os.path.normpath(run_dir))
    for candidate in (run_dir, parent):
        frame = _read_csv(os.path.join(candidate, "prompt_screen_ranking.csv"))
        if frame is not None:
            return frame
    return None


def _read_csv(path: str):
    if not os.path.isfile(path):
        return None
    try:
        import pandas as pd

        return pd.read_csv(path)
    except Exception as exc:  # a truncated CSV must not take the app down
        logger.warning("could not read %s: %s", path, exc)
        return None


def load_timing(run_dir: str) -> dict:
    """``{"aggregate": {...} | None, "per_model": {model: [record]}}``.

    The per-model files hold an ``extract`` summary line plus one line per PhenoBERT pass. Both are
    kept, because "the model was cached" and "the model actually ran" are different costs.
    """
    out: dict = {"aggregate": None, "per_model": {}}
    agg = _read_jsonl(os.path.join(run_dir, "slm_ensemble_timing.jsonl"), "timing")
    if agg:
        out["aggregate"] = agg[0]
    if os.path.isdir(run_dir):
        prefix = "slm_ensemble_timing_"
        for name in sorted(os.listdir(run_dir)):
            if name.startswith(prefix) and name.endswith(".jsonl"):
                model = name[len(prefix):-len(".jsonl")]
                out["per_model"][model] = _read_jsonl(os.path.join(run_dir, name), "timing")
    return out


def load_run_log_tail(run_dir: str, n_lines: int = 40) -> str:
    """The last lines of the run log, the fastest way to see why a model is missing.

    The Free Listing generation run has one ``run.log`` per cohort; :mod:`hpo_extraction.phenojury.generation_prompts` writes
    ``run_{prompt_key}_{model}.log`` per array cell instead, so eight tasks stop
    interleaving into one file. Fall back to the most recently written ``run*.log`` so the panel
    keeps answering the question it exists for.
    """
    path = os.path.join(run_dir, "run.log")
    if not os.path.isfile(path):
        candidates = []
        if os.path.isdir(run_dir):
            try:
                candidates = [os.path.join(run_dir, n) for n in os.listdir(run_dir)
                              if n.startswith("run") and n.endswith(".log")]
            except OSError:
                candidates = []
        if not candidates:
            return ""
        path = max(candidates, key=_mtime)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n_lines:])
    except OSError:
        return ""


def load_raw(run_dir: str) -> dict:
    """Everything on disk for one cohort, in one dict. The only function the registry calls."""
    models = discover_models(run_dir)
    rules = find_rules(run_dir)
    read = models["extracted"] or models["active"]
    return {
        "run_dir": run_dir,
        "models": models,
        "rules": rules,
        "records": {m: load_extractions(run_dir, m) for m in read},
        "detections": {m: load_detections(run_dir, m) for m in models["active"]},
        "phenobert": {m: load_phenobert(run_dir, m) for m in models["linked"]},
        "predictions": {r: load_predictions(run_dir, r) for r in rules},
        "agg_summary": load_agg_summary(run_dir),
        "slm_metrics": load_slm_metrics(run_dir),
        "timing": load_timing(run_dir),
        # earlier only; ``None`` for a Free Listing generation run cohort, which every reader of these already handles.
        "prompt_diagnostics": load_prompt_diagnostics(run_dir),
        "prompt_ranking": load_prompt_ranking(run_dir),
    }
