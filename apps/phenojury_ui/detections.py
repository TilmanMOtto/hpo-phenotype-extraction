"""The detection tensor and the vote bitmasks, what every number in this app is derived from.

Three tables, all server-side:

**replies**, one row per ``(model, report, sentence)``: the raw generation, the ``strip_think``-ed
text PhenoBERT actually received, and whether the model wrote anything at all.

**pb**, one row per PhenoBERT detection, *including negated ones*: the matched phrase, the
confidence, the negation flag, and the span **inside the reply**. Recovering that span is the
fiddly part and is worth doing. PhenoBERT's offsets index the concatenated per-report input
file that ``phenobert_runner.write_phenobert_input`` builds, not any single reply. That function is
deterministic, so rebuilding it here translates an offset into ``(sentence, span within the reply)``
with no fuzzy matching. Note it is built from the *stripped* replies, highlighting the raw reply
with an offset computed on it would be off by the length of the ``<think>`` block.

**masks**, ``masks[report][hpo] = bitmask``, bit *i* set when model *i* detected that term in that
report. Every aggregation in this app is a popcount over these masks, which is what lets the k
slider be a live control rather than a rerun. They are built from ``detections_{model}.jsonl``,
never from the TSVs: that file is what the driver itself aggregated, so recomputing from anything
else would be scoring a different run than the one that shipped.
"""

from __future__ import annotations

import bisect
import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Popcount and the rest of the mask arithmetic live in ``hpo_extraction.phenojury.ensemble_votes``, shared with
# An earlier exploratory run, which asks what the vote rule costs and needs the identical definition of a voter.
# Re-exported here so this module's callers keep importing it from where they always did. That
# module imports nothing heavier than ``typing``, so it is safe on a login node where the driver's
# torch/mlflow chain is not (see this package's ``votes`` docstring).
from hpo_extraction.phenojury.ensemble_votes import popcount  # noqa: E402,F401


# ``hpo_extraction.phenojury.ensemble_eval`` imports mlflow, so on a login node where that fails the app must not go
# with it, this is the only thing it is needed for here. The vendored copy is asserted equal to
# The real one in ``test_exp13_06_ui``.
_NOTHING_EXACT = {
    "", "no", "none", "na", "n a", "nil", "negative", "no phenotype",
    "no phenotypes", "no symptoms", "no symptom", "no findings",
}
_NOTHING_SUBSTR = ("no phenotype", "no symptom", "no clinical sign", "no abnormal")


def _wrote_something(text: str | None) -> bool:
    """``hpo_extraction.phenojury.ensemble_eval.wrote_something``, vendored for import resilience."""
    import re

    from hpo_extraction.models.verdict import strip_think   # stdlib-only module, always importable

    if not text:
        return False
    cleaned = re.sub(r"[^a-z0-9/ ]", "", strip_think(text).strip().lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned in _NOTHING_EXACT:
        return False
    return not any(pat in cleaned for pat in _NOTHING_SUBSTR)


@lru_cache(maxsize=1)
def _wrote_something_impl():
    """Resolve the implementation once. This is called per reply, a cohort has tens of thousands,
    and re-entering the try/import on each one is pure overhead for an answer that cannot change.
    """
    try:
        from hpo_extraction.phenojury.ensemble_eval import wrote_something as real
    except Exception:  # noqa: BLE001 - the degradation is logged once, by votes._driver
        return _wrote_something
    return real


def wrote_something(text: str | None) -> bool:
    """Prefer the real implementation. Fall back to the vendored copy if mlflow is unavailable."""
    return _wrote_something_impl()(text)


def productivity(built: dict, sent_hpos: dict, models: list[str]) -> list[dict]:
    """``hpo_extraction.phenojury.ensemble_eval.compute_slm_productivity``, read off the replies already built.

    Same numbers, no second pass: ``build_replies`` has already run ``wrote_something`` over every
    generation in the cohort and kept the verdict, so recomputing it means re-running a regex over
    tens of thousands of strings to learn what the bundle is holding. ``test_exp13_06_ui`` asserts
    this equals the driver's function row for row.
    """
    per_model: dict[str, list[dict]] = {m: [] for m in models}
    for (model, report_id, sent_num), reply in built["replies"].items():
        if model in per_model:
            per_model[model].append({"report_id": report_id, "sentence_number": sent_num,
                                     "wrote": reply["wrote"]})

    rows = []
    for model in models:
        replies = per_model[model]
        n_sentences = len(replies)
        sm = sent_hpos.get(model, {})
        n_wrote = n_wrote_and_hpo = 0
        for reply in replies:
            if not reply["wrote"]:
                continue
            n_wrote += 1
            if sm.get(reply["report_id"], {}).get(reply["sentence_number"]):
                n_wrote_and_hpo += 1
        rows.append({
            "model": model,
            "n_sentences": n_sentences,
            "n_wrote_something": n_wrote,
            "n_wrote_and_hpo": n_wrote_and_hpo,
            "pct_of_wrote": round(100.0 * n_wrote_and_hpo / n_wrote, 2) if n_wrote else 0.0,
            "pct_of_all": round(100.0 * n_wrote_and_hpo / n_sentences, 2) if n_sentences else 0.0,
        })
    return rows


def bits(mask: int, models: list[str]) -> list[str]:
    """The models whose bit is set, in ``models`` order."""
    return [m for i, m in enumerate(models) if mask >> i & 1]


def mask_of(subset, models: list[str]) -> int:
    """Bitmask for a set of model names, ignoring names not in *models*."""
    wanted = set(subset)
    return sum(1 << i for i, m in enumerate(models) if m in wanted)


# ── replies, and where PhenoBERT saw them ────────────────────────────────────

def build_replies(records_by_model: dict[str, list[dict]]) -> dict[str, Any]:
    """``{"replies": {(model, report, sent): {...}}, "offsets": {(model, report): (starts, sents)}}``.

    ``starts[i]`` is the character offset at which sentence ``sents[i]``'s reply begins inside the
    concatenated PhenoBERT input. The layout is ``"sentence_number: {n};\\n{llm_output}"`` blocks
    with every line terminated by ``phenobert_runner.terminate_lines``, joined by ``"\\n"``, in
    ascending sentence order, reproduced here, not imported because the real function writes
    to disk. :func:`assert_layout_matches` pins the two together.

    The offsets index ``pb_text``, the *terminated* reply, the exact bytes PhenoBERT was handed, not ``stripped``. Anything drawing PhenoBERT spans must render ``pb_text``; ``stripped`` stays
    the model's own answer.
    """
    from hpo_extraction.phenojury.phenobert import terminate_lines
    from hpo_extraction.models.verdict import strip_think

    replies: dict[tuple[str, str, int], dict] = {}
    offsets: dict[tuple[str, str], tuple[list[int], list[int]]] = {}

    for model, records in records_by_model.items():
        by_report: dict[str, list[dict]] = {}
        for r in records:
            raw = r.get("llm_output", "") or ""
            report_id = str(r.get("patient_id", r.get("report_id", "")))
            sent_num = int(r.get("sentence_number", -1))
            stripped = strip_think(raw)
            replies[(model, report_id, sent_num)] = {
                "raw": raw,
                "stripped": stripped,
                "pb_text": terminate_lines(stripped),
                "sentence_text": r.get("sentence_text", "") or "",
                "wrote": wrote_something(raw),
            }
            by_report.setdefault(report_id, []).append({**r, "patient_id": report_id})

        for report_id, report_records in by_report.items():
            ordered = sorted(report_records, key=lambda r: int(r["sentence_number"]))
            starts: list[int] = []
            sent_nums: list[int] = []
            cursor = 0
            for i, r in enumerate(ordered):
                sent_num = int(r["sentence_number"])
                pb_text = replies[(model, report_id, sent_num)]["pb_text"]
                header = f"sentence_number: {sent_num};\n"
                starts.append(cursor + len(header))
                sent_nums.append(sent_num)
                cursor += len(header) + len(pb_text) + (1 if i < len(ordered) - 1 else 0)
            offsets[(model, report_id)] = (starts, sent_nums)

    return {"replies": replies, "offsets": offsets}


def locate(offsets: dict, model: str, report_id: str, start: int) -> tuple[int | None, int | None]:
    """Map a PhenoBERT character offset to ``(sentence_number, offset within that reply)``."""
    entry = offsets.get((model, report_id))
    if not entry:
        return None, None
    starts, sent_nums = entry
    i = bisect.bisect_right(starts, start) - 1
    if i < 0:
        return None, None
    return sent_nums[i], start - starts[i]


def assert_layout_matches(records_by_model: dict[str, list[dict]], model: str, report_id: str,
                          built: dict) -> None:
    """Fix the offset arithmetic to ``write_phenobert_input`` itself.

    Rebuilds the input file the real function would have written and checks that every offset this
    module computed lands on the start of the corresponding reply. If the input layout ever
    changes, this fails loudly instead of silently mis-highlighting every span in the app.
    """
    from hpo_extraction.phenojury.phenobert import build_phenobert_input
    from hpo_extraction.models.verdict import strip_think

    records = [
        {**r, "patient_id": str(r.get("patient_id", r.get("report_id", ""))),
         "llm_output": strip_think(r.get("llm_output", "") or "")}
        for r in records_by_model.get(model, [])
        if str(r.get("patient_id", r.get("report_id", ""))) == report_id
    ]
    truth, _spans = build_phenobert_input(records, report_id)

    starts, sent_nums = built["offsets"][(model, report_id)]
    for start, sent_num in zip(starts, sent_nums):
        stripped = built["replies"][(model, report_id, sent_num)]["pb_text"]
        if truth[start:start + len(stripped)] != stripped:
            raise AssertionError(
                f"offset {start} for {model}/{report_id}/sentence {sent_num} does not land on the "
                "reply, the PhenoBERT input layout has changed"
            )


# ── the PhenoBERT table ──────────────────────────────────────────────────────

def build_pb_table(pb_by_model: dict[str, list[dict]], built: dict) -> list[dict]:
    """Flatten the per-model PhenoBERT rows, resolving sentence number and in-reply span.

    Sentence attribution takes the TSV's own ``sentence_count`` column when the install wrote one,
    and the offsets otherwise, the same two routes, in the same order, as
    ``phenobert_runner.parse_phenobert_sentences``. A row that neither route can place keeps
    ``sentence_number = None`` and is reported, not dropped: silently losing it is how the
    the Free Listing generation run "N sentences → 0 detections" failure stayed invisible for a whole cohort.
    """
    rows: list[dict] = []
    for model, entries in pb_by_model.items():
        for e in entries:
            report_id = str(e["patient_id"])
            sent_num = e.get("sentence_number")
            in_reply = None
            located, offset = locate(built["offsets"], model, report_id, int(e["start"]))
            if sent_num is None:
                sent_num = located
                in_reply = offset
            elif located == sent_num:
                in_reply = offset
            rows.append({
                "model": model, "report_id": report_id, "sentence_number": sent_num,
                "hpo_id": e["hpo_id"], "phrase": e["phrase"], "confidence": e["confidence"],
                "negated": bool(e["negated"]), "start": e["start"], "end": e["end"],
                "offset_in_reply": in_reply,
            })
    return rows


def index_pb(pb_rows: list[dict]) -> dict:
    """``{(report, hpo): [row]}`` and ``{(model, report, sent): [row]}`` for the drill-downs."""
    by_term: dict[tuple[str, str], list[dict]] = {}
    by_sentence: dict[tuple[str, str, int], list[dict]] = {}
    for row in pb_rows:
        by_term.setdefault((row["report_id"], row["hpo_id"]), []).append(row)
        if row["sentence_number"] is not None:
            key = (row["model"], row["report_id"], int(row["sentence_number"]))
            by_sentence.setdefault(key, []).append(row)
    return {"by_term": by_term, "by_sentence": by_sentence}


# ── the vote masks ───────────────────────────────────────────────────────────

def sent_hpos_from_detections(detections_by_model: dict[str, list[dict]]) -> dict:
    """``{model: {report: {sentence: {hpo: count}}}}``, the inverse of ``_write_detections``.

    Mirrors ``slm_ensemble_experiment._read_detections``, including the last-write-wins
    behaviour on a duplicated (report, sentence, hpo) key, so the app aggregates the same tensor
    the driver did.
    """
    out: dict[str, dict] = {}
    for model, rows in detections_by_model.items():
        per_report: dict[str, dict[int, dict[str, int]]] = {}
        for rec in rows:
            report_id = str(rec["report_id"])
            sent_num = int(rec["sentence_number"])
            per_report.setdefault(report_id, {}).setdefault(sent_num, {})
            per_report[report_id][sent_num][rec["hpo_id"]] = int(rec["count"])
        out[model] = per_report
    return out


def report_hpo_counts(sent_hpos: dict) -> dict:
    """``{model: {report: {hpo: count}}}``, counts summed over sentences.

    Same collapse as ``hpo_extraction.phenojury.ensemble_eval.derive_patient_hpos``. Kept here so the mask builder does
    not need the driver's import chain, and differentially tested against it.
    """
    out: dict[str, dict] = {}
    for model, reports in sent_hpos.items():
        out[model] = {}
        for report_id, sentences in reports.items():
            agg: dict[str, int] = {}
            for hpos in sentences.values():
                for hpo_id, count in hpos.items():
                    agg[hpo_id] = agg.get(hpo_id, 0) + count
            out[model][report_id] = agg
    return out


def build_masks(report_hpos: dict, models: list[str], report_ids: list[str],
                min_count: int = 1) -> dict:
    """``{report: {hpo: mask}}`` over *models*, applying the detection-count floor.

    Only models in *models* get a bit, and bit *i* is ``models[i]``, so a mask is only ever
    meaningful alongside the model list it was built with. Terms nobody detected are absent, not present with mask 0, which keeps the term space proportional to the evidence.
    """
    masks: dict[str, dict[str, int]] = {rid: {} for rid in report_ids}
    for i, model in enumerate(models):
        bit = 1 << i
        per_report = report_hpos.get(model, {})
        for report_id in report_ids:
            row = masks[report_id]
            for hpo_id, count in per_report.get(report_id, {}).items():
                if count >= min_count:
                    row[hpo_id] = row.get(hpo_id, 0) | bit
    return masks


def sentence_masks(sent_hpos: dict, models: list[str], report_ids: list[str],
                   min_count: int = 1) -> dict:
    """``{report: {sentence: {hpo: mask}}}``, the per-sentence view the plurality rule votes on."""
    out: dict[str, dict[int, dict[str, int]]] = {rid: {} for rid in report_ids}
    for i, model in enumerate(models):
        bit = 1 << i
        for report_id in report_ids:
            for sent_num, hpos in sent_hpos.get(model, {}).get(report_id, {}).items():
                bucket = out[report_id].setdefault(int(sent_num), {})
                for hpo_id, count in hpos.items():
                    if count >= min_count:
                        bucket[hpo_id] = bucket.get(hpo_id, 0) | bit
    return out


def sentences_from_replies(built: dict, report_ids: list[str]) -> dict:
    """``{report: {sentence_number: text}}`` recovered from the extractions.

    Every model saw the same Stanza segmentation, so any model's ``sentence_text`` answers this.
    It is why the app needs neither the raw report files nor a Stanza install: the segmentation is
    already persisted, once per model, in ``llm_extractions_{model}.jsonl``.
    """
    out: dict[str, dict[int, str]] = {rid: {} for rid in report_ids}
    for (_model, report_id, sent_num), reply in built["replies"].items():
        if report_id in out and sent_num not in out[report_id] and reply["sentence_text"]:
            out[report_id][sent_num] = reply["sentence_text"]
    return out
