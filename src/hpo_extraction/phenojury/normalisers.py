"""One interface over the ways a generated string becomes an HPO identifier.

Every ensemble number in this project is the composition of two independent choices, what the SLM
was *asked* to write (the prompt) and what reads what it wrote (the normaliser), and the pipeline
has only ever varied the first. This module makes the second a swept axis, so
``phenojury_protocol`` can evaluate the ``(prompt, normaliser)`` **pair** rather than
picking a prompt under one reader and hoping the choice transfers.

Four readers, all returning ``{patient_id: {sentence_number: {hpo_id: count}}}``, the shape
:func:`hpo_extraction.phenojury.ensemble_eval.run_phenobert_per_model` and :func:`hpo_extraction.baselines.autopcr_link.link_records` already
share, so ``_write_detections``, ``derive_patient_hpos``, ``vote_k_sets`` and every result-table library scorer
consume the output unchanged:

``phenobert_raw``
    What the shipped pipeline did. PhenoBERT over the **whole generation**, already on disk as
    ``detections_{model}.jsonl``. Nothing re-runs.

``phenobert_candidates``
    PhenoBERT over the **parsed candidate strings**, one per line. This column is not a refinement,
    it is what makes the grid legible: ``build_phenobert_input`` feeds
    ``terminate_lines(llm_output)`` verbatim, so on ``q4_span_json`` the cached reader sees JSON
    braces and field names while the dictionary and SapBERT readers see the parsed ``term`` field.
    Comparing those two directly measures the parser, not the normaliser.

``dictionary``
    :func:`hpo_extraction.baselines.autopcr_link.link_records`, the earlier runs'lexical linker, unchanged. Loses 0.111-0.149 µF1
    to PhenoBERT on GSC+ and wins precision decisively (0.936 against 0.826), which is the
    profile that makes a prompt x normaliser interaction plausible, not a formality.

``sapbert``
    Nearest label string under a biomedical encoder, over :func:`surface_index.build_surface_index`'s
    40 335 strings. The acceptance threshold is **fixed**, never swept: a normaliser that arrives
    carrying a selected parameter the other three do not have would win the grid on its tuning.

**No model is imported here.** The SapBERT encoder is injected as ``encode_batch``, as
:func:`ontology_index.build_definition_index` takes one, so this module stays importable on a login
node and ``dictionary`` stays pure CPU with no torch anywhere in its path.

**Linkage rows are the artifact that did not exist before.** The cache records *which terms* a
(report, sentence, model) produced and never *which string produced which term*, so "the normaliser
lost it" and "the model never wrote it" were indistinguishable. :data:`LINKAGE_FIELDS` is one schema
across all four readers, and for PhenoBERT it is a **parse of the TSVs already on disk**, not a
re-run. It also recovers what ``detections_*.jsonl`` discards: that file is the post-``Neg`` view,
so a term the model wrote and PhenoBERT then suppressed as negated is invisible in it. ``an earlier exploratory run``
records that collapsing those two splitting GSC+ misses 48 against 199.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, Sequence

from hpo_extraction.baselines.autopcr_link import link_records
from hpo_extraction.phenojury.phenobert import iter_phenobert_rows, terminate_lines
from hpo_extraction.phenojury.prompt_diagnostics import _output_lines, _shape_of, term_candidates

logger = logging.getLogger(__name__)

#: The readers, in the order the grid reports them. ``phenobert_raw`` is first because it is the
#: provenance column, every published ensemble number in this repository is that reader.
NORMALISERS: tuple[str, ...] = (
    "phenobert_raw",
    "phenobert_candidates",
    "dictionary",
    "sapbert",
)

#: The three that make up S2's grid. ``phenobert_raw`` sits outside it: it reads a different input
#: (the whole generation) and is reported beside the grid as the as-shipped reference, never inside
#: it as a fourth competitor.
GRID_NORMALISERS: tuple[str, ...] = ("phenobert_candidates", "dictionary", "sapbert")

#: One row per (candidate string -> identifier) decision, shared by all four readers.
#:
#: ``line_index`` is the index into :func:`prompt_diagnostics._output_lines` of the reply, or ``-1``
#: where the reader cannot attribute to a line, which is ``phenobert_raw``, since it works
#: on character offsets into a text built from the whole reply, not on parsed lines. That
#: ``-1`` is information, not a gap: it says this reader never saw a line.
#:
#: ``negated`` is only ever true for a PhenoBERT reader. The other two are handed candidate strings
#: with no surrounding context and have no negation detector; ``False`` there means "not assessed",
#: which is why :func:`detections_from_linkage` filters on it, not any caller doing so.
LINKAGE_FIELDS: tuple[str, ...] = (
    "report_id",
    "sentence_number",
    "line_index",
    "candidate_string",
    "normaliser",
    "hpo_id",
    "score",
    "source",
    "negated",
)

#: SapBERT's acceptance threshold, fixed before any run. See the module docstring: this is
#: not a swept axis. It is NOT AutoPCR's published value, which an earlier version of
#: this comment claimed: AutoPCR links on retrieval alone at >= 0.95 (tau_1), consults its language
#: model in [0.85, 0.95) (tau_2) and links nothing below 0.85 (third_party/AutoPCR/HPO_evaluation.py). 0.8
#: sits below AutoPCR's lowest threshold on purpose -- this normaliser has no model check behind a
#: match, and the jury's vote does the filtering instead. Changing it moves every SapBERT cell of
#: The PhenoJury protocol, so it is a rerun, not an edit.
SAPBERT_TAU: float = 0.8

#: The encoder the AutoPCR vendor already stages on the cluster. Named here so the surface-form
#: cache and the query encoder cannot drift to two different models.
SAPBERT_ENCODER: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


# ── Candidate strings: the unit three of the four readers operate on ──────────

def candidate_strings(record: dict, shape: str) -> list[tuple[int, str]]:
    """``[(line_index, candidate)]`` for one ``llm_extractions_*.jsonl`` record.

    Reasoning is stripped and list markers removed by :func:`prompt_diagnostics._output_lines`, then
    each line is split into the substrings that are *supposed to be* ontology terms under *shape*, q5's first pipe field, p3's right column, q4's ``term`` values, the whole line otherwise.

    A reply that :func:`ensemble_eval.wrote_something` would reject still yields its candidates here. The sentinel filter belongs to the reader, and ``dictionary`` applies it internally through
    :func:`agent_link.link_record`. Baking it in at two different depths would double-filter.
    """
    out: list[tuple[int, str]] = []
    for line_index, line in enumerate(_output_lines(record.get("llm_output"))):
        for candidate in term_candidates(line, shape):
            cleaned = " ".join((candidate or "").split())
            if cleaned:
                out.append((line_index, cleaned))
    return out


def build_candidate_phenobert_input(
    records: Sequence[dict], patient_id, shape: str
) -> tuple[str, list[tuple[int, int, int]], list[tuple[int, int, int, int]]]:
    """The ``phenobert_candidates`` input for one patient: ``(text, spans, line_spans)``.

    Layout is byte-compatible with :func:`phenobert_runner.build_phenobert_input`, the
    same ``sentence_number: {n};`` marker, the same :func:`terminate_lines` treatment, the same block
    join, so :func:`phenobert_runner.parse_phenobert_sentences` attributes detections to sentences
    over it **unchanged**. Only the body differs: one parsed candidate per line instead of the raw
    reply.

    ``line_spans`` is the extra return this layout can afford and the raw one cannot:
    ``[(start, end, sentence_number, line_index)]`` for every candidate, which is what lets
    :func:`phenobert_linkage` name the string a detection came from, not only its sentence.
    """
    patient_records = sorted(
        (r for r in records if str(r.get("patient_id")) == str(patient_id)),
        key=lambda r: int(r["sentence_number"]),
    )
    blocks: list[str] = []
    spans: list[tuple[int, int, int]] = []
    line_spans: list[tuple[int, int, int, int]] = []
    pos = 0
    for rec in patient_records:
        sent_num = int(rec["sentence_number"])
        marker = terminate_lines(f"sentence_number: {sent_num}")
        candidates = candidate_strings(rec, shape)
        body_lines = [terminate_lines(text) for _, text in candidates]
        block = "\n".join([marker, *body_lines])
        blocks.append(block)
        spans.append((pos, pos + len(block), sent_num))
        # Walk the block to give every candidate its own absolute span. The arithmetic mirrors the
        # join. A drift of one character here would attribute every detection to the
        # previous candidate, silently and plausibly.
        cursor = pos + len(marker) + 1
        for (line_index, _), body in zip(candidates, body_lines):
            line_spans.append((cursor, cursor + len(body), sent_num, line_index))
            cursor += len(body) + 1
        pos += len(block) + 1          # +1 for the "\n" that joins the blocks
    return "\n".join(blocks), spans, line_spans


# ── Reading PhenoBERT's own output back, richer than the shipped parse ────────

def _line_index_at(start: int, line_spans: Iterable[tuple[int, int, int, int]]) -> int:
    for block_start, block_end, _sent, line_index in line_spans:
        if block_start <= start < block_end:
            return line_index
    return -1


def phenobert_linkage(
    output_dir: str,
    inputs: dict,
    *,
    normaliser: str,
    line_spans: dict | None = None,
) -> list[dict]:
    """Every PhenoBERT detection as a :data:`LINKAGE_FIELDS` row, **negated ones included**.

    :func:`phenobert_runner.parse_phenobert_sentences` is the shipped parse and it drops ``Neg``
    rows, which is correct for scoring and destroys the one distinction this analysis needs: a term
    the model wrote and PhenoBERT suppressed is not the same failure as a term nobody wrote. So this
    walks the same TSVs with the same :func:`iter_phenobert_rows` reassembly, PhenoBERT writes the
    matched phrase verbatim, so a phrase containing a newline spans physical lines, and keeps the
    flag as a column instead of as a filter.

    *inputs* is ``{patient_id: (text, spans)}``, as :func:`build_phenobert_input` returns it. The
    sentence number comes from the patched install's ``sentence_count`` column when present and from
    verified character offsets otherwise, mirroring the shipped logic.
    """
    from hpo_extraction.phenojury.phenobert import _sentence_from_offset  # one definition, reused

    rows: list[dict] = []
    if not os.path.isdir(output_dir):
        logger.warning("no PhenoBERT output at %s", output_dir)
        return rows

    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        if not os.path.isfile(fpath):
            continue
        patient_id = os.path.splitext(fname)[0]
        text, spans = (inputs or {}).get(patient_id, ("", []))
        patient_line_spans = (line_spans or {}).get(patient_id, ())
        with open(fpath, "r", encoding="utf-8") as fh:
            raw = fh.read()

        for start, end, phrase, hpo_id, after in iter_phenobert_rows(raw):
            negated = bool(after) and after[-1].strip().lower() == "neg"
            sent_num = None
            if len(after) >= 2 and after[1].strip().isdigit():
                sent_num = int(after[1].strip())
            elif spans:
                sent_num = _sentence_from_offset(start, end, phrase, text, spans)
            if sent_num is None:
                continue
            score = None
            if after and after[0].strip():
                try:
                    score = float(after[0].strip())
                except ValueError:
                    score = None
            rows.append({
                "report_id": patient_id,
                "sentence_number": sent_num,
                "line_index": _line_index_at(start, patient_line_spans),
                "candidate_string": phrase,
                "normaliser": normaliser,
                "hpo_id": hpo_id,
                "score": score,
                "source": "phenobert",
                "negated": negated,
            })
    return rows


def detections_from_linkage(rows: Iterable[dict], *, keep_negated: bool = False) -> dict:
    """``{patient_id: {sentence_number: {hpo_id: count}}}`` from linkage rows.

    ``keep_negated`` defaults to False so the result matches the shipped ``detections_*.jsonl``, that equality is what ``test_normaliser_grid`` pins, and it is the only thing keeping
    ``phenobert_raw`` honest as the provenance column.
    """
    out: dict[str, dict[int, dict[str, int]]] = {}
    for row in rows:
        if row.get("negated") and not keep_negated:
            continue
        hpo_id = row.get("hpo_id")
        if not hpo_id:
            continue
        bucket = out.setdefault(str(row["report_id"]), {}).setdefault(
            int(row["sentence_number"]), {})
        bucket[hpo_id] = bucket.get(hpo_id, 0) + 1
    return out


def load_cached_detections(path: str | Path) -> dict:
    """Read a shipped ``detections_{model}.jsonl`` into the common shape.

    Note the key rename the cache carries: ``llm_extractions_*`` says ``patient_id`` and
    ``detections_*`` says ``report_id`` for the same thing. Both are accepted, because a loader that
    handles only one of them fails on half the artifacts and does so silently.
    """
    out: dict[str, dict[int, dict[str, int]]] = {}
    path = Path(path)
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(rec.get("report_id", rec.get("patient_id", "")))
            try:
                sent = int(rec["sentence_number"])
            except (KeyError, TypeError, ValueError):
                continue
            hpo_id = rec.get("hpo_id")
            if not rid or not hpo_id:
                continue
            bucket = out.setdefault(rid, {}).setdefault(sent, {})
            bucket[hpo_id] = bucket.get(hpo_id, 0) + int(rec.get("count", 1) or 1)
    return out


def write_linkage(rows: Iterable[dict], path: str | Path) -> Path:
    """Write linkage rows as JSONL in :data:`LINKAGE_FIELDS` order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({k: row.get(k) for k in LINKAGE_FIELDS}) + "\n")
    return path


def write_detections(detections: dict, model_key: str, path: str | Path) -> Path:
    """Write the common shape as a ``detections_*.jsonl``-schema file.

    Same five fields and same sort order as
    ``hpo_extraction.phenojury.generation._write_detections``, so anything that reads the shipped artifact
    reads this one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for report_id in sorted(detections):
            for sent_num in sorted(detections[report_id]):
                for hpo_id in sorted(detections[report_id][sent_num]):
                    fh.write(json.dumps({
                        "report_id": report_id,
                        "model": model_key,
                        "sentence_number": int(sent_num),
                        "hpo_id": hpo_id,
                        "count": int(detections[report_id][sent_num][hpo_id]),
                    }) + "\n")
    return path


# ── The SapBERT reader ───────────────────────────────────────────────────────

def surface_corpus(tree) -> tuple[list[str], list[str]]:
    """``(hpo_ids, texts)`` over every label string, deduplicated by :func:`surface_key`.

    Deduplication counts for cost and not for correctness: the release writes 40 335 distinct
    label strings over 18 354 terms, and embedding the same string twice would buy nothing. Where a
    key collides across terms, the release has one, ``ASD``, the first identifier in
    sorted order wins and the collision is logged, because a silent arbitrary winner on an
    abbreviation is the failure mode ``surface_index`` was written to avoid.
    """
    from hpo_extraction.retrieval.surface_index import build_surface_index, surface_key

    _surface2hpo, hpo2surfaces = build_surface_index(tree)
    seen: dict[str, str] = {}
    ids: list[str] = []
    texts: list[str] = []
    n_collisions = 0
    for hpo_id in sorted(hpo2surfaces):
        for surface in hpo2surfaces[hpo_id]:
            key = surface_key(surface.text)
            if key in seen:
                if seen[key] != hpo_id:
                    n_collisions += 1
                continue
            seen[key] = hpo_id
            ids.append(hpo_id)
            texts.append(surface.text)
    if n_collisions:
        logger.info("surface corpus: %d colliding form(s) resolved to the first id", n_collisions)
    logger.info("surface corpus: %d distinct form(s) over %d term(s)", len(texts), len(hpo2surfaces))
    return ids, texts


def build_surface_matrix(
    tree,
    encode_batch,
    *,
    encoder_name: str = SAPBERT_ENCODER,
    cache_path: str | Path,
    fingerprint: str,
    batch_size: int = 256,
):
    """Embed every label string once and cache it, keyed by ontology fingerprint + encoder.

    Same contract and same invalidation rule as
    :func:`ontology_index.build_definition_index`, over label strings, not definitions:
    ``encode_batch(list[str]) -> ndarray`` is supplied by the caller so this module never imports a
    model, and a cache built against another ontology or another encoder is refused, not
    warned about.
    """
    import numpy as np

    cache_path = Path(cache_path)
    ids, texts = surface_corpus(tree)
    logger.info("embedding %d surface form(s) with %s", len(texts), encoder_name)

    chunks = []
    for start in range(0, len(texts), batch_size):
        chunks.append(np.asarray(encode_batch(texts[start:start + batch_size]), dtype="float32"))
    vectors = np.vstack(chunks) if chunks else np.zeros((0, 1), dtype="float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, ids=np.array(ids), texts=np.array(texts), vectors=vectors,
        fingerprint=fingerprint, encoder=encoder_name,
    )
    logger.info("wrote %s (%d x %d)", cache_path, *vectors.shape)
    return ids, texts, vectors


def load_surface_matrix(cache_path: str | Path, *, fingerprint: str, encoder_name: str):
    """Load a cached surface matrix, or ``None`` when absent or stale.

    Staleness is a refusal for the reason ``load_definition_index`` gives: a cache built against
    another ontology indexes terms by a position that no longer means the same thing, and one built
    by another encoder returns confident cosines from an unrelated vector space.
    """
    import numpy as np

    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as data:
        if str(data["fingerprint"]) != fingerprint:
            logger.warning("%s was built against ontology %s, not %s — ignoring it",
                           cache_path, str(data["fingerprint"]), fingerprint)
            return None
        if str(data["encoder"]) != encoder_name:
            logger.warning("%s was built with %s, not %s — ignoring it",
                           cache_path, str(data["encoder"]), encoder_name)
            return None
        return ([str(i) for i in data["ids"]], [str(t) for t in data["texts"]], data["vectors"])


def sapbert_link_records(
    records: Sequence[dict],
    report_ids: Sequence[str],
    *,
    surface_ids: Sequence[str],
    surface_vectors,
    encode_batch,
    shape: str,
    tau: float = SAPBERT_TAU,
    model_key: str = "",
    batch_size: int = 256,
) -> tuple[dict, list[dict]]:
    """``(sent_hpos, linkage)``, nearest label string per candidate string, above *tau*.

    Every distinct candidate string in the cell is encoded **once** and the whole cell is resolved
    with one matrix product. That is not only faster: it makes the reader a pure function of the
    string, which is what lets the same cached row serve every downstream ablation, and it is the
    property the other two readers already have.

    A candidate whose best cosine falls below *tau* resolves to nothing and still gets a linkage row
    with its score. Those rows are the entire point, they are how S1's normaliser-union gap becomes
    attributable to "nobody wrote it", not "SapBERT refused it".
    """
    import numpy as np

    wanted = {str(rid) for rid in report_ids}
    pending: list[tuple[str, int, int, str]] = []
    for record in records:
        pid = str(record.get("patient_id", record.get("report_id", "")))
        if pid not in wanted:
            continue
        try:
            sent_num = int(record.get("sentence_number"))
        except (TypeError, ValueError):
            logger.warning("skipping record with unusable sentence_number in report %s", pid)
            continue
        for line_index, candidate in candidate_strings(record, shape):
            pending.append((pid, sent_num, line_index, candidate))

    if not pending:
        return {}, []

    # One encode per distinct string, not per occurrence. The same term is written by many models
    # on many sentences. On GSC+ this collapses the query set by roughly an order of magnitude.
    distinct = sorted({candidate for _, _, _, candidate in pending})
    chunks = []
    for start in range(0, len(distinct), batch_size):
        chunks.append(np.asarray(encode_batch(distinct[start:start + batch_size]), dtype="float32"))
    queries = np.vstack(chunks)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12

    sims = queries @ np.asarray(surface_vectors, dtype="float32").T
    best = sims.argmax(axis=1)
    best_score = sims[np.arange(sims.shape[0]), best]
    resolved = {
        text: (str(surface_ids[best[i]]), float(best_score[i]))
        for i, text in enumerate(distinct)
    }

    sent_hpos: dict[str, dict[int, dict[str, int]]] = {}
    linkage: list[dict] = []
    n_accepted = 0
    for pid, sent_num, line_index, candidate in pending:
        hpo_id, score = resolved[candidate]
        accepted = score >= tau
        if accepted:
            n_accepted += 1
            bucket = sent_hpos.setdefault(pid, {}).setdefault(sent_num, {})
            bucket[hpo_id] = bucket.get(hpo_id, 0) + 1
        linkage.append({
            "report_id": pid,
            "sentence_number": sent_num,
            "line_index": line_index,
            "candidate_string": candidate,
            "normaliser": "sapbert",
            "hpo_id": hpo_id if accepted else None,
            "score": score,
            "source": "sapbert" if accepted else "below_tau",
            "negated": False,
        })

    logger.info(
        "sapbert %s | %d candidate(s), %d distinct, %d accepted at tau=%.2f (%.1f%%)",
        model_key or "?", len(pending), len(distinct), n_accepted, tau,
        100.0 * n_accepted / max(1, len(pending)),
    )
    return sent_hpos, linkage


# ── The dispatch ─────────────────────────────────────────────────────────────

def dictionary_linkage(trace: Iterable[dict]) -> list[dict]:
    """:func:`agent_link.link_records`' trace, re-expressed in :data:`LINKAGE_FIELDS`.

    One trace row is one hypothesis line and may carry several identifiers. One linkage row is one
    identifier. An unresolved line keeps a row with ``hpo_id = None`` and its route, for the same
    reason SapBERT's below-tau rows are kept.

    ``line_index`` is ``-1``: ``link_records`` traces by line *text*, not by position, and inventing
    an index by re-parsing the reply here would be a second, drifting definition of what a line is.
    """
    out: list[dict] = []
    for row in trace:
        hpo_ids = list(row.get("hpo_ids") or [])
        base = {
            "report_id": str(row.get("report_id", "")),
            "sentence_number": int(row.get("sentence_number", -1)),
            "line_index": -1,
            "candidate_string": row.get("line", ""),
            "normaliser": "dictionary",
            "score": None,
            "source": row.get("route", "none"),
            "negated": False,
        }
        if not hpo_ids:
            out.append({**base, "hpo_id": None})
            continue
        for hpo_id in hpo_ids:
            out.append({**base, "hpo_id": hpo_id})
    return out


def run_phenobert_candidates(cfg, cell, records, report_ids, shape, work_dir: Path):
    """Ground the **parsed candidate strings** and return ``(output_dir, inputs, line_spans)``.

    This is the reader that makes the grid comparable at all. ``build_phenobert_input`` feeds
    ``terminate_lines(llm_output)``, the raw reply, so on ``q4_span_json`` the shipped column read
    JSON braces and field names while every other reader here reads the parsed ``term`` field.
    Comparing those directly would measure the parser.

    The input layout is byte-compatible with the shipped one (same sentence marker, same line
    termination, same block join), so ``parse_phenobert_sentences`` and the offset attribution work
    over it unchanged.
    """
    from hpo_extraction.phenojury.phenobert import run_phenobert

    inputs, line_spans = {}, {}
    for rid in report_ids:
        text, spans, lines = build_candidate_phenobert_input(records, rid, shape)
        inputs[str(rid)] = (text, spans)
        line_spans[str(rid)] = lines

    input_dir = work_dir / "pb_input"
    output_dir = work_dir / f"phenobert_output_{cell['model_key']}__candidates"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for rid, (text, _spans) in inputs.items():
        (input_dir / f"{rid}.txt").write_text(text, encoding="utf-8")
    logger.info("wrote %d candidate-input file(s) to %s", len(inputs), input_dir)

    run_phenobert(
        phenobert_dir=cfg.phenobert_dir, input_dir=str(input_dir), output_dir=str(output_dir),
        phenobert_python=cfg.phenobert_python, stanza_dir=cfg.stanza_dir,
        p1=float(cfg.p1), p2=float(cfg.p2), p3=float(cfg.p3),
        chunk_size=int(cfg.phenobert_chunk_size),
    )
    return str(output_dir), inputs, line_spans


def normalise_cell(
    records: Sequence[dict],
    report_ids: Sequence[str],
    normaliser: str,
    prompt_key: str = "",
    *,
    shape: str | None = None,
    model_key: str = "",
    cached_detections_path: str | Path | None = None,
    phenobert_output_dir: str | None = None,
    phenobert_inputs: dict | None = None,
    phenobert_line_spans: dict | None = None,
    surface2hpo: dict | None = None,
    ontology_index=None,
    dictionary_mode: str = "lexical",
    dictionary_accept_sources: Sequence[str] = ("exact", "synonym"),
    segment_lines: bool = True,
    surface_ids: Sequence[str] | None = None,
    surface_vectors=None,
    encode_batch=None,
    tau: float = SAPBERT_TAU,
) -> tuple[dict, list[dict]]:
    """``(detections, linkage)`` for one ``(prompt, model, normaliser)`` cell.

    The dispatch is thin. Each reader already exists somewhere and is called here,
    not reimplemented, so a difference between two columns of the grid is a difference
    between two readers the rest of the project also uses, not between this module's rendering of
    them.

    ``phenobert_candidates`` is the one reader this function does not run end to end: PhenoBERT is a
    subprocess in a different conda environment, so ``phenojury_normalisation``'s driver invokes
    :func:`phenobert_runner.run_phenobert` over the input
    :func:`build_candidate_phenobert_input` writes and passes the resulting directory back in here.
    Keeping the subprocess out of this module is what lets it import on a login node.
    """
    if shape is None:
        shape = _shape_of(prompt_key) if prompt_key else "plain"

    if normaliser in ("phenobert_raw", "phenobert_candidates"):
        if phenobert_output_dir is None:
            if normaliser != "phenobert_raw" or cached_detections_path is None:
                raise ValueError(
                    f"{normaliser} needs phenobert_output_dir (or, for phenobert_raw, "
                    "cached_detections_path) — there is nothing to read otherwise"
                )
            # The TSVs were not pulled with this cache. Detections still re-run. Linkage
            # cannot, and an empty list saying so beats a fabricated one.
            logger.warning(
                "no phenobert_output dir for %s/%s — detections from the cached jsonl, no linkage",
                prompt_key or "?", model_key or "?")
            return load_cached_detections(cached_detections_path), []
        linkage = phenobert_linkage(
            phenobert_output_dir, phenobert_inputs or {},
            normaliser=normaliser, line_spans=phenobert_line_spans,
        )
        return detections_from_linkage(linkage), linkage

    if normaliser == "dictionary":
        if surface2hpo is None:
            raise ValueError("dictionary needs surface2hpo from surface_index.build_surface_index")
        sent_hpos, trace = link_records(
            list(records), list(report_ids),
            surface2hpo=surface2hpo, index=ontology_index,
            prompt_key=prompt_key, shape=shape, model_key=model_key,
            mode=dictionary_mode, accept_sources=tuple(dictionary_accept_sources),
            segment_lines=segment_lines,
        )
        return sent_hpos, dictionary_linkage(trace)

    if normaliser == "sapbert":
        if surface_ids is None or surface_vectors is None or encode_batch is None:
            raise ValueError(
                "sapbert needs surface_ids, surface_vectors and encode_batch — build them with "
                "build_surface_matrix, which takes the encoder as a callable"
            )
        return sapbert_link_records(
            records, report_ids,
            surface_ids=surface_ids, surface_vectors=surface_vectors,
            encode_batch=encode_batch, shape=shape, tau=tau, model_key=model_key,
        )

    raise ValueError(f"unknown normaliser {normaliser!r}; expected one of {NORMALISERS}")


# ── Where a cell's inputs and outputs live ───────────────────────────────────
#
# The four prompt conditions are NOT stored alike, and pretending they are is the single easiest way to
# produce a grid with a silent hole in it. The Free Listing generation run wrote the p0 condition before earlier existed, so it
# sits at `{results}/phenojury_generation_free_listing/{cohort}/` with no prompt path segment and names its
# metrics file `slm_ensemble_slm_metrics.csv`. The three q conditions sit at
# `{results}/phenojury_generation_other_prompts/{cohort}/{prompt_key}/` and name theirs `slm_metrics.csv`.
# `hpo_extraction.phenojury.generation_prompts` documents the alternative, drop the Free Listing generation run's extractions into a `p0_baseline/`
# cell and they are picked up verbatim, and `source_cell_dir` accepts either, preferring the staged
# cell when it exists so a repository that has done the staging needs no special case at all.

#: The prompt whose cache lives under the Free Listing generation run, not generation run of the other prompts.
BASELINE_PROMPT = "p0_baseline"

EXP13_06_DIRNAME = "phenojury_generation_free_listing"
EXP14_01_DIRNAME = "phenojury_generation_other_prompts"


def source_cell_dir(results_dir: str | Path, cohort: str, prompt_key: str) -> Path:
    """Where ``(cohort, prompt_key)``'s cached generations and groundings actually are.

    Raises, not returning a non-existent path: a missing cell must stop the grid, not become
    a condition that quietly scores zero.
    """
    results_dir = Path(results_dir)
    staged = results_dir / EXP14_01_DIRNAME / cohort / prompt_key
    if staged.is_dir():
        return staged
    if prompt_key == BASELINE_PROMPT:
        legacy = results_dir / EXP13_06_DIRNAME / cohort
        if legacy.is_dir():
            return legacy
    raise FileNotFoundError(
        f"no cached cell for prompt {prompt_key!r} on cohort {cohort!r} under {results_dir} — "
        f"looked in {staged} and, for the baseline, {results_dir / EXP13_06_DIRNAME / cohort}"
    )


def cell_models(cell_dir: str | Path) -> list[str]:
    """The models whose grounding actually completed in this cell, in sorted order.

    A model is present only when its ``detections_*.jsonl`` exists **and is non-empty**, matching
    ``slm_ensemble_experiment``'s own definition of an active model. That definition is essential
    here: ``an earlier exploratory run`` D1 records ``q7_recall x apertus`` failing grounding three times while
    generation succeeded, which made that condition a seven-model ensemble. An analysis that assumed eight
    would have reported a matched contrast that was not matched.
    """
    cell_dir = Path(cell_dir)
    out = []
    for path in sorted(cell_dir.glob("detections_*.jsonl")):
        if path.stat().st_size > 0:
            out.append(path.name[len("detections_"):-len(".jsonl")])
    return out


def cell_extractions(cell_dir: str | Path, model_key: str) -> list[dict]:
    """One model's cached generations, skipping torn lines the way the driver's own reader does."""
    path = Path(cell_dir) / f"llm_extractions_{model_key}.jsonl"
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def grid_cell_dir(output_dir: str | Path, exp_id: str, cohort: str, prompt_key: str) -> Path:
    """Where this experiment writes ``(cohort, prompt_key)``'s re-normalised detections."""
    return Path(output_dir) / exp_id / cohort / prompt_key


def detections_name(model_key: str, normaliser: str) -> str:
    """File name of a juror's detections under one normaliser."""
    return f"detections_{model_key}__{normaliser}.jsonl"


def linkage_name(model_key: str, normaliser: str) -> str:
    """File name of a juror's candidate-to-term linkage under one normaliser."""
    return f"linkage_{model_key}__{normaliser}.jsonl"
