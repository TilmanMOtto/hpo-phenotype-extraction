"""earlier artifacts on disk → DataFrames and plain dicts. No metric is computed here.

Three things this module is careful about, each because the naive version is silently wrong:

**The summary line.** ``*_predictions.jsonl`` holds one row per accepted term *plus* one
``{"report_id": ..., "summary": true, "predicted_set": [...], "gold_set": [...]}`` terminator per
report. That line is the only record of a report whose prediction set is empty, and at
τ_prune = 0.5 on the tree runs, many reports are that. Parsing it as a prediction row
yields a phantom prediction with no ``hpo_id``. Dropping it loses every empty report from the
denominator and inflates macro recall. It is split out and used as the authority on cohort
membership.

**Truncated tails.** A wall-clock kill leaves a half-written final line. It is raised, naming file
and line, not skipped, a partial cohort must be visible rather than quietly averaged. (The one
exception is the byte-offset scan, which by design runs over already-validated bytes.)

**The two big artifacts are indexed, not loaded whole.** ``*_calls.jsonl`` writes one line per
retrieved sentence per visited node per report, millions of lines and gigabytes, and
``*_nodes.jsonl`` writes one per visited node, which at the loosest swept τ_prune is 1 051 846 rows
for 170 MB on GSC+ and ~200× what the tightest τ writes. Both are handled the same way:
:func:`scan_offsets` makes one streaming pass recording each report's byte span, and
:func:`read_spans` then ``seek()``s to a single report's bytes. That is what lets the deep-dive open
a loose-τ cell at all, and it costs the same at either end of the sweep.

``*_nodes.jsonl`` *is* also read whole, by :func:`load_nodes`, because the cohort aggregates need
every row. That path is columnar, not a list of dicts, and its two id columns are
categorical, see :func:`_nodes_frame`.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from .discovery import FileSet

#: Yes/No is decided by ``margin > 0`` in ``hpo_extraction.models.llama.LlamaLogitsLLM._record``,
#: so the string is always one of these two. Compared by equality, not substring: a
#: substring test would count "No" inside a longer word, and there is no longer word to allow for.
YES = "Yes"


def read_jsonl(path: str):
    """Stream one JSONL file, skipping blank lines. A malformed line raises, naming file:line."""
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(f"{path}:{lineno}: {exc.msg}", exc.doc, exc.pos) from None


def read_fileset(fs: FileSet):
    """Stream every record of a FileSet, one merged file, or the shards in index order."""
    for path in fs.paths:
        yield from read_jsonl(path)


def fileset_mtime(fs: FileSet) -> float:
    """Newest mtime across the FileSet's paths, the cache key input. 0.0 when absent."""
    return max((os.path.getmtime(p) for p in fs.paths if os.path.isfile(p)), default=0.0)


# ── per-artifact readers ─────────────────────────────────────────────────────

_NODE_COLUMNS = {
    "report_id": "category", "hpo_id": "category", "depth": "int16",
    "prune_score": "float64", "accept_score": "float64",
    "expanded": "bool", "accepted": "bool", "is_gold": "int8",
}

#: Columns of ``*_nodes.jsonl`` that are read straight out of the JSON, in the order the columnar
#: builder fills them. Anything the driver adds later lands in neither and is dropped, which is
#: The point: at a loose τ_prune this file is a million rows, and an unexpected key would otherwise
#: become a million-element object column nothing reads.
_NODE_FIELDS = ("report_id", "hpo_id", "depth", "prune_score", "accept_score",
                "expanded", "accepted", "is_gold")


def empty_nodes() -> pd.DataFrame:
    """The zero-row node frame, typed as a loaded one, so callers need no special case."""
    return pd.DataFrame({
        column: pd.Series([], dtype=dtype) for column, dtype in _NODE_COLUMNS.items()
    })


#: Rows converted to typed arrays at a time. The columnar builder still accumulates Python objects
#: between flushes, one boxed float per score, one pointer per field, so without a bound the peak
#: is proportional to the file, not to this. 250k rows is ~15 MB of transient objects and
#: costs nothing in speed. The whole 1 051 846-row artifact in one go was 500 MB.
_NODES_CHUNK = 250_000


def _nodes_block(columns: dict) -> dict:
    """One chunk of accumulated Python lists → typed numpy arrays, freeing the objects."""
    return {
        "report_id": np.asarray([("" if v is None else str(v)) for v in columns["report_id"]],
                                dtype=object),
        "hpo_id": np.asarray([("" if v is None else str(v)) for v in columns["hpo_id"]],
                             dtype=object),
        # float first, so a missing key coerces to NaN, not raising. The -1 fills come
        # after the concat. is_gold = -1 reads as "the artifact did not say", which is not the
        # same claim as "the artifact said no", that distinction is why it is not filled with 0.
        "depth": pd.to_numeric(columns["depth"], errors="coerce").astype("float64"),
        "prune_score": pd.to_numeric(columns["prune_score"], errors="coerce"),
        "accept_score": pd.to_numeric(columns["accept_score"], errors="coerce"),
        "expanded": np.asarray([bool(v) for v in columns["expanded"]], dtype=bool),
        "accepted": np.asarray([bool(v) for v in columns["accepted"]], dtype=bool),
        "is_gold": pd.to_numeric(columns["is_gold"], errors="coerce").astype("float64"),
    }


def _nodes_frame(records) -> pd.DataFrame:
    """Build the typed node frame from an iterable of records, one column and one chunk at a time.

    ``pd.DataFrame(list(records))`` was the obvious spelling and is why a loose-tau cell could not
    be opened: it holds the whole file as one dict per row *before* the frame exists, then infers
    object columns from it. At tau_prune = 0.0045 on GSC+ that is 1 051 846 dicts for 170 MB of
    JSON, and the resulting frame is another 170 MB of object columns.

    Here the records are never materialised, each chunk is converted to typed arrays as soon as it
    is full, and ``report_id`` / ``hpo_id`` end up categorical, 228 and ~18 000 distinct values
    against a million rows, so the column is int codes plus one copy of each string, not a
    million pointers. The frame is 28 MB, and the peak is bounded by :data:`_NODES_CHUNK`.
    """
    columns: dict[str, list] = {name: [] for name in _NODE_FIELDS}
    blocks: list[dict] = []
    n_buffered = 0

    for record in records:
        for name in _NODE_FIELDS:
            columns[name].append(record.get(name))
        n_buffered += 1
        if n_buffered >= _NODES_CHUNK:
            blocks.append(_nodes_block(columns))
            for value in columns.values():
                value.clear()
            n_buffered = 0
    if n_buffered:
        blocks.append(_nodes_block(columns))
    del columns

    if not blocks:
        return empty_nodes()

    joined = {name: (blocks[0][name] if len(blocks) == 1
                     else np.concatenate([b[name] for b in blocks]))
              for name in _NODE_FIELDS}
    blocks.clear()

    frame = pd.DataFrame({
        "report_id": pd.Categorical(joined["report_id"]),
        "hpo_id": pd.Categorical(joined["hpo_id"]),
        "depth": pd.Series(joined["depth"]).fillna(-1).astype("int16"),
        "prune_score": joined["prune_score"],
        "accept_score": joined["accept_score"],
        "expanded": joined["expanded"],
        "accepted": joined["accepted"],
        "is_gold": pd.Series(joined["is_gold"]).fillna(-1).astype("int8"),
    })
    return frame


def load_nodes(fs: FileSet) -> pd.DataFrame:
    """``*_nodes.jsonl`` → one row per (report, visited node) at this configuration.

    Nodes the traversal never reached are absent by design, that absence *is* the pruning
    signal, and reconstructing them is :mod:`~apps.treephenorag_ui.pruning`'s job, not this one's.
    """
    if not fs.present:
        return empty_nodes()
    frame = _nodes_frame(read_fileset(fs))
    if frame.empty:
        return empty_nodes()
    # A resumed run can re-append a report's block. Merge_shards de-duplicates, unmerged shards
    # do not. Keeping the first occurrence matches merge_shards.py's own rule. Run on the
    # category *codes*: duplicated() over two object columns of a million strings hashes every one
    # of them, over the codes it hashes two int arrays.
    duplicated = pd.DataFrame({
        "report_id": frame["report_id"].cat.codes,
        "hpo_id": frame["hpo_id"].cat.codes,
    }).duplicated(keep="first")
    if not duplicated.any():
        return frame
    return frame.loc[~duplicated].reset_index(drop=True)


def load_predictions(fs: FileSet) -> dict:
    """``*_predictions.jsonl`` → ``{"predicted", "gold", "report_ids", "n_hit_rows"}``.

    ``predicted`` and ``gold`` are ``{report_id: set[str]}`` taken from the summary lines, which
    are authoritative: they list every report the driver processed, including the ones that
    predicted nothing. The per-hit rows are counted only as a cross-check (``n_hit_rows``). Any
    report present in the hit rows but absent from the summary lines is reported in
    ``reports_without_summary``, not silently added.
    """
    predicted: dict[str, set[str]] = {}
    gold: dict[str, set[str]] = {}
    hit_reports: set[str] = set()
    n_hit_rows = 0

    for rec in read_fileset(fs):
        rid = rec.get("report_id")
        if rec.get("summary"):
            # Later shards must not clobber an earlier summary for the same report. First wins,
            # matching the de-duplication rule everywhere else in this module.
            if rid not in predicted:
                predicted[rid] = {str(h) for h in rec.get("predicted_set") or []}
                gold[rid] = {str(h) for h in rec.get("gold_set") or []}
        else:
            n_hit_rows += 1
            if rid is not None:
                hit_reports.add(rid)

    return {
        "predicted": predicted,
        "gold": gold,
        "report_ids": sorted(predicted),
        "n_hit_rows": n_hit_rows,
        "reports_without_summary": sorted(hit_reports - set(predicted)),
    }


def load_traversal(fs: FileSet) -> pd.DataFrame:
    """``*_traversal.jsonl`` → one row per report, ``depth_table`` kept as a list column."""
    if not fs.present:
        return pd.DataFrame(columns=["report_id", "n_unique_nodes_visited",
                                     "total_frontier_insertions", "n_slm_calls",
                                     "n_accepted", "depth_table"])
    frame = pd.DataFrame(list(read_fileset(fs)))
    if frame.empty:
        return frame
    return frame.drop_duplicates(subset=["report_id"], keep="first").reset_index(drop=True)


def load_timing(fs: FileSet) -> pd.DataFrame:
    """``*_timing.jsonl`` → one row per report. Note this is JSONL. The earlier runs/10 wrote a JSON array.

    Per-report is the tree and flat-top-M shape, not a universal one: the PhenoBERT baseline times a single
    ``annotate`` subprocess over the whole cohort and writes one *stage* row with no ``report_id``
    at all (``phenobert_experiment.py:287-294``). De-duplicating on a column that is not there
    raises, so the de-duplication is conditional, there is nothing to collapse in a file that has
    one row per run.
    """
    if not fs.present:
        return pd.DataFrame(columns=["report_id", "n_slm_calls", "n_nodes_visited"])
    frame = pd.DataFrame(list(read_fileset(fs)))
    if frame.empty or "report_id" not in frame.columns:
        return frame
    return frame.drop_duplicates(subset=["report_id"], keep="first").reset_index(drop=True)


def load_node_metadata(fs: FileSet) -> pd.DataFrame:
    """``*_node_metadata.jsonl`` → one row per node, de-duplicated on ``hpo_id``.

    This sidecar is static per node and is written once per run, so a shard union repeats most
    of it; ``merge_shards.py`` de-duplicates on ``hpo_id`` and so does this.

    ``layer1_organ`` arrives as a **list**, HPO is a DAG, so a node can sit under several organ
    systems (``core/node_metadata.py:58``). A list in a DataFrame column is unhashable, so any
    ``groupby`` or ``value_counts`` on it raises. It is split here into a scalar
    ``layer1_organ`` (the primary, i.e. lexicographically first, so the choice is stable across
    runs) plus ``layer1_organs`` (the full list) and ``n_layer1_organs``. Charts group by the
    scalar. Nothing is lost.
    """
    columns = ["hpo_id", "hpo_label", "subtree_size", "n_direct_children", "is_leaf",
               "depth_bfs", "depth_long", "layer1_organ", "layer1_organs", "n_layer1_organs",
               "in_phenotypic_abnormality"]
    if not fs.present:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(list(read_fileset(fs)))
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame.drop_duplicates(subset=["hpo_id"], keep="first").reset_index(drop=True)
    return _split_layer1(frame)


def _split_layer1(frame: pd.DataFrame) -> pd.DataFrame:
    """``layer1_organ: [ids]`` → scalar primary + the list + a count. Idempotent."""
    if "layer1_organ" not in frame.columns:
        return frame
    values = frame["layer1_organ"]
    if not values.map(lambda v: isinstance(v, list)).any():
        frame["layer1_organs"] = values.map(lambda v: [] if v is None else [v])
        frame["n_layer1_organs"] = frame["layer1_organs"].map(len)
        return frame

    organs = values.map(lambda v: sorted(v) if isinstance(v, list) else ([] if v is None else [v]))
    frame["layer1_organs"] = organs
    frame["n_layer1_organs"] = organs.map(len)
    frame["layer1_organ"] = organs.map(lambda v: v[0] if v else None)
    return frame


def load_segments(fs: FileSet) -> pd.DataFrame:
    """``rag_hpo_retrieved_segments.jsonl``, the one earlier artifact that still carries text."""
    if not fs.present:
        return pd.DataFrame(columns=["report_id", "hpo_id", "hpo_label", "rank", "text",
                                     "cosine_sim", "slm_verdict"])
    return pd.DataFrame(list(read_fileset(fs)))


# ── byte-offset indexing, shared by nodes.jsonl and calls.jsonl ──────────────

def scan_offsets(fs: FileSet, visit=None) -> dict:
    """One streaming pass over a FileSet, recording each report's byte span.

    Returns ``{"offsets": {report_id: [(path_index, start, end), ...]}, "paths": [...],
    "n_rows": int}``. *visit*, when given, is called with every parsed record, so a caller that
    also wants a summary gets it from the same pass, not a second one.

    The ranges are spans of *whole lines*, so a reader can ``seek`` + ``read`` and hand the bytes
    straight back to ``json.loads``. They are accumulated per contiguous run, not per line:
    the drivers write a report's records as one block, so a report is normally one range per shard,
    and a resumed run that re-did a block gets two. Building them generically costs nothing and
    removes the need to trust contiguity.
    """
    offsets: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    n_rows = 0

    for path_index, path in enumerate(fs.paths):
        current_rid: str | None = None
        run_start = 0
        with open(path, "rb") as f:
            offset = 0
            for raw in f:
                length = len(raw)
                start = offset
                offset += length
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise json.JSONDecodeError(
                        f"{path}: byte {start}: {exc.msg}", exc.doc, exc.pos) from None
                n_rows += 1
                rid = rec.get("report_id")
                if rid != current_rid:
                    if current_rid is not None:
                        offsets[current_rid].append((path_index, run_start, start))
                    current_rid, run_start = rid, start
                if visit is not None:
                    visit(rec)
            if current_rid is not None:
                offsets[current_rid].append((path_index, run_start, offset))

    return {"offsets": dict(offsets), "paths": list(fs.paths), "n_rows": n_rows}


def read_spans(scan: dict, report_id: str):
    """Yield one report's records by seeking to its byte ranges. Nothing else is parsed."""
    ranges = scan.get("offsets", {}).get(str(report_id))
    if not ranges:
        return
    paths = scan["paths"]
    by_path: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for path_index, start, end in ranges:
        by_path[path_index].append((start, end))
    for path_index, spans in by_path.items():
        with open(paths[path_index], "rb") as f:
            for start, end in sorted(spans):
                f.seek(start)
                blob = f.read(end - start)
                for line in blob.splitlines():
                    line = line.strip()
                    if line:
                        yield json.loads(line)


# ── nodes.jsonl: the per-report index that keeps a loose τ_prune openable ─────

def scan_nodes(fs: FileSet) -> dict:
    """Index ``*_nodes.jsonl`` by report without keeping a single record.

    This is what makes the deep-dive independent of τ_prune. The cohort node table is
    ``#reports × #visited nodes``, and #visited nodes runs ~200× between the tightest and loosest
    swept τ_prune, 24.8 nodes per report at τ = 0.62 against 4 613 at τ = 0.0045, where the file
    is 1 051 846 rows. Reading one report's block is the same work at either end of the sweep.

    Adds ``report_ids`` to :func:`scan_offsets`'s result, in file order, so the deep-dive can list
    reports without touching predictions.
    """
    scan = scan_offsets(fs)
    scan["report_ids"] = [rid for rid in scan["offsets"] if rid is not None]
    return scan


def nodes_for_report(scan: dict, report_id: str) -> pd.DataFrame:
    """One report's node records, typed like :func:`load_nodes`'s cohort-wide frame.

    Bounded by one report's visited nodes, so it costs the same on any configuration. The
    de-duplication rule is :func:`load_nodes`'s, applied within the report, a resumed run that
    re-wrote this report's block is the only way a duplicate reaches here.
    """
    frame = _nodes_frame(read_spans(scan, report_id))
    if frame.empty:
        return frame
    duplicated = frame["hpo_id"].cat.codes.duplicated(keep="first")
    if not duplicated.any():
        return frame
    return frame.loc[~duplicated].reset_index(drop=True)


# ── calls.jsonl: summary pass + byte-offset index ────────────────────────────

CALL_SUMMARY_COLUMNS = ["report_id", "hpo_id", "n_calls", "max_margin", "min_margin",
                        "max_cosine", "mean_cosine", "n_yes"]


def scan_calls(fs: FileSet) -> dict:
    """One streaming pass over ``*_calls.jsonl``: per-(report, node) summary + byte offsets.

    Returns ``{"summary": DataFrame, "offsets": {report_id: [(path_index, start, end), ...]},
    "paths": [...], "n_rows": int}``, :func:`scan_offsets`'s result plus the summary, both built
    in the one pass.
    """
    summary: dict[tuple[str, str], list] = {}

    def visit(rec) -> None:
        key = (rec.get("report_id"), rec.get("hpo_id"))
        margin = float(rec.get("margin", 0.0))
        cosine = float(rec.get("cosine_sim", 0.0))
        is_yes = int(rec.get("verdict") == YES)
        entry = summary.get(key)
        if entry is None:
            summary[key] = [1, margin, margin, cosine, cosine, is_yes]
        else:
            entry[0] += 1
            entry[1] = max(entry[1], margin)
            entry[2] = min(entry[2], margin)
            entry[3] = max(entry[3], cosine)
            entry[4] += cosine
            entry[5] += is_yes

    scan = scan_offsets(fs, visit)

    if summary:
        keys = list(summary)
        values = np.asarray([summary[k] for k in keys], dtype="float64")
        frame = pd.DataFrame({
            "report_id": pd.Series([k[0] for k in keys], dtype="string"),
            "hpo_id": pd.Series([k[1] for k in keys], dtype="string"),
            "n_calls": values[:, 0].astype("int32"),
            "max_margin": values[:, 1],
            "min_margin": values[:, 2],
            "max_cosine": values[:, 3],
            "mean_cosine": values[:, 4] / np.maximum(values[:, 0], 1.0),
            "n_yes": values[:, 5].astype("int32"),
        })
    else:
        frame = pd.DataFrame(columns=CALL_SUMMARY_COLUMNS)

    scan["summary"] = frame
    return scan


def calls_for_report(scan: dict, report_id: str) -> pd.DataFrame:
    """Every SLM call of one report, read by seeking to its byte ranges.

    This is the only path that ever materialises raw call records, and it is bounded by one
    report's worth of them.
    """
    records = list(read_spans(scan, report_id))
    if not records:
        return pd.DataFrame(columns=["report_id", "hpo_id", "ctx_type", "rank", "sent_index",
                                     "cosine_sim", "logit_yes", "logit_no", "logsumexp_all",
                                     "top1_token_id", "margin", "verdict"])
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    subset = [c for c in ("hpo_id", "ctx_type", "rank") if c in frame.columns]
    return frame.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


# ── report texts ─────────────────────────────────────────────────────────────

#: The encoding ``hpo_extraction.data.loading.load_txt`` reads reports with, and therefore the encoding the
#: pipeline's segmentation was computed over. It is not a guess about the files, it is a
#: transcription of what inference did, and it has to stay that way.
REPORT_ENCODING = "latin1"


def load_report_texts(input_dir: str) -> dict[str, str]:
    """``{report_id: text}`` from a directory of ``.txt`` files, as inference read them.

    A transcription of ``hpo_extraction.data.loading.load_txt``, the ``.txt`` filter, the ``latin1``
    decode, and the ``split(".")[0].replace("_", ":")`` keying, not an import, because
    that module pulls in ``sentence_transformers`` at import time and this app runs on a login
    node. :func:`assert_matches_load_txt` pins the transcription.

    **The encoding is essential, not a detail.** This used to decode ``utf-8`` with
    ``errors="replace"`` while claiming parity in its docstring. Every HCY report is German, so
    every report containing an umlaut decoded to a different string here than it did during
    inference, and since ``evidence.aligns_to_text`` checks a recovered segmentation by locating
    its sentences in this text, *every such report was silently dropped* and the deep-dive showed
    "sentence text unavailable" for it. A decoding difference is invisible until it is essential,
    and here it is essential.
    """
    texts: dict[str, str] = {}
    if not input_dir or not os.path.isdir(input_dir):
        return texts
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding=REPORT_ENCODING) as f:
            texts[name.split(".")[0].replace("_", ":")] = f.read()
    return texts


def assert_matches_load_txt(input_dir: str) -> None:
    """Fix :func:`load_report_texts` to ``hpo_extraction.data.loading.load_txt``, key for key and byte for
    byte. Run from the self-check on a machine where the reports are actually reachable."""
    from hpo_extraction.data.loading import load_txt

    expected = load_txt(input_dir)
    actual = load_report_texts(input_dir)
    differing = sorted(k for k in set(actual) & set(expected) if actual[k] != expected[k])
    assert actual == expected, "\n".join([
        "loaders.load_report_texts drifted from hpo_extraction.data.loading.load_txt",
        f"  keys only here:     {sorted(set(actual) - set(expected))[:5]}",
        f"  keys only there:    {sorted(set(expected) - set(actual))[:5]}",
        f"  differing contents: {differing[:5]}",
    ])
