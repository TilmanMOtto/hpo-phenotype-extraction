"""The per-(report, node) analysis table every view reads. Server-side only. Never serialised.

One row per node the traversal visited, **plus** a synthetic row for every annotated term it never
reached. Without those synthetic rows the table would answer "what did the model get wrong about
the nodes it looked at", which is the smaller half of the question: at τ_prune = 0.5 on the earlier tree runs most annotated terms are never visited at all, and a table built only from ``*_nodes.jsonl``
would show a handful of false negatives and no sign of the hundreds that were never asked about.

The columns fall into four groups:

``state`` / ``expanded`` / ``accepted`` / ``prune_score`` / ``accept_score``
    What the run did. ``state`` is ``visited`` for a scored node and otherwise the ledger's fate
    for an annotated term that was not reached (``blocked``, ``unreached``, ``outside_graph``).

``in_gold`` / ``outcome`` / ``fp_class`` / ``fn_class`` / ``nearest_gold``
    What was true, and what kind of wrong the run was. Classification comes from
    ``hpo_extraction.evaluation.metrics.errors``, the same functions that produce the thesis's error
    taxonomy table, so a bucket count here and a bucket count there are the same measurement.

``subtree_has_gold``
    The label ``prune_score`` should be judged against. ``prune_score`` predicts *subtree
    presence*, not node presence. Scoring it against ``in_gold`` would mark a perfectly behaved
    gate as badly miscalibrated at every internal node. Computed by unioning the ancestors of the
    report's annotated terms, which is ``O(|gold|)`` per report rather than one subtree materialisation
    per visited node.

``n_calls`` / ``max_margin`` / ``max_cosine`` / ``n_yes``
    What the retriever and the SLM actually produced, joined from the ``*_calls.jsonl`` summary.
    Absent when the calls artifact is missing, which is not an error, it is tau-independent and
    large, and several views do not need it.

    Note that this join is **not** per-τ. ``tree_experiment`` memoises node evaluation across the
    whole τ sweep and writes the calls once (``tree_experiment.py:326-345``), so a node visited at
    τ = 0.1 but pruned away at τ = 0.5 still has call records. On a synthetic row that is a
    feature, it shows what the SLM *would* have said about a term pruning never let it reach, but it means ``n_calls`` must never be summed as this configuration's inference cost.
    ``*_traversal.jsonl``'s ``n_slm_calls`` is the per-τ cost, and it is what the scorecard uses.

**Ground truth is joined, never read off the artifact.** ``*_nodes.jsonl`` carries an ``is_gold`` flag, but
it records whichever ground truth file was configured at inference time. On HCY that is
``hcy_ground_truth.csv``, while the thesis tables were computed against
``hcy_ground_truth_raw.csv``, two different sets. The flag is kept as ``is_gold_artifact`` and any
disagreement is counted and surfaced, never reconciled.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pandas as pd

from . import pruning

COLUMNS = [
    "report_id", "hpo_id", "hpo_label", "state", "depth",
    "prune_score", "accept_score", "expanded", "accepted",
    "in_gold", "is_gold_artifact", "gold_disagrees", "outcome",
    "subtree_has_gold", "fate", "culprit", "culprit_prune_score",
    "fp_class", "fp_reference", "fp_distance", "fn_class",
    "layer1_organ", "subtree_size", "n_direct_children", "is_leaf",
    "n_calls", "max_margin", "max_cosine", "n_yes",
]


#: Columns supplied by a join, not by the row builders. Kept out of the initial frame so
#: The merge cannot be shadowed by an empty placeholder of the same name.
_JOINED_COLUMNS = frozenset({
    "fate", "culprit", "culprit_prune_score",
    "fp_class", "fp_reference", "fp_distance", "fn_class",
    "hpo_label", "layer1_organ", "subtree_size", "n_direct_children", "is_leaf",
    "n_calls", "max_margin", "max_cosine", "n_yes",
})


def _outcome(predicted: bool, in_gold: bool) -> str:
    if predicted and in_gold:
        return "TP"
    if predicted:
        return "FP"
    return "FN" if in_gold else "TN"


def build_node_table(
    nodes_df: pd.DataFrame,
    gold: dict[str, set[str]],
    view,
    ledger: dict,
    *,
    metadata: pd.DataFrame | None = None,
    call_summary: pd.DataFrame | None = None,
    classify_errors: bool = True,
) -> pd.DataFrame:
    """Assemble the table. ``nodes_df`` is one configuration's ``*_nodes.jsonl``.

    Args:
        nodes_df: as returned by :func:`loaders.load_nodes`.
        ground truth: ``{report_id: gold terms}`` from the chosen ground truth file.
        view: an ``OntologyView``.
        ledger: :func:`pruning.prune_ledger`'s output, for the unreached annotated terms' fates.
        metadata: ``*_node_metadata.jsonl``, optional.
        call_summary: ``scan_calls(...)["summary"]``, optional.
        classify_errors: run the Garcia FP taxonomy and the FN taxonomy. Each false positive costs
            a bounded ontology walk. On a run with tens of thousands of them that is seconds, not
            milliseconds, so the selfcheck turns it off.
    """
    visited = _visited_rows(nodes_df, gold, view)
    seen = set(zip(visited["report_id"], visited["hpo_id"])) if not visited.empty else set()
    unreached = _unreached_gold_rows(ledger, gold, view, seen=seen)

    # Columns filled by the joins below are absent here. Declaring them empty first
    # and merging afterwards makes pandas suffix the incoming values (``culprit_ledger``), and
    # The all-NaN placeholder wins, which silently empties the culprit column on the
    # rows that need it.
    columns = [c for c in COLUMNS if c not in _JOINED_COLUMNS]
    frames = [f for f in (visited, pd.DataFrame(unreached, columns=columns)) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)

    frame = _attach_fates(frame, ledger)
    if classify_errors:
        frame = _attach_error_classes(frame, gold, view)
    else:
        for column in ("fp_class", "fp_reference", "fp_distance", "fn_class"):
            frame[column] = pd.NA
    frame = _attach_metadata(frame, metadata, view)
    frame = attach_calls(frame, call_summary)
    return _compact(frame.reindex(columns=COLUMNS))


#: String columns cast to ``category`` once the table is assembled. All five are high-cardinality
#: ids or labels, which is where the bytes are: on a million-row table ``hpo_id`` alone is a million
#: pointers into ~18 000 distinct strings.
#:
#: The low-cardinality columns (``state``, ``outcome``, ``fate``, ``fp_class``, ``fn_class``,
#: ``layer1_organ``) are **left as objects**. They would save little, a handful of
#: shared strings behind a million pointers, and they are the ones read through ``value_counts()``
#: and ``fillna()``, both of which change behaviour on a categorical: ``value_counts`` reports every
#: declared category including the ones with no rows, and ``fillna`` with a value outside the
#: categories raises.
_COMPACT_COLUMNS = ("report_id", "hpo_id", "hpo_label", "culprit", "fp_reference")


def _compact(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast the id and label columns to ``category``, in place, once the joins are done.

    the *last* step. ``_attach_fates``, ``_attach_metadata`` and ``attach_calls`` all
    merge on ``report_id``/``hpo_id``, and a merge between a categorical and an object key upcasts
    the result back to object, so compacting earlier would cost the copy and keep none of it.
    """
    for column in _COMPACT_COLUMNS:
        if column in frame.columns and not isinstance(frame[column].dtype, pd.CategoricalDtype):
            frame[column] = frame[column].astype("category")
    return frame


def _seen_keys(rows: list[dict]) -> set[tuple[str, str]]:
    return {(r["report_id"], r["hpo_id"]) for r in rows}


def _visited_rows(nodes_df: pd.DataFrame, gold: dict[str, set[str]], view) -> pd.DataFrame:
    """One row per scored node, with the two ground truth labels the two scores are judged against.

    Built with whole-column operations, not a row loop. This is the largest table in the
    app, 540k rows on a full HCY tree run, and iterating it in Python to build one dict per row
    was six seconds of the fifteen a cold cell open took. The two set memberships are the only
    part that needs the ontology, and both are resolved into ``(report_id, hpo_id)`` pair sets
    once per report, then tested for the whole column at once.
    """
    if nodes_df is None or nodes_df.empty:
        return pd.DataFrame(columns=[c for c in COLUMNS if c not in _JOINED_COLUMNS])

    # Membership is raw string equality, matching ``scoring.align`` and therefore
    # ``result_tables/loaders.aligned_sets``. If this resolved and the scorecard did not, the table's
    # TP/FP/FN counts would quietly disagree with the metrics above them. Resolution is used
    # *only* for the ancestry lookup, where it is about reaching the right node in the graph
    #, not about deciding a label.
    annotated_pairs: set[tuple[str, str]] = set()
    ancestry_pairs: set[tuple[str, str]] = set()
    for report_id, terms in gold.items():
        report_id = str(report_id)
        for term in terms:
            annotated_pairs.add((report_id, str(term)))
            resolved = view.resolve(term)
            if resolved is None:
                continue
            # subtree_has_gold(n) == n is an ancestor-or-self of some annotated term, so unioning the
            # annotated terms' ancestor closures answers it for every node at once, |ground truth| lookups
            # per report instead of one subtree materialisation per visited node.
            for ancestor in view.ancestors(resolved):
                ancestry_pairs.add((report_id, ancestor))

    frame = pd.DataFrame({
        "report_id": nodes_df["report_id"].astype(str),
        "hpo_id": nodes_df["hpo_id"].astype(str),
    })
    keys = pd.MultiIndex.from_arrays([frame["report_id"], frame["hpo_id"]])
    in_gold = keys.isin(annotated_pairs)

    # subtree_has_gold is asked of the *resolved* node, so alt ids land on the same answer as the
    # terms they redirect to. Resolution is memoised per node by OntologyView, and the unique set
    # is far smaller than the column.
    unique_ids = frame["hpo_id"].unique()
    resolved_map = {h: (view.resolve(h) or h) for h in unique_ids}
    resolved_column = frame["hpo_id"].map(resolved_map)
    subtree_has_gold = pd.MultiIndex.from_arrays(
        [frame["report_id"], resolved_column]).isin(ancestry_pairs)

    accepted = nodes_df["accepted"].to_numpy(dtype=bool)
    artifact = pd.to_numeric(nodes_df["is_gold"], errors="coerce")

    frame["state"] = "visited"
    frame["depth"] = pd.to_numeric(nodes_df["depth"], errors="coerce")
    frame["prune_score"] = nodes_df["prune_score"].astype(float)
    frame["accept_score"] = nodes_df["accept_score"].astype(float)
    frame["expanded"] = nodes_df["expanded"].to_numpy(dtype=bool)
    frame["accepted"] = accepted
    frame["in_gold"] = in_gold
    frame["is_gold_artifact"] = artifact
    # Only meaningful where the artifact actually carried a flag; `&`/`!=` precedence here is the
    # opposite of what it reads like, so the grouping is spelled out.
    artifact_says_gold = artifact.fillna(0).astype(float) != 0
    frame["gold_disagrees"] = artifact.notna() & (artifact_says_gold != in_gold)
    frame["outcome"] = np.where(accepted,
                                np.where(in_gold, "TP", "FP"),
                                np.where(in_gold, "FN", "TN"))
    frame["subtree_has_gold"] = subtree_has_gold
    return frame.reindex(columns=[c for c in COLUMNS if c not in _JOINED_COLUMNS])


def _unreached_gold_rows(ledger: dict, gold: dict[str, set[str]], view,
                         seen: set[tuple[str, str]]) -> list[dict]:
    """A synthetic FN row for every annotated term the traversal never scored.

    These carry no score, the node was never asked, so ``prune_score`` and ``accept_score`` are
    NaN, not 0.0. Writing 0.0 would let them into the calibration sample as confidently
    absent, which is the exact mistake ``app/tree_ui`` documents as the reason its τ_prune slider
    is locked.
    """
    rows: list[dict] = []
    for entry in ledger.get("rows", ()):
        if entry["fate"] in ("found", "rejected"):
            continue
        key = (entry["report_id"], entry["hpo_id"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "report_id": entry["report_id"],
            "hpo_id": entry["hpo_id"],
            "state": entry["fate"],
            "depth": entry.get("depth"),
            "prune_score": np.nan,
            "accept_score": np.nan,
            "expanded": False,
            "accepted": False,
            "in_gold": True,
            "is_gold_artifact": None,
            "gold_disagrees": False,
            "outcome": "FN",
            "subtree_has_gold": True,
        })
    return rows


def _attach_fates(frame: pd.DataFrame, ledger: dict) -> pd.DataFrame:
    """Join the ledger's fate and culprit onto the ground truth rows.

    The culprit is the whole point of the blocked rows, it is what the deep-dive's "never
    reached, severed by X" note and the FN table's `blocked by` column read, so this join must
    land. It only lands because the three columns are absent from ``frame``. See
    :data:`_JOINED_COLUMNS`.
    """
    columns = ["fate", "culprit", "culprit_prune_score"]
    rows = ledger.get("rows", ())
    if not rows:
        frame["fate"] = pd.NA
        frame["culprit"] = pd.NA
        frame["culprit_prune_score"] = np.nan
        return frame

    fates = pd.DataFrame(rows)[["report_id", "hpo_id", *columns]]
    merged = frame.merge(fates, on=["report_id", "hpo_id"], how="left", validate="one_to_one")
    assert not any(f"{c}_x" in merged.columns for c in columns), \
        "the ledger join was shadowed by a placeholder column, see _JOINED_COLUMNS"
    return merged


def _attach_error_classes(frame: pd.DataFrame, gold: dict[str, set[str]], view) -> pd.DataFrame:
    """Garcia FP bucket + distance for every FP, and the FN bucket for every FN.

    Both are computed against the *resolved* ground truth and predicted sets of the same report, which is
    what ``errors.error_taxonomy`` and ``errors.false_negative_taxonomy`` do internally, so a
    per-row bucket here and the cohort table there agree by design, not by luck.
    """
    from hpo_extraction.evaluation.metrics import errors

    fp_class: dict[int, str] = {}
    fp_reference: dict[int, str | None] = {}
    fp_distance: dict[int, int | None] = {}
    fn_class: dict[int, str] = {}

    # Raw sets in, as ``result_tables/sections.py:369`` passes them to ``error_taxonomy``,
    # these functions resolve internally, and resolving twice would change which codes reach them.
    predicted_by_report: dict[str, set[str]] = defaultdict(set)
    for record in frame.loc[frame["accepted"] == True].itertuples():  # noqa: E712 - pandas mask
        predicted_by_report[record.report_id].add(str(record.hpo_id))

    # observed=True is a no-op while report_id is an object column, which it is here because
    # _compact runs last. Stated anyway: if that order ever changes, the silent failure is a
    # groupby that yields an empty group per unused category and classifies nothing.
    for report_id, group in frame.groupby("report_id", observed=True, sort=False):
        gold_res = set(gold.get(report_id, ()))
        pred_res = predicted_by_report.get(report_id, set())
        for record in group.itertuples():
            if record.outcome == "FP":
                bucket, reference = errors.classify_false_positive(record.hpo_id, gold_res, view)
                fp_class[record.Index] = bucket
                fp_reference[record.Index] = reference
                # ``classify_false_positive`` already ran the arg-min over the ground-truth set to find
                # ``reference``, so the eq. (5) distance is the distance to *that* term, one
                # more bidirectional BFS. Calling ``nearest_gold_distance`` here instead would
                # redo the whole search, doubling the dominant cost of the error analysis for a
                # number already implied by the answer.
                # ``assert_distance_matches_library`` pins the two forms together.
                fp_distance[record.Index] = (
                    view.undirected_distance(record.hpo_id, reference)
                    if reference is not None else None
                )
            elif record.outcome == "FN":
                fn_class[record.Index] = errors.classify_false_negative(
                    record.hpo_id, pred_res, view)

    frame["fp_class"] = pd.Series(fp_class, dtype="object").reindex(frame.index)
    frame["fp_reference"] = pd.Series(fp_reference, dtype="object").reindex(frame.index)
    frame["fp_distance"] = pd.Series(fp_distance, dtype="Float64").reindex(frame.index)
    frame["fn_class"] = pd.Series(fn_class, dtype="object").reindex(frame.index)
    return frame


def assert_distance_matches_library(predicted, gold, view) -> None:
    """Fix the reference-term shortcut to ``errors.nearest_gold_distance``.

    The table reports the distance to the term ``classify_false_positive`` picked as nearest,
    not re-running the arg-min. That is the same number by definition, but only as long
    as the two functions keep agreeing about which term is nearest, so it is asserted, not
    reasoned about.
    """
    from hpo_extraction.evaluation.metrics import errors

    for code in predicted:
        _, reference = errors.classify_false_positive(code, gold, view)
        mine = view.undirected_distance(code, reference) if reference is not None else None
        theirs = errors.nearest_gold_distance(code, gold, view)
        assert mine == theirs, f"{code}: distance {mine} != nearest_gold_distance {theirs}"


def _attach_metadata(frame: pd.DataFrame, metadata: pd.DataFrame | None, view) -> pd.DataFrame:
    """Join the node-metadata sidecar, falling back to the ontology for the label and organ.

    The sidecar only covers nodes this run evaluated, so the synthetic rows for never-reached ground truth
    terms would otherwise have no label at all, and an FN table of bare HPO ids is unreadable.
    """
    columns = ["subtree_size", "n_direct_children", "is_leaf", "layer1_organ"]
    if metadata is not None and not metadata.empty:
        keep = ["hpo_id", "hpo_label"] + [c for c in columns if c in metadata.columns]
        frame = frame.merge(metadata[keep], on="hpo_id", how="left")
    else:
        frame["hpo_label"] = pd.NA
        for column in columns:
            frame[column] = pd.NA

    missing = frame["hpo_label"].isna()
    if missing.any():
        frame.loc[missing, "hpo_label"] = [
            _label(view, hpo_id) for hpo_id in frame.loc[missing, "hpo_id"]
        ]

    # The sidecar covers only the nodes this run evaluated, so the synthetic rows for never-reached
    # annotated terms fall through to the ontology. Both routes must end up holding the same thing,
    # An organ-system *label*, or a chart grouped on the column would split one organ into two.
    if "layer1_organ" in frame.columns:
        gap = frame["layer1_organ"].isna()
        if gap.any():
            frame.loc[gap, "layer1_organ"] = [
                _layer1(view, hpo_id) for hpo_id in frame.loc[gap, "hpo_id"]
            ]
        frame["layer1_organ"] = [
            _label(view, value) if isinstance(value, str) and value.startswith("HP:") else value
            for value in frame["layer1_organ"]
        ]
    return frame


def _label(view, hpo_id: str) -> str:
    resolved = view.resolve(hpo_id) or hpo_id
    entry = view.tree.data.get(resolved)
    names = entry.get("Name") if entry else None
    return names[0] if names else hpo_id


def _layer1(view, hpo_id: str):
    """The primary layer-1 organ id for a node, or NA.

    Primary = lexicographically first, matching ``loaders._split_layer1``, a node under several
    organ systems must land in the same bucket whichever route it arrived by.
    """
    resolved = view.resolve(hpo_id)
    if not resolved:
        return pd.NA
    organs = sorted(view.layer1(resolved))
    return organs[0] if organs else pd.NA


def attach_calls(frame: pd.DataFrame, call_summary: pd.DataFrame | None) -> pd.DataFrame:
    """Join the per-(report, node) call summary. Missing calls artifact → NA columns, not zeros.

    Public because the scan that produces ``call_summary`` is the single most expensive thing this
    package does, a multi-gigabyte streaming read, and most views do not need it. The registry
    therefore builds the node table without calls and the two views that want them (FP anatomy's
    retrieval panel, the report deep-dive) call this afterwards with the scanned summary.
    """
    columns = ["n_calls", "max_margin", "max_cosine", "n_yes"]
    if call_summary is None or call_summary.empty:
        for column in columns:
            frame[column] = pd.NA
        return frame
    keep = ["report_id", "hpo_id"] + [c for c in columns if c in call_summary.columns]
    merged = frame.merge(
        call_summary[keep].astype({"report_id": "object", "hpo_id": "object"}),
        on=["report_id", "hpo_id"], how="left",
    )
    for column in columns:
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged


class ReportStates(Mapping):
    """``{report: {visited, expanded, accepted, prune_score}}``, materialised one report at a time.

    This is the shape :func:`pruning.prune_ledger` and :func:`pruning.fp_factory` take, and
    building all of it eagerly was the single most memory-expensive structure in the app: three
    ``set``s and a ``dict[str, float]`` per report, i.e. a second copy of the whole artifact in
    pure Python objects. At the loosest swept τ_prune that is a million interned strings and a
    million boxed floats. Both consumers iterate reports one at a time, so nothing needs them all
    to exist at once.

    What is retained instead is columnar: the categorical *codes* of ``hpo_id`` plus the boolean
    and float columns, and a dict of positional indices per report, tens of megabytes, not
    hundreds. ``__getitem__`` slices those and builds one report's dicts, caching the last one so
    a consumer that asks twice in a row does not pay twice.

    ``accept_score`` is **not** provided. Neither consumer reads it, and carrying it
    doubled the structure for nothing. The accept scores live in the node table, which is where
    every view that wants them already looks.

    Kept here, not in ``pruning`` so that module stays free of pandas and testable on
    hand-written dicts, a plain ``dict`` of the same shape still works everywhere this does.
    """

    def __init__(self, nodes_df: pd.DataFrame):
        self._indices: dict[str, object] = {}
        self._last: tuple[str, dict] | None = None
        if nodes_df is None or nodes_df.empty:
            self._codes = self._categories = None
            self._expanded = self._accepted = self._prune = None
            return

        hpo = nodes_df["hpo_id"]
        if not isinstance(hpo.dtype, pd.CategoricalDtype):
            hpo = hpo.astype(str).astype("category")
        self._codes = hpo.cat.codes.to_numpy()
        self._categories = hpo.cat.categories.astype(str).to_numpy()
        self._expanded = nodes_df["expanded"].to_numpy(dtype=bool)
        self._accepted = nodes_df["accepted"].to_numpy(dtype=bool)
        self._prune = nodes_df["prune_score"].to_numpy(dtype="float64")
        # observed=True: report_id is categorical, and a frame sliced to a subset of reports keeps
        # The unused categories. Without it every dropped report comes back as an empty group.
        grouped = nodes_df.groupby("report_id", observed=True, sort=True).indices
        self._indices = {str(rid): idx for rid, idx in grouped.items()}

    def _build(self, report_id: str) -> dict:
        idx = self._indices[report_id]
        ids = self._categories[self._codes[idx]]
        return {
            "visited": set(ids),
            "expanded": set(ids[self._expanded[idx]]),
            "accepted": set(ids[self._accepted[idx]]),
            "prune_score": dict(zip(ids, self._prune[idx])),
        }

    def __getitem__(self, report_id: str) -> dict:
        report_id = str(report_id)
        if report_id not in self._indices:
            raise KeyError(report_id)
        if self._last is not None and self._last[0] == report_id:
            return self._last[1]
        state = self._build(report_id)
        self._last = (report_id, state)
        return state

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def state_by_report(nodes_df: pd.DataFrame) -> ReportStates:
    """``*_nodes.jsonl`` → a lazy :class:`ReportStates` over it."""
    return ReportStates(nodes_df)


def gold_disagreements(frame: pd.DataFrame) -> int:
    """How many visited nodes' artifact ``is_gold`` flag disagrees with the joined ground-truth set."""
    if frame is None or frame.empty or "gold_disagrees" not in frame.columns:
        return 0
    return int(frame["gold_disagrees"].fillna(False).astype(bool).sum())


def false_positives_by_report(frame: pd.DataFrame) -> dict[str, list[str]]:
    """``{report_id: [FP hpo ids]}``, the input :func:`pruning.fp_factory` takes."""
    out: dict[str, list[str]] = defaultdict(list)
    if frame is None or frame.empty:
        return {}
    for record in frame.loc[frame["outcome"] == "FP"].itertuples():
        out[record.report_id].append(record.hpo_id)
    return dict(out)


def labels_map(frame: pd.DataFrame) -> dict[str, str]:
    """``{hpo_id: label}`` for everything in the table, the leaderboards' ``labels`` argument."""
    if frame is None or frame.empty:
        return {}
    pairs = frame[["hpo_id", "hpo_label"]].dropna().drop_duplicates(subset=["hpo_id"])
    return dict(zip(pairs["hpo_id"].astype(str), pairs["hpo_label"].astype(str)))


class LabelLookup(Mapping):
    """``labels_map`` with the ontology behind it.

    The node table only covers nodes the run visited, and a culprit is by design a node
    whose *children* were not visited, often it was not scored in the report being displayed
    either. Without a fallback the culprit leaderboard prints an HPO id twice and names nothing,
    which is the one column a reader most needs to be able to read.
    """

    def __init__(self, frame: pd.DataFrame, view):
        self._known = labels_map(frame)
        self._view = view
        self._cache: dict[str, str] = {}

    def __getitem__(self, key: str) -> str:
        key = str(key)
        if key in self._known:
            return self._known[key]
        if key not in self._cache:
            self._cache[key] = _label(self._view, key)
        return self._cache[key]

    def __iter__(self):
        return iter(self._known)

    def __len__(self) -> int:
        return len(self._known)

    def get(self, key, default=None):
        """Label of HPO identifier *key*, from the run's nodes or else the ontology; *default* when unknown."""
        try:
            return self[key]
        except Exception:
            return default
