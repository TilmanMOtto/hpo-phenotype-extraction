"""
Retrieval-stage analysis against segment-level ground truth.

Shared core of the retrieval experiments in group 0, promoted here once
``an earlier exploratory run`` needed the same ontology helpers, scoring,
metrics and plots as ``an earlier exploratory run``.

The experiments themselves keep only their stage functions (``run_compute``,
``run_report``, …). Everything reusable lives here:

  * ontology     ``ancestors_with_hops``, ``descendants_or_self``, ``label_of``
  * scoring      ``max_pooled_sims`` (+ ``verify_against_calculator``)
  * metrics      ``metrics_by_hop``, ``per_hpo_table``, ``threshold_sweep``, ``auc``
  * plots        ``plot_*``

Two conventions every caller depends on:

**True-path rule.** A segment is relevant for query HPO ``q`` if it is annotated with
``q`` *or with any descendant of ``q``*. Without propagating annotations up the
ontology, every ancestor query would look like a miss by design.

**Negatives.** An (segment, HPO) pair with no annotation is treated as a true negative.
That only holds for exhaustively reviewed patients. The **curated** ground truth
(``<hcy>/curated_ground_truth_<date>/hcy_curated_annotations.csv``, from ``hcy_ground_truth``) is the file
that assumption is now safe on: every report was read end to end and every annotation either
placed on the words it came from or explicitly marked unplaceable. Its predecessor
``annotations_confirmed.csv`` is still readable, and ``load_inputs`` tells the two apart.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp, rankdata  # noqa: E402

from hpo_extraction.evaluation.segment_metrics import compute_mrr, compute_segment_recall_at_k  # noqa: E402
from hpo_extraction.ontology.hpo_tree import HPOTree  # noqa: E402

K_VALUES = [1, 3, 5, 10, 20]

# The pipeline's retrieval window. Everything ranked below this is a sentence the
# LLM never sees, which is what "missed" means throughout these experiments.
DASHBOARD_TOP_N = 5

# Hop levels beyond this are pooled into one bucket, the tail is thin and long.
MAX_HOP_BUCKET = 4


# ── Ontology helpers ──────────────────────────────────────────────────────────

def ancestors_with_hops(tree: HPOTree, hpo: str) -> dict[str, int]:
    """
    {ancestor → edge distance from *hpo*}, including *hpo* itself at distance 0.

    ``getAllFatherHPOByHPO`` returns the transitive ancestor *set* with no distance,
    so walk ``Is_a`` (direct parents) breadth-first to recover the hop count. The HPO
    DAG lets a node be reached by several paths; BFS takes the shortest, which is the
    conservative (most specific) attribution.

    Restricted to the phenotypic-abnormality subtree, so the near-root ``HP:0000001``
    ("All") and ``HP:0000118`` ("Phenotypic abnormality") never enter the query set, their contexts are too generic to retrieve against meaningfully.
    """
    pa = tree.phenotypic_abnormalityNT
    if hpo not in pa:
        return {}

    hops: dict[str, int] = {hpo: 0}
    queue: deque[str] = deque([hpo])
    while queue:
        node = queue.popleft()
        for parent in tree.data[node]["Is_a"]:
            if parent in pa and parent not in hops:
                hops[parent] = hops[node] + 1
                queue.append(parent)
    return hops


def descendants_or_self(tree: HPOTree, hpo: str) -> set[str]:
    """{hpo} ∪ all transitive descendants, the set whose annotations imply *hpo*."""
    return {hpo} | set(tree.data[hpo]["Child"].keys())


def descendants_with_hops(tree: HPOTree, hpo: str) -> dict[str, int]:
    """
    {descendant → edge distance *down* from *hpo*}, including *hpo* itself at 0.

    The mirror of ``ancestors_with_hops``, walking ``Son`` (direct children) instead of
    ``Is_a``. Subtracting ``depth_dict`` values would not do: the HPO DAG can reach a
    descendant by a shorter path through another branch, so its global depth may be no
    greater than the query's. BFS from the query gives the true distance within *this*
    subtree, which is what a depth-capped context union means.
    """
    pa = tree.phenotypic_abnormalityNT
    if hpo not in pa:
        return {}

    hops: dict[str, int] = {hpo: 0}
    queue: deque[str] = deque([hpo])
    while queue:
        node = queue.popleft()
        for child in tree.data[node]["Son"]:
            if child in pa and child not in hops:
                hops[child] = hops[node] + 1
                queue.append(child)
    return hops


def label_of(tree: HPOTree, hpo: str) -> str:
    """Label of an HPO term, or the identifier itself when the ontology does not know it."""
    try:
        return str(tree.getNameByHPO(hpo))
    except Exception:
        return hpo


#: The column that identifies ``hcy_curated_annotations.csv``, the annotation table shipped by
#: ``hcy_ground_truth``'s ``dataset`` stage, as opposed to the legacy ``annotations_confirmed.csv``.
CURATED_MARKER = "in_gold"

#: The rest of the curated table's evidence columns, which this module then relies on.
CURATED_COLUMNS = ("in_gold", "anchored", "anchor_how")

#: How an annotation came to sit on its segment, in ``curated_gold.ANCHOR_KINDS`` order.
#: ``curated`` is a person pointing at the words. ``lexical`` is the curation harness scanning the
#: whole report for the trigger and taking the first hit, a guess that can land on the wrong
#: occurrence, which is why it can be excluded (``exclude_anchor_how=[lexical]``) rather than
#: silently trusted.
LOCATION_KINDS = ("curated", "segment", "offset", "context", "lexical")

_TRUE = {"1", "true", "True", "TRUE", "yes"}


def _flags(column: pd.Series) -> pd.Series:
    return column.astype(str).str.strip().isin(_TRUE)


def _load_curated(
    ann: pd.DataFrame, path: str, require_location: bool, exclude_location_how: tuple
) -> tuple[pd.DataFrame, dict]:
    """``hcy_curated_annotations.csv`` → the (patient, segment, HPO) triples to score against.

    The file is one row per **candidate** annotation, every term the curation pass considered,
    the excluded ones flagged, not deleted (``in_gold`` / ``exclude_reason``) so a reader
    holding a different policy can rebuild a different ground truth. This applies the shipped policy and
    nothing more:

    * ``in_gold == 1``, the policy's verdict, with qualifiers and deletion proposals already
      applied;
    * ``anchored == 1``, the term sits on a segment and a trigger word that occurs in it. Under
      the default policy ``in_gold`` implies this, but the ``keep_unanchored`` variant breaks the
      implication, and an unanchored annotation has no segment to rank at all, it is unusable
      *here* whatever a code-set scorer does with it;
    * optionally, ``anchor_how`` not in *exclude_anchor_how*.

    ``prior_annotation`` and ``daphne`` overlap by design, so one phenotype can arrive as several
    rows. They are folded to one row per (patient, segment, HPO), the unit this analysis ranks, keeping the best-ranked ``anchor_how`` as the surviving row's provenance.
    """
    missing = [c for c in CURATED_COLUMNS if c not in ann.columns]
    if missing:
        raise ValueError(f"{path} looks like a curated annotation table but has no "
                         f"{missing} column(s)")

    n_all = len(ann)
    prov: dict = {"gold_source": "curated", "n_candidates": n_all,
                  "n_excluded_by_anchor_how": 0}

    in_gold = _flags(ann["in_gold"])
    prov["n_excluded_by_policy"] = int((~in_gold).sum())
    ann = ann[in_gold]

    if require_location:
        anchored = _flags(ann["anchored"])
        prov["n_unanchored"] = int((~anchored).sum())
        ann = ann[anchored]
    else:
        # Even with the evidence rule off, a row with no segment index has nothing to rank.
        placed = ann["segment_idx"].astype(str).str.strip() != ""
        prov["n_unanchored"] = int((~placed).sum())
        ann = ann[placed]

    ann = ann.copy()
    ann["segment_idx"] = pd.to_numeric(ann["segment_idx"], errors="coerce")
    unplaceable = ann["segment_idx"].isna()
    if unplaceable.any():
        raise ValueError(
            f"{path}: {int(unplaceable.sum())} annotation(s) are anchored and in the gold but "
            f"carry no segment_idx — the table disagrees with its own evidence rule.")
    ann["segment_idx"] = ann["segment_idx"].astype(int)

    kept = ann["anchor_how"].map(
        lambda h: LOCATION_KINDS.index(h) if h in LOCATION_KINDS else len(LOCATION_KINDS))
    ann = (ann.assign(_anchor_rank=kept)
              .sort_values("_anchor_rank", kind="stable")
              .drop_duplicates(subset=["patient_id", "segment_idx", "hpo_code"], keep="first")
              .drop(columns="_anchor_rank"))
    n_folded = (n_all - prov["n_excluded_by_policy"] - prov["n_unanchored"] - len(ann))
    prov["n_folded_duplicates"] = n_folded

    # After the fold, not before: an annotation two files recorded, one by hand, one by a
    # lexical scan, is one annotation with a curator's placement, and excluding `lexical` should
    # not cost it. Applied here, the count means annotations actually lost, not rows seen.
    if exclude_location_how:
        excluded = ann["anchor_how"].astype(str).str.strip().isin(set(exclude_location_how))
        prov["n_excluded_by_anchor_how"] = int(excluded.sum())
        prov["excluded_anchor_how"] = ",".join(exclude_location_how)
        ann = ann[~excluded]

    by_how = ann["anchor_how"].value_counts().to_dict()
    prov["anchor_how"] = ",".join(f"{k}={v}" for k, v in sorted(by_how.items()))
    prov["n_lexical"] = int(by_how.get("lexical", 0))
    print(f"      curated gold: {n_all} candidate rows → {len(ann)} scoreable annotations "
          f"({prov['n_excluded_by_policy']} excluded by policy, "
          f"{prov['n_unanchored']} unanchored, "
          f"{prov['n_folded_duplicates']} duplicate rows folded, "
          f"{prov['n_excluded_by_anchor_how']} excluded by anchor_how)")
    print(f"      placed by: {prov['anchor_how'] or '(none)'}")
    return ann, prov


def load_inputs(
    segmented_reports: str,
    annotations: str,
    gold_source: str = "auto",
    require_location: bool = True,
    exclude_location_how: object = (),
) -> tuple[dict[str, list[str]], pd.DataFrame, list[str]]:
    """Segments + annotations, joined on patient and checked for segmentation drift.

    Two annotation formats are accepted, and which one arrived is detected from the columns.
    *gold_source* ``curated``/``confirmed`` forces the reading, and then a mismatch is an error,
    not a silent fallback, reading the curated table as a flat file would score against
    the very terms its policy excluded.

    ``curated``
        ``<hcy>/curated_ground_truth_<date>/hcy_curated_annotations.csv``, from ``hcy_ground_truth``'s ``dataset``
        stage. **The ground truth this analysis should be measured against.** Every term names the segment
        and the words it came from, which is the join a rank analysis needs, and the terms
        whose trigger word occurs nowhere in the report, unretrievable by design, and
        indistinguishable from a labelling error, are marked, not counted as misses.
    ``confirmed``
        ``annotations_confirmed.csv``, the pre-curation file: one row per already-located
        annotation, no policy columns. Kept so an old run reproduces.

    ``segment_idx`` is index-compatible with *segmented_reports* in both, because the curation app
    lays its screen out on the same ``segmented_reports.csv`` read by the same group-and-sort
    (``sources.load_segments`` ≡ ``data_loader.load_segmented_reports``).

    Returns ``(segments_by_patient, ann, patients)``. ``ann.attrs["gold"]`` carries the provenance
    counts, which callers log and print.
    """
    from hpo_extraction.data.segmented_reports import load_segmented_reports

    if gold_source not in ("auto", "curated", "confirmed"):
        raise ValueError(f"gold_source must be auto|curated|confirmed, not {gold_source!r}")

    segments_by_patient = load_segmented_reports(segmented_reports)
    ann = pd.read_csv(annotations, dtype=str, keep_default_na=False)
    for col in ("patient_id", "segment_idx", "hpo_code"):
        if col not in ann.columns:
            raise ValueError(f"annotations CSV has no '{col}' column: {annotations}")

    looks_curated = CURATED_MARKER in ann.columns
    if gold_source == "curated" and not looks_curated:
        raise ValueError(
            f"gold_source=curated but {annotations} has no '{CURATED_MARKER}' column — that is "
            f"the pre-curation annotations_confirmed.csv shape")
    if gold_source == "confirmed" and looks_curated:
        raise ValueError(
            f"gold_source=confirmed but {annotations} is a curated annotation table (it has "
            f"'{CURATED_MARKER}'). Its rows include the terms the curation policy excluded, so "
            f"reading it flat would score retrieval against annotations the gold rejects")

    if isinstance(exclude_location_how, str):
        exclude_location_how = [exclude_location_how] if exclude_location_how else []
    exclude_location_how = tuple(str(h).strip() for h in (exclude_location_how or ())
                               if str(h).strip())

    if looks_curated and gold_source != "confirmed":
        ann, prov = _load_curated(ann, annotations, require_location, exclude_location_how)
    else:
        ann = ann.copy()
        ann["segment_idx"] = ann["segment_idx"].astype(int)
        prov = {"gold_source": "confirmed", "n_candidates": len(ann)}

    patients = sorted(set(ann["patient_id"]) & set(segments_by_patient))
    dropped = sorted(set(ann["patient_id"]) - set(segments_by_patient))
    if dropped:
        print(f"  [WARN] {len(dropped)} annotated patients absent from the segmented "
              f"reports, skipped: {dropped}")
    ann = ann[ann["patient_id"].isin(patients)]

    # A segment index that doesn't exist in the report means the two files disagree
    # about the segmentation, the join would silently mislabel every rank below.
    for pid, group in ann.groupby("patient_id"):
        n_seg = len(segments_by_patient[pid])
        bad = group[group["segment_idx"] >= n_seg]
        if not bad.empty:
            raise ValueError(
                f"{pid} has annotations at segment_idx "
                f"{sorted(bad['segment_idx'].unique())} but only {n_seg} segments. "
                f"The annotation CSV and the segmented reports are out of sync."
            )

    prov.update({"n_patients": len(patients), "n_annotations": len(ann),
                 "n_annotated_hpos": int(ann["hpo_code"].nunique())})
    ann.attrs["gold"] = prov
    print(f"      {len(patients)} patients, {len(ann)} annotations, "
          f"{ann['hpo_code'].nunique()} distinct annotated HPOs "
          f"[{prov['gold_source']} gold]")
    return segments_by_patient, ann, patients


def build_query_set(
    tree: HPOTree, ann: pd.DataFrame
) -> tuple[dict[tuple[str, str], tuple[int, str]], list[str]]:
    """
    Every (patient, query HPO) the pipeline would issue: each annotated HPO plus all
    its ancestors.

    ``hop`` is the distance to the *nearest* annotated HPO of that patient, the most
    specific attribution available when several annotations share an ancestor.
    """
    annotated = sorted(set(ann["hpo_code"]))
    outside = [h for h in annotated if h not in tree.phenotypic_abnormalityNT]
    if outside:
        print(f"  [WARN] {len(outside)} annotated HPOs outside the phenotypic-abnormality "
              f"subtree, skipped: {outside[:5]}")

    hops_cache = {h: ancestors_with_hops(tree, h) for h in annotated}
    queries: dict[tuple[str, str], tuple[int, str]] = {}
    for pid, group in ann.groupby("patient_id"):
        for h in set(group["hpo_code"]):
            for q, hop in hops_cache.get(h, {}).items():
                key = (pid, q)
                if key not in queries or hop < queries[key][0]:
                    queries[key] = (hop, h)

    all_query_hpos = sorted({q for _, q in queries})
    print(f"      {len(all_query_hpos)} distinct query HPOs "
          f"({len(annotated)} annotated + ancestors), {len(queries)} (patient, HPO) pairs")
    return queries, all_query_hpos


def build_relevance(
    tree: HPOTree, ann: pd.DataFrame, queries: dict[tuple[str, str], tuple[int, str]]
) -> dict[tuple[str, str], dict[int, set[str]]]:
    """
    Ground truth under the **true-path rule**: (patient, query) →
    {relevant segment_idx → the annotations that imply the query}.
    """
    desc_cache = {q: descendants_or_self(tree, q) for _, q in queries}
    seg_anns: dict[tuple[str, int], set[str]] = {}
    for row in ann.itertuples():
        seg_anns.setdefault((row.patient_id, row.segment_idx), set()).add(row.hpo_code)

    relevant: dict[tuple[str, str], dict[int, set[str]]] = {}
    for (pid, q) in queries:
        d = desc_cache[q]
        relevant[(pid, q)] = {
            idx: (hpos & d)
            for (p, idx), hpos in seg_anns.items()
            if p == pid and (hpos & d)
        }
    print(f"      {sum(len(v) for v in relevant.values())} relevant "
          f"(patient, query HPO, segment) triples")
    return relevant


def read_context_sentences(context_dir: str, hpo: str) -> list[str]:
    """The synthetic-sentence sentences a query HPO is retrieved against, one per bullet line."""
    path = Path(context_dir) / f"{hpo.replace(':', '_')}.txt"
    if not path.is_file():
        return []
    from hpo_extraction.data.loading import _BULLET

    with open(path, encoding="latin1") as fh:
        return [_BULLET.sub("", line.strip()) for line in fh if line.strip()]


# ── Scoring ───────────────────────────────────────────────────────────────────

def max_pooled_sims(seg_emb: np.ndarray, ctx_emb: np.ndarray) -> np.ndarray:
    """
    Cosine similarity of every segment to its best-matching context sentence.

    Vectorised equivalent of ``SymptomScoreCalculator.text_2_context`` with
    ``mode="max"``, the configuration every ``experiments/*/run.py`` constructs.
    The pure-Python original is a double loop over ~22M pairs at this scale;
    ``verify_against_calculator`` asserts the two agree.
    """
    seg = seg_emb / np.linalg.norm(seg_emb, axis=1, keepdims=True)
    ctx = ctx_emb / np.linalg.norm(ctx_emb, axis=1, keepdims=True)
    return (seg @ ctx.T).max(axis=1)


def verify_against_calculator(context_dict, seg_emb: np.ndarray, hpo: str) -> None:
    """Assert the vectorised scorer reproduces the pipeline's own retriever."""
    from hpo_extraction.retrieval.similarity import SymptomScoreCalculator

    calc = SymptomScoreCalculator(context_dict)  # mode="max", top_n="all"
    reference = np.asarray(calc.text_2_context(list(seg_emb), hpo))
    fast = max_pooled_sims(seg_emb, context_dict[hpo])
    delta = float(np.abs(reference - fast).max())
    if delta > 1e-5:
        raise AssertionError(
            f"vectorised scorer diverges from SymptomScoreCalculator on {hpo}: "
            f"max |Δ| = {delta:.2e}"
        )
    print(f"      scorer check on {hpo}: max |Δ| = {delta:.2e}  [OK]")


def rank_descending(sims: np.ndarray) -> np.ndarray:
    """1-based ranks, highest score first, ties broken by segment order (deterministic)."""
    order = np.lexsort((np.arange(len(sims)), -sims))
    rank_of = np.empty(len(sims), dtype=int)
    rank_of[order] = np.arange(1, len(sims) + 1)
    return rank_of


# ── Metrics ───────────────────────────────────────────────────────────────────

def hop_bucket(hop: int) -> str:
    """Display bucket of a distance in the ontology (0 is the annotated term itself)."""
    if hop == 0:
        return "0 (annotated)"
    if hop >= MAX_HOP_BUCKET:
        return f"{MAX_HOP_BUCKET}+"
    return str(hop)


def bucket_order(labels) -> list[str]:
    """Sort hop buckets numerically, keeping '0 (annotated)' first and 'N+' last."""
    def key(b: str) -> float:
        return float(b.split()[0].rstrip("+")) + (0.5 if b.endswith("+") else 0.0)
    return sorted(set(labels), key=key)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """
    ROC-AUC via the Mann-Whitney U identity, probability that a random relevant
    segment outscores a random irrelevant one. Ties count as 0.5, and this avoids a
    scikit-learn dependency the project does not otherwise carry.
    """
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def as_metric_records(df: pd.DataFrame) -> list[dict]:
    """Reshape to the record schema that evaluation/segment_metrics.py expects."""
    return [
        {
            "patient_id": r.patient_id,
            "hpo_id": r.query_hpo,
            "rank": int(r.rank),
            "cosine_sim": float(r.cosine_sim),
            "is_gt_relevant": bool(r.is_gt_relevant),
        }
        for r in df.itertuples()
    ]


def metrics_by_hop(scores: pd.DataFrame) -> pd.DataFrame:
    """Recall@K, MRR and rank statistics for each hop bucket."""
    rows = []
    for bucket in bucket_order(scores["hop_bucket"]):
        sub = scores[scores["hop_bucket"] == bucket]
        recs = as_metric_records(sub)
        recall = compute_segment_recall_at_k(recs, k_values=K_VALUES)
        rel = sub[sub["is_gt_relevant"]]
        # One rank per (patient, query): the best-ranked relevant segment.
        best = rel.groupby(["patient_id", "query_hpo"])[["rank", "norm_rank"]].min()
        row = {
            "hop": bucket,
            "n_pairs": sub.groupby(["patient_id", "query_hpo"]).ngroups,
            "n_relevant": len(rel),
            "mrr": compute_mrr(recs),
            "median_rank": float(best["rank"].median()) if len(best) else float("nan"),
            "mean_rank": float(best["rank"].mean()) if len(best) else float("nan"),
            "median_norm_rank": float(best["norm_rank"].median()) if len(best) else float("nan"),
            "auc": auc(
                rel["cosine_sim"].to_numpy(),
                sub[~sub["is_gt_relevant"]]["cosine_sim"].to_numpy(),
            ),
        }
        row.update({f"recall@{k}": recall.get(k, float("nan")) for k in K_VALUES})
        rows.append(row)
    return pd.DataFrame(rows)


def per_hpo_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Per query-HPO Recall@1, mean rank, score separability and score scale."""
    rows = []
    for (hpo, label), sub in scores.groupby(["query_hpo", "query_label"]):
        recs = as_metric_records(sub)
        recall = compute_segment_recall_at_k(recs, k_values=[1, 5])
        rel = sub[sub["is_gt_relevant"]]
        best = rel.groupby(["patient_id", "query_hpo"])["rank"].min()
        rows.append({
            "query_hpo": hpo,
            "query_label": label,
            "hop": sub["hop"].min(),
            "n_pairs": sub.groupby(["patient_id", "query_hpo"]).ngroups,
            "recall_at_1": recall.get(1, float("nan")),
            "recall_at_5": recall.get(5, float("nan")),
            "mean_rank": float(best.mean()) if len(best) else float("nan"),
            "auc": auc(
                rel["cosine_sim"].to_numpy(),
                sub[~sub["is_gt_relevant"]]["cosine_sim"].to_numpy(),
            ),
            "mean_top1_sim": float(sub["top1_sim"].mean()),
            "mean_rel_sim": float(rel["cosine_sim"].mean()) if len(rel) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("recall_at_1")


def threshold_sweep(scores: pd.DataFrame, n: int = 60) -> pd.DataFrame:
    """
    For each τ: what fraction of relevant segments survives, and how many segments
    per (patient, HPO) pair does the pipeline still have to classify?

    This is the operational cost/recall curve a global cosine cutoff would buy.
    """
    sims = scores["cosine_sim"].to_numpy()
    rel_mask = scores["is_gt_relevant"].to_numpy()
    n_pairs = scores.groupby(["patient_id", "query_hpo"]).ngroups
    taus = np.linspace(float(sims.min()), float(sims.max()), n)
    return pd.DataFrame([
        {
            "tau": float(t),
            "recall": float((sims[rel_mask] >= t).mean()),
            "segments_per_pair": float((sims >= t).sum() / n_pairs),
        }
        for t in taus
    ])


def overall_metrics(scores: pd.DataFrame, by_hop: pd.DataFrame) -> dict:
    """
    Main metrics for MLflow. "Overall" = across every (patient, HPO) pair the
    pipeline actually queries, i.e. GT HPOs *and* their ancestors.
    """
    n_pairs = scores.groupby(["patient_id", "query_hpo"]).ngroups
    rel = scores[scores["is_gt_relevant"]]
    irr = scores[~scores["is_gt_relevant"]]
    metrics = {
        f"overall_recall_at_{k}": scores[scores["is_gt_relevant"] & (scores["rank"] <= k)]
        .groupby(["patient_id", "query_hpo"]).ngroups / n_pairs
        for k in K_VALUES
    }
    metrics["overall_auc"] = auc(rel["cosine_sim"].to_numpy(), irr["cosine_sim"].to_numpy())
    metrics["mean_segments_per_report"] = len(scores) / n_pairs
    # itertuples() would mangle "recall@5" into a positional field, index by name.
    for _, row in by_hop.iterrows():
        tag = str(row["hop"]).split()[0].replace("+", "plus")
        metrics[f"hop{tag}_recall_at_5"] = float(row["recall@5"])
        metrics[f"hop{tag}_mrr"] = float(row["mrr"])
        metrics[f"hop{tag}_auc"] = float(row["auc"])
    return metrics


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_rank_by_hop(by_hop: pd.DataFrame, path: Path) -> None:
    """Plot the rank of the evidence segment against the distance of the query term, and save it to *path*."""
    # Hop buckets are categorical ("0 (annotated)" … "4+"), so plot against position
    # and label the ticks, a numeric axis would imply "4+" is 4.
    x = np.arange(len(by_hop))
    labels = list(by_hop["hop"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    for k in K_VALUES:
        ax1.plot(x, by_hop[f"recall@{k}"], marker="o", label=f"Recall@{k}")
    ax1.plot(x, by_hop["mrr"], marker="s", linestyle="--", color="black", label="MRR")
    ax1.set_title("Retrieval quality vs ontology hop distance", fontsize=13)
    ax1.set_ylabel("Recall / MRR", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)

    ax2.plot(x, by_hop["median_rank"], marker="o", color="#d62728", label="median rank")
    ax2.plot(x, by_hop["mean_rank"], marker="s", color="#ff7f0e", label="mean rank")
    ax2.set_title("Rank of the annotated sentence vs hop distance", fontsize=13)
    ax2.set_ylabel("Rank of first relevant segment", fontsize=11)
    ax2.legend(fontsize=9)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Hops from the annotated HPO", fontsize=11)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rank_distribution_by_hop(records: pd.DataFrame, path: Path, max_rank: int = 20) -> None:
    """Plot the distribution of the best evidence rank per distance bucket (ranks up to *max_rank*), and save it."""
    buckets = bucket_order(records["hop_bucket"])
    fig, axes = plt.subplots(len(buckets), 1, figsize=(9, 2.3 * len(buckets)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, bucket in zip(axes, buckets):
        best = (records[records["hop_bucket"] == bucket]
                .groupby(["patient_id", "query_hpo"])["rank"].min())
        counts = [int((best == r).sum()) for r in range(1, max_rank + 1)]
        beyond = int((best > max_rank).sum())
        labels = [str(r) for r in range(1, max_rank + 1)] + [f">{max_rank}"]
        colors = ["#1f77b4"] * max_rank + ["#d62728"]
        ax.bar(labels, counts + [beyond], color=colors, alpha=0.85)
        ax.set_ylabel("pairs", fontsize=10)
        ax.set_title(f"hop {bucket}  (n={len(best)} pairs)", fontsize=11, loc="left")
        ax.grid(alpha=0.3, axis="y")

    axes[-1].set_xlabel("Rank of the first relevant segment", fontsize=11)
    fig.suptitle("Where the annotated sentence lands, by hop distance", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_similarity_distribution(scores: pd.DataFrame, path: Path) -> None:
    """Plot cosine similarities of evidence and other segments with a Kolmogorov-Smirnov test, and save it."""
    rel = scores[scores["is_gt_relevant"]]["cosine_sim"].to_numpy()
    irr = scores[~scores["is_gt_relevant"]]["cosine_sim"].to_numpy()
    ks_stat, p_val = ks_2samp(rel, irr)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(scores["cosine_sim"].min(), 0), scores["cosine_sim"].max(), 50)
    ax.hist(irr, bins=bins, alpha=0.55, density=True, color="#d62728",
            label=f"irrelevant (n={len(irr)})")
    ax.hist(rel, bins=bins, alpha=0.55, density=True, color="#2ca02c",
            label=f"GT-relevant (n={len(rel)})")
    ax.set_title(f"Cosine similarity, relevant vs irrelevant segments\n"
                 f"AUC={auc(rel, irr):.3f}   KS={ks_stat:.3f} (p={p_val:.2e})", fontsize=13)
    ax.set_xlabel("Cosine similarity", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_similarity_by_hop(scores: pd.DataFrame, path: Path) -> None:
    """Plot cosine similarities of evidence and other segments per distance bucket, and save it."""
    buckets = bucket_order(scores["hop_bucket"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    rel_data = [scores[(scores["hop_bucket"] == b) & scores["is_gt_relevant"]]["cosine_sim"]
                for b in buckets]
    irr_data = [scores[(scores["hop_bucket"] == b) & ~scores["is_gt_relevant"]]["cosine_sim"]
                for b in buckets]
    pos = np.arange(len(buckets))
    for data, offset, color, label in (
        (rel_data, -0.18, "#2ca02c", "GT-relevant"),
        (irr_data, 0.18, "#d62728", "irrelevant"),
    ):
        bp = ax1.boxplot(data, positions=pos + offset, widths=0.3, patch_artist=True,
                         showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.plot([], [], color=color, linewidth=6, alpha=0.7, label=label)
    ax1.set_xticks(pos)
    ax1.set_xticklabels(buckets)
    ax1.set_title("Similarity score by hop distance", fontsize=13)
    ax1.set_xlabel("Hops from the annotated HPO", fontsize=11)
    ax1.set_ylabel("Cosine similarity", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, axis="y")

    rel = scores[scores["is_gt_relevant"]]
    margins = [rel[rel["hop_bucket"] == b]["margin_to_top1"] for b in buckets]
    bp = ax2.boxplot(margins, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.7)
    ax2.set_xticks(np.arange(1, len(buckets) + 1))
    ax2.set_xticklabels(buckets)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_title("How far the annotated sentence sits below rank 1", fontsize=13)
    ax2.set_xlabel("Hops from the annotated HPO", fontsize=11)
    ax2.set_ylabel("GT score − top-1 score", fontsize=11)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_similarity_by_rank(scores: pd.DataFrame, path: Path, max_rank: int = 20) -> None:
    """Plot the cosine similarity at each retrieval rank up to *max_rank*, and save it."""
    sub = scores[scores["rank"] <= max_rank]
    ranks = sorted(sub["rank"].unique())
    data = [sub[sub["rank"] == r]["cosine_sim"] for r in ranks]

    fig, ax = plt.subplots(figsize=(max(8, len(ranks) * 0.5), 5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.7)
    ax.set_xticks(np.arange(1, len(ranks) + 1))
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_title("Cosine similarity by rank position", fontsize=13)
    ax.set_xlabel("Rank", fontsize=11)
    ax.set_ylabel("Cosine similarity", fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cross_hpo_scale(per_hpo: pd.DataFrame, path: Path) -> None:
    """
    Whether a single global cosine threshold can work at all.

    If the per-HPO top-1 scores span a wide band, one τ is simultaneously too strict
    for some HPOs and too loose for others, and per-HPO thresholds are the only fix.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.hist(per_hpo["mean_top1_sim"].dropna(), bins=30, color="#1f77b4", alpha=0.85)
    lo, hi = per_hpo["mean_top1_sim"].min(), per_hpo["mean_top1_sim"].max()
    ax1.set_title(f"Mean top-1 similarity per query HPO\n"
                  f"spread: {lo:.2f} – {hi:.2f}  (σ={per_hpo['mean_top1_sim'].std():.3f})",
                  fontsize=13)
    ax1.set_xlabel("Mean top-1 cosine similarity", fontsize=11)
    ax1.set_ylabel("Number of HPOs", fontsize=11)
    ax1.grid(alpha=0.3, axis="y")

    valid = per_hpo.dropna(subset=["auc"])
    ax2.scatter(valid["mean_top1_sim"], valid["auc"], c=valid["hop"],
                cmap="viridis", alpha=0.75, s=28)
    ax2.axhline(0.5, color="red", linestyle="--", linewidth=1, label="chance")
    ax2.set_title("Score separability vs score scale, per HPO", fontsize=13)
    ax2.set_xlabel("Mean top-1 cosine similarity", fontsize=11)
    ax2.set_ylabel("ROC-AUC (relevant vs irrelevant)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    fig.colorbar(ax2.collections[0], ax=ax2, label="hop")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_threshold_sweep(sweep: pd.DataFrame, path: Path) -> None:
    """Plot recall of evidence segments and segments kept against a cosine threshold, and save it."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sweep["tau"], sweep["recall"], color="#2ca02c", label="recall of GT segments")
    ax1.set_xlabel("Cosine similarity threshold τ", fontsize=11)
    ax1.set_ylabel("Recall of GT segments", color="#2ca02c", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#2ca02c")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(sweep["tau"], sweep["segments_per_pair"], color="#1f77b4",
             label="segments kept per (patient, HPO)")
    ax2.set_ylabel("Segments surviving per (patient, HPO)", color="#1f77b4", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    ax1.set_title("Cost / recall of a global cosine cutoff", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_hpo_recall(per_hpo: pd.DataFrame, path: Path, n: int = 25) -> None:
    """Plot the *n* query terms with the lowest Recall@1, and save it."""
    worst = per_hpo.nsmallest(n, "recall_at_1")
    labels = [f"{r.query_label[:38]} (hop {r.hop}, n={r.n_pairs})" for r in worst.itertuples()]

    fig, ax = plt.subplots(figsize=(9, max(4, len(worst) * 0.32)))
    ax.barh(labels, worst["recall_at_1"], color="#d62728", alpha=0.85)
    ax.set_title(f"{n} worst query HPOs by Recall@1", fontsize=13)
    ax.set_xlabel("Recall@1", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
