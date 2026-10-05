"""
Query-set variants for the retrieval-curve analysis, five ways to represent the query term, on one cost axis.

The question
------------
Retrieval here scores a report sentence against a *representation of the query HPO*. Today that
representation is ``U_v``: ~36 synthetic-sentence sentences an LLM generated for term ``v``
(``resources/context_data/context_HCY_llama_api``). An earlier exploratory run showed a second one, pool ``U_v``
with every descendant's, is better. Neither has ever been compared against a representation that
needs **no generator model at all**, nor against **chance**, and without those two evidence locations the
synthetic-sentence pool's value is unmeasured: it could be doing the work of an ontology lookup, or of
nothing.

Five conditions, one scoring rule
---------------------------
Every condition supplies a set of vectors ``R(v)`` for the query term, and the similarity of a segment
is the **same max-pooled cosine** in all five::

    sim(segment, v) = max over r in R(v) of cos(segment, r)

===  ======================================================  ==============================
ID   R(v)                                                    vectors per term
===  ======================================================  ==============================
R1   ``U_v``, the term's own synthetic-sentence sentences             ~36 (the shipped system)
R1u  ``U_v`` union every ``U_y``, y in desc_closure(v)       up to ~650k
R3   ``embed(label + " " + definition + " " + synonyms)``    1 (no generator model)
R3u  R3(v) union R3(y), y in desc_closure(v)                 1 per pooled term
R5   nothing, the report's sentences in random order        0 (control, closed form)
===  ======================================================  ==============================

R1/R1u and R3/R3u are a 2x2: representation (LLM synthetic-sentence corpus vs ontology text) crossed with
pooling (the term alone vs its whole descendant closure), over the **identical** node set. That is
what separates the closure's contribution from the representation's -- ``R1u - R1`` and
``R3u - R3`` are the same intervention measured on two different indices.

R1 and R1u are an earlier exploratory run's ``k0`` and ``kinf`` under new names, and that is deliberate: they are
the built-in controls. ``run_compute`` asserts R1 reproduces ``SymptomScoreCalculator`` (so it *is*
the pipeline's retrieval), that each ``u`` condition is >= its base everywhere (a max over a superset
cannot shrink) and that the two coincide wherever the closure pool is the own term alone.

R5 is not simulated. Its hit rate is the exact expectation, so it carries no Monte-Carlo noise and
needs no index, see :func:`r5_topk_hit`.

The gate, and why the x axis is what it is
------------------------------------------
Ranking is not the deliverable. The **gate** is, the rule deciding which of a patient's sentences
are forwarded to the SLM for one query term. Two gates, in the encoding
``experiments/figures/style.py`` already fixes for the an earlier exploratory run family:

``top-k``      keep the k best-ranked sentences. k is an integer, so only the marked points exist.
``global tau`` keep sentences with ``sim >= tau``, **one cutoff shared by every query**,
               continuous in tau.

Their natural parameters are not comparable (a rank position is not a cosine) and are not
comparable *across conditions* either, because the four index conditions put their scores on different
scales. So both gates and all five conditions are plotted against the one quantity every one of them
controls and that a reader can price: the **mean number of sentences that survive the gate per
(patient, query) pair**. At k=5 that is ~4.6 rather than 5, because short reports run out of
sentences, which is the point of measuring it, not assuming it.

What is on the y axis
---------------------
**Pair-level hit rate**: the fraction of (patient, query term) pairs for which *at least one*
ground truth sentence survives the gate. This is the quantity that bounds the pipeline, because
``AnyYesAggregator`` calls a term positive on a single Yes, a second ground truth sentence for a pair
already covered buys nothing downstream. It is the same definition ``fig6`` uses, not the
segment-level recall of ``threshold_sweep.csv``.

Figure A takes the **annotated terms** (hop 0, the terms the curator actually annotated).
Figure B takes the **ancestor closure** (each annotated term *and* every ancestor of it in the
phenotypic-abnormality subtree), faceted by the query term's own **absolute depth** below
``HP:0000118``, band 1 is an organ-system node, band 6+ is a specific finding. Relevance in both
follows the true-path rule: a segment is relevant for ``q`` if it is annotated with ``q`` or with
any descendant of ``q``.

Uncertainty
-----------
Report-clustered nonparametric bootstrap, 10 000 draws, ``hpo_extraction.evaluation.stats.bootstrap``'s own
resampler so the interval on every condition comes from the **identical** draws. The sampling unit is
the **patient**, never the annotation: a patient with a severe presentation supplies a dozen ground truth
terms that stand or fall together, and both the numerator and the denominator are resampled with
them. Both axes move under resampling. The CSV carries an interval for each, the figures band the
hit rate.

Stages (driven by ``run.py``)
-----------------------------
``compute``   score every (condition, patient, query, segment) -> ``qs_scores.csv.gz``  (this module)
``report``    gate curves, bootstrap intervals, Figure A, Figure B  (:mod:`gate_curves`)

``report`` reads only ``qs_scores.csv.gz``, so ``stages=[report]`` re-plots locally without the
model or the context directory. The names and grids both stages share are defined here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hpo_extraction.retrieval.descendant_closure import build_union_members  # noqa: E402
# Re-exported: run.py reads R3_TEMPLATE, and both names are part of this module's public
# surface for anything that wants the retrieval-curve analysis's R3 definition.
from hpo_extraction.retrieval.term_information_index import (  # noqa: E402,F401
    R3_SYNONYM_JOIN,
    R3_TEMPLATE,
    build_r3_index,
    r3_text,
)
from hpo_extraction.evaluation.retrieval_analysis import (  # noqa: E402
    DASHBOARD_TOP_N,
    build_query_set,
    build_relevance,
    label_of,
    load_inputs,
    max_pooled_sims,
    rank_descending,
    verify_against_calculator,
)
from hpo_extraction.ontology.hpo_tree import HPO_class, HPOTree  # noqa: E402

# ── Artifacts ─────────────────────────────────────────────────────────────────

SCORES_FILE = "qs_scores.csv.gz"
PAIRS_FILE = "qs_pairs.csv"
OPERATING_FILE = "operating_points.csv"
DELTAS_FILE = "paired_deltas.csv"
GOLD_FILE = "gold_provenance.json"
R3_FILE = "r3_provenance.json"
SEGMENTS_JSONL = "retrieval_retrieved_segments.jsonl"
SUMMARY_FILE = "query_set_analysis.md"
FIG_A = "figA_terminal_hit"
FIG_B = "figB_ancestor_closure"
FIG_C = "figC_all_queries"

#: Figure C's single facet: every (patient, query) pair the pipeline would issue, the annotated terms
#: *and* their ancestors, pooled. It is the union of Figure A's facet with Figure B's five, so the
#: three figures partition nothing and overlap by design: A is the terminal terms alone, B splits
#: The whole set by depth, C is that same whole set undivided. Quoting a C number as if it were an
#: A number is the one misreading to guard against, which is why it has its own figure id in
#: ``operating_points.csv``, not riding along as a sixth depth band.
FACET_ALL = "gold + ancestors"

# ── The conditions ──────────────────────────────────────────────────────────────────

#: Conditions that own an index and therefore appear in ``qs_scores.csv.gz``. R5 is closed form.
#: The two ``u`` conditions pool over the **same** closure as their base condition, so R1 vs R1u and R3 vs R3u
#: are the same intervention applied to two different representations.
SCORED_CONDITIONS = ("R1", "R1u", "R3", "R3u")
CONDITIONS = ("R1", "R1u", "R3", "R3u", "R5")

CONDITION_NOTE = {
    "R1": "the shipped system: the term's own LLM-generated exemplar sentences",
    "R1u": "exp00_05's kinf: the exemplar sentences of the term and its whole descendant closure",
    "R3": "one vector straight from the ontology — no generator model, no per-term corpus",
    "R3u": "R3 pooled over the same descendant closure as R1u — one vector per term, still no "
           "generator model",
    "R5": "chance: the report's own sentences in random order. Closed form, no index",
}

#: The base condition each ``u`` condition is the closure of. ``run_compute`` asserts the pair is ordered
#: (a max over a superset cannot shrink) and identical wherever the pool is a singleton.
CLOSURE_OF = {"R1u": "R1", "R3u": "R3"}

#: The exact string R3 embeds lives in ``hpo_extraction.retrieval.term_information_index`` and is re-exported here, because
#: ``treephenorag_scores_terminfo`` builds a TreePhenoRAG traversal on the same index. Two copies that drifted would
#: leave the two experiments incomparable while both still looked correct.

# ── Gates ─────────────────────────────────────────────────────────────────────

#: Integer k values drawn as points on the top-k curve. Only these exist. The joining line is a
#: guide, not a set of reachable settings.
K_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60)

#: Number of points on each continuous curve (global tau, and R5's Bernoulli analogue).
N_CONTINUOUS = 120

#: The x axis stops here. The ungated pool is ~37 sentences per pair. Drawing that far spends most
#: of the axis in a region where every condition has saturated, so the cap is stated in the summary
#: instead.
X_MAX = 25.0

#: The window the pipeline actually ships (``top_n=5``).
SHIPPED_K = 5

# ── Depth bands (Figure B) ────────────────────────────────────────────────────

#: ``tree.depth_dict`` is BFS depth from ``HP:0000118`` itself (depth 0), so a query term's value
#: is already its depth *below* Phenotypic abnormality, band 1 is an organ-system node.
DEPTH_BANDS = ("1", "2", "3", "4-5", "6+")


def depth_band(depth: int) -> str:
    """Absolute ontology depth -> the band it is reported in."""
    if depth <= 1:
        return "1"
    if depth <= 3:
        return str(depth)
    if depth <= 5:
        return "4-5"
    return "6+"


# ── R3: the ontology-only representation ─────────────────────────────

# ``r3_text`` and ``build_r3_index`` are imported from :mod:`hpo_extraction.retrieval.term_information_index` above. They were
# defined here until the term-information score store needed to run a whole traversal on this index. Promoting them was
# The only way to make "the same R3" a checkable fact, not a claim.


# ── R5: chance, in closed form ────────────────────────────────────────────────

def r5_topk_hit(n_segments, n_relevant, k: int) -> np.ndarray:
    """P(at least one relevant sentence in a uniform random draw of ``min(k, n)`` sentences).

    ``1 - C(n-g, k') / C(n, k')`` evaluated as the running product
    ``prod_{t<k'} (n-g-t)/(n-t)``, which is exact in float64 at these sizes and needs no factorial.

    This is an *expectation*, so the condition carries no Monte-Carlo noise: the only uncertainty in its
    curve is the cohort's, which is what the bootstrap is for.
    """
    n = np.asarray(n_segments, dtype=np.float64)
    g = np.asarray(n_relevant, dtype=np.float64)
    if n.size == 0:
        return np.zeros(0)
    k_eff = np.minimum(float(k), n)
    miss = np.ones_like(n)
    for t in range(int(k_eff.max())):
        active = t < k_eff
        num = np.maximum(n - g - t, 0.0)
        den = np.maximum(n - t, 1.0)
        miss = np.where(active, miss * num / den, miss)
    return 1.0 - miss


def r5_bernoulli_hit(n_relevant, p: float) -> np.ndarray:
    """P(at least one relevant sentence) when each sentence is kept independently w.p. ``p``.

    The continuous analogue of the global-tau gate for a scorer that carries no information: an
    absolute cutoff on a meaningless score keeps a random *fraction* of the sentences, not
    a fixed count. Drawn dashed, like every other condition's tau curve.
    """
    return 1.0 - (1.0 - p) ** np.asarray(n_relevant, dtype=np.float64)


# ── Stage: compute ────────────────────────────────────────────────────────────

def run_compute(
    segmented_reports: str,
    annotations: str,
    context_dir: str,
    sent_transformer_dir: str,
    output_dir: str | Path,
    hpo_json: str | None = None,
    gold_source: str = "auto",
    require_location: bool = True,
    exclude_location_how: tuple = (),
) -> dict:
    """Score every (condition, patient, query HPO, segment). Returns summary counts."""
    from sentence_transformers import SentenceTransformer

    from hpo_extraction.retrieval.descendant_closure import UnionScorer
    from hpo_extraction.data.loading import LazyContextDict

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading segments and the curated gold...")
    segments_by_patient, ann, patients = load_inputs(
        segmented_reports, annotations, gold_source=gold_source,
        require_location=require_location, exclude_location_how=exclude_location_how,
    )
    gold_prov = dict(ann.attrs.get("gold", {}))

    print("[2/7] Expanding annotated HPOs to their ancestor closure...")
    tree = HPOTree(hpo_json_path=hpo_json) if hpo_json else HPOTree()
    tree.buildHPOTree()
    queries, all_query_hpos = build_query_set(tree, ann)

    print("[3/7] Propagating the gold up the ontology (true-path rule)...")
    relevant = build_relevance(tree, ann, queries)

    print("[4/7] Building the descendant closures (R1u)...")
    model = SentenceTransformer(sent_transformer_dir)
    context_dict = LazyContextDict(context_dir, model)
    available = set(context_dict.keys())

    missing = [q for q in all_query_hpos if q not in available]
    if missing:
        raise ValueError(
            f"{len(missing)} query HPOs have no context file in {context_dir} — R1 is undefined "
            f"for them and the arm would be measuring corpus coverage: {missing[:10]}"
        )

    members = build_union_members(all_query_hpos, tree, available, union_depth=-1)
    bad = [q for q, pool in members.items() if q not in pool]
    if bad:
        raise AssertionError(f"R1u pool does not contain the query itself for {bad[:5]}")
    # A singleton pool is the query's own context and nothing else, so R1u must equal R1 on it.
    # That is a slightly wider set than the ontology leaves, it also catches a node whose whole
    # subtree happens to have no context file, and it is the right invariant: it is a property of
    # The pool actually scored, not of the DAG.
    singleton = {q for q, pool in members.items() if len(pool) == 1}
    pool_sizes = [len(p) for p in members.values()]
    print(f"      {len(all_query_hpos)} query terms, {len(singleton)} with a singleton pool; "
          f"R1u pool per query — median {int(np.median(pool_sizes))}, max {max(pool_sizes)}")

    print("[5/7] Encoding segments and building the two indices...")
    seg_emb = {pid: np.asarray(model.encode(segments_by_patient[pid]), dtype=np.float32)
               for pid in patients}

    # R1 is the pipeline's own retrieval, not an approximation of it.
    verify_against_calculator(context_dict, seg_emb[patients[0]], all_query_hpos[0])

    synthetic_scorer = UnionScorer(context_dict, members)
    synthetic_scorer.prepare()
    universe = synthetic_scorer.universe
    print(f"      closure universe: {len(universe)} HPOs, "
          f"{sum(synthetic_scorer.n_context_sentences.values())} exemplar sentences")

    # The ontology index over the SAME universe, wrapped so it is one "context sentence" per term.
    # UnionScorer then gives R3 as own_sims and R3u as query_sims, with the identical
    # associative-max reduction it gives R1/R1u, so the two representations are compared through
    # one code path, not two, and R3u's pool is R1u's pool.
    r3_vectors, r3_prov = build_r3_index(
        tree, universe, lambda texts: model.encode(texts, show_progress_bar=False)
    )
    print(f"      R3: {r3_prov['n_terms']} terms, {r3_prov['n_without_definition']} without a "
          f"definition, {r3_prov['n_label_only']} label-only, "
          f"median {r3_prov['median_chars']} chars")
    r3_dict = {h: r3_vectors[i][None, :] for i, h in enumerate(universe)}
    ontology_scorer = UnionScorer(r3_dict, members)
    ontology_scorer.prepare()
    r3_prov["universe"] = len(universe)
    r3_prov["n_query_terms"] = len(all_query_hpos)

    scorers = {"R1": synthetic_scorer, "R1u": synthetic_scorer,
               "R3": ontology_scorer, "R3u": ontology_scorer}

    print("[6/7] Scoring every (arm, patient, query, segment)...")
    queries_by_patient: dict[str, list[str]] = {}
    for (pid, q) in queries:
        queries_by_patient.setdefault(pid, []).append(q)

    score_rows: list[dict] = []
    t0 = time.time()
    checked_own = False
    for i, pid in enumerate(patients):
        synthetic_scorer.score_patient(seg_emb[pid])
        ontology_scorer.score_patient(seg_emb[pid])
        for q in sorted(queries_by_patient.get(pid, ())):
            hop, source = queries[(pid, q)]
            sims = {
                arm: (scorers[arm].query_sims(q) if arm in CLOSURE_OF
                      else scorers[arm].own_sims(q)).astype(np.float64)
                for arm in SCORED_CONDITIONS
            }

            if not checked_own:
                # own_sims must be the max-pooled cosine against the query's OWN context, which is
                # what verify_against_calculator just fixed to SymptomScoreCalculator.
                delta = float(np.abs(
                    sims["R1"] - max_pooled_sims(seg_emb[pid], context_dict[q])).max())
                if delta > 1e-5:
                    raise AssertionError(
                        f"R1 (UnionScorer.own_sims) diverges from the pipeline's max-pooled "
                        f"cosine on {pid}/{q}: max |delta| = {delta:.2e}")
                print(f"      R1 == SymptomScoreCalculator on {pid}/{q}: "
                      f"max |delta| = {delta:.2e}  [OK]")
                checked_own = True

            for closed, base in CLOSURE_OF.items():
                # A max over a superset cannot shrink.
                if (sims[closed] < sims[base] - 1e-6).any():
                    raise AssertionError(
                        f"{closed} < {base} for {pid}/{q}: the closure pool is a superset of the "
                        f"own pool, so its max cannot be lower")
                # ...and where the pool is the own term alone, the two are the same set.
                if q in singleton and float(np.abs(sims[closed] - sims[base]).max()) > 1e-6:
                    raise AssertionError(
                        f"{q} has a singleton union pool but {closed} differs from {base}")

            rel = relevant[(pid, q)]
            depth = int(tree.depth_dict.get(q, -1))
            for arm in SCORED_CONDITIONS:
                s = sims[arm]
                rank_of = rank_descending(s)
                for idx in range(len(s)):
                    score_rows.append({
                        "arm": arm,
                        "patient_id": pid,
                        "query_hpo": q,
                        "hop": hop,
                        "depth": depth,
                        "source_hpo": source,
                        "segment_idx": idx,
                        "rank": int(rank_of[idx]),
                        "n_segments": len(s),
                        "cosine_sim": float(s[idx]),
                        "is_gt_relevant": idx in rel,
                    })
        if (i + 1) % 20 == 0:
            print(f"      {i + 1}/{len(patients)} patients ({time.time() - t0:.0f}s)")

    scores = pd.DataFrame(score_rows)
    scores["depth_band"] = scores["depth"].map(depth_band)
    scores["query_label"] = scores["query_hpo"].map(lambda q: label_of(tree, q))
    print(f"      {len(scores)} score rows in {time.time() - t0:.0f}s")

    # Every query exists because some annotation of that patient implies it, so the true-path rule
    # must give it at least one relevant segment. A pair with none would sit in the denominator of
    # every hit rate below and never be hittable.
    per_pair_rel = scores.groupby(["arm", "patient_id", "query_hpo"])["is_gt_relevant"].sum()
    if (per_pair_rel == 0).any():
        empty = per_pair_rel[per_pair_rel == 0]
        raise AssertionError(
            f"{len(empty)} (patient, query) pairs have no relevant segment under the true-path "
            f"rule: {list(empty.index[:5])}")

    print("[7/7] Writing artifacts...")
    scores.to_csv(out_dir / SCORES_FILE, index=False, compression="gzip")
    (out_dir / GOLD_FILE).write_text(json.dumps(gold_prov, indent=2, default=str))
    (out_dir / R3_FILE).write_text(json.dumps(r3_prov, indent=2, default=str))

    # Dashboard artifact: the shipped condition, truncated to the pipeline's own window.
    # No LLM runs here, so no slm_verdict and no predictions file.
    top = scores[(scores["arm"] == "R1") & (scores["rank"] <= DASHBOARD_TOP_N)]
    with open(out_dir / SEGMENTS_JSONL, "w") as fh:
        for r in top.itertuples():
            fh.write(json.dumps({
                "patient_id": r.patient_id,
                "hpo_id": r.query_hpo,
                "hpo_label": r.query_label,
                "rank": int(r.rank),
                "text": segments_by_patient[r.patient_id][r.segment_idx],
                "cosine_sim": round(float(r.cosine_sim), 6),
                "slm_verdict": None,
                "is_gt_relevant": bool(r.is_gt_relevant),
                "hop": int(r.hop),
                "depth": int(r.depth),
            }) + "\n")

    print(f"\n  -> {out_dir / SCORES_FILE}\n  -> {out_dir / GOLD_FILE}"
          f"\n  → {out_dir / R3_FILE}\n  → {out_dir / SEGMENTS_JSONL}")
    return {
        "n_patients": len(patients),
        "n_annotations": len(ann),
        "n_query_hpos": len(all_query_hpos),
        "n_singleton_pool_queries": len(singleton),
        "n_pairs": len(queries),
        "n_gold_pairs": sum(1 for key in queries if queries[key][0] == 0),
        "n_score_rows": len(scores),
        "n_ctx_hpos": len(universe),
        "n_ctx_sentences": int(sum(synthetic_scorer.n_context_sentences.values())),
        "r3_n_label_only": r3_prov["n_label_only"],
    }
