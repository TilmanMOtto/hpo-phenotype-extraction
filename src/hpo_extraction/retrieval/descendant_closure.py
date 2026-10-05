"""Descendant-union context retrieval, scored via the associative-max trick.

**What the union is.** A generic ancestor ("Abnormality of the nervous system") has generic
synthetic-sentence sentences, while the patient sentence that should match it is specific ("tonic-clonic
seizures"), and that specific sentence lives in a *descendant's* context file. An earlier exploratory run
established the fix at the retrieval stage: pool an HPO's synthetic-sentence sentences with those of its
descendants. At ``kinf`` top-10 that reached R@10 = 0.947, the best retrieval number in the
project.

**Why this module exists rather than reusing an earlier exploratory run's ``UnionContextDict``.** That class
vstacks each query's union pool and hands it to ``SymptomScoreCalculator``, whose ``max`` mode is
a Python double loop over every (segment, context-sentence) pair. On a node whose subtree has
~19k descendants that is millions of Python-level cosine calls per patient, and it never evicts
the vstacked pools. Unusable for a full-ontology scan.

**The associative-max trick** (from an earlier exploratory run's own compute path) makes it cheap. The union score
of a segment against query ``q`` is

    max over m in members(q), over s in sentences(m)  cos(segment, s)

and ``max`` is associative, so it factors:

    own_max[segment, m] = max over s in sentences(m) cos(segment, s)     # computed once
    union[segment, q]   = max over m in members(q)  own_max[segment, m]  # a column-max

The expensive part is computed once per *context* HPO, not once per (query, patient), and
each query then costs a masked max over columns. Identical results, asserted against
``SymptomScoreCalculator`` by ``verify_against_calculator``.
"""

from __future__ import annotations

import logging

import numpy as np

from hpo_extraction.evaluation.retrieval_analysis import descendants_with_hops

logger = logging.getLogger(__name__)


def union_members(desc_hops: dict[str, int], k: int, available: set[str]) -> list[str]:
    """Context HPOs forming the pool for one query at depth cap ``k`` (``k < 0`` = unbounded).

    ``desc_hops`` includes the query itself at distance 0, so ``k=0`` yields the query's own
    context, the built-in control that must reproduce non-union retrieval. Lifted from
    ``an earlier exploratory run/union_ranks.py`` so both consumers agree.
    """
    return sorted(
        d for d, dist in desc_hops.items()
        if (k < 0 or dist <= k) and d in available
    )


def build_union_members(
    query_hpos, tree, available: set[str], union_depth: int = -1
) -> dict[str, list[str]]:
    """``{query → context pool}`` for every query HPO.

    ``available`` is the set of HPOs whose embeddings can actually be produced. A descendant with
    no context sentences is silently dropped. A query with an empty pool would be unscorable, so
    it falls back to itself and is logged, that is a data problem worth seeing, not one to
    swallow.
    """
    members: dict[str, list[str]] = {}
    for q in query_hpos:
        pool = union_members(descendants_with_hops(tree, q), union_depth, available)
        if not pool:
            logger.warning("empty union pool for %s; falling back to own context", q)
            pool = [q] if q in available else []
        members[q] = pool
    return members


class UnionScorer:
    """Union-context similarity for many queries against one patient's segments.

    Usage per patient::

        scorer = UnionScorer(context_dict, members)      # once, builds the column index
        scorer.score_patient(seg_emb)                    # once per patient
        sims = scorer.query_sims(hpo_id)                 # cheap, per node

    ``score_patient`` holds an ``n_segments × n_context_hpos`` float32 matrix, a few MB for a
    clinical report, and ``query_sims`` is a column-max over it.
    """

    def __init__(self, context_dict, members: dict[str, list[str]]):
        self._context_dict = context_dict
        self._members = members
        self._universe = sorted({m for pool in members.values() for m in pool})
        self._col_of = {h: i for i, h in enumerate(self._universe)}
        self._cols = {
            q: np.array([self._col_of[m] for m in pool], dtype=np.int64)
            for q, pool in members.items()
        }
        self._own_max: np.ndarray | None = None
        self._ctx_matrix: np.ndarray | None = None
        self._group_starts: np.ndarray | None = None
        self.n_context_sentences: dict[str, int] = {}

    @property
    def universe(self) -> list[str]:
        """The distinct context HPOs whose embeddings this scorer needs."""
        return self._universe

    def pool_size(self, hpo_id: str) -> int:
        """Number of terms whose sentences are pooled for *hpo_id*."""
        return len(self._members[hpo_id])

    def context_sentence_count(self, hpo_id: str) -> int:
        """Total context sentences backing a query's union pool."""
        self.prepare()
        return int(sum(self.n_context_sentences.get(m, 0) for m in self._members[hpo_id]))

    def prepare(self) -> None:
        """Stack every context sentence into one normalised matrix, with group boundaries.

        Without this, ``score_patient`` does one small matmul per context HPO. A full shard's
        union spans most of the ontology (~18k HPOs), so that is ~2.4M tiny matmuls per shard, dominated by per-call overhead, not arithmetic. Stacking turns it into a single
        large matmul plus a segmented max, which is what BLAS is for.

        Idempotent; ``score_patient`` calls it on first use.
        """
        if self._ctx_matrix is not None:
            return
        blocks, starts, cursor = [], [], 0
        width = None
        for h in self._universe:
            ctx = np.asarray(self._context_dict[h], dtype=np.float32)
            if ctx.ndim == 1:
                ctx = ctx[None, :]
            if width is None:
                width = ctx.shape[1]
            elif ctx.shape[1] != width:
                # A stale/wrong-model embedding cache slipped through. Name the culprit rather
                # than letting np.vstack raise an anonymous dimension error. LazyContextDict now
                # self-heals such caches, so this should only fire for a hand-injected embedding.
                raise ValueError(
                    f"context embedding for {h} has width {ctx.shape[1]}, expected {width}; "
                    "the context cache is inconsistent (likely built with a different embedding "
                    "model). Delete the context_dir/.cache directory and rerun."
                )
            self.n_context_sentences[h] = len(ctx)
            blocks.append(ctx)
            starts.append(cursor)
            cursor += len(ctx)
        matrix = np.vstack(blocks)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # An all-zero embedding would divide by zero. Leave it as zeros (similarity 0).
        np.divide(matrix, norms, out=matrix, where=norms > 0)
        self._ctx_matrix = matrix
        self._group_starts = np.asarray(starts, dtype=np.int64)

    def score_patient(self, seg_emb: np.ndarray, chunk_size: int = 64) -> None:
        """Precompute ``own_max[segment, context_hpo]`` for one patient.

        Chunked over *segments*, not context sentences so each HPO's block stays contiguous
        and ``reduceat`` remains valid. This bounds the transient
        ``chunk_size x n_context_sentences`` product: the full union spans ~652k context sentences
        (an earlier exploratory run's figure), so an unchunked long report would allocate over a gigabyte for a
        matrix that is immediately reduced away.
        """
        self.prepare()
        seg = np.asarray(seg_emb, dtype=np.float32)
        if seg.ndim == 1:
            seg = seg[None, :]
        norms = np.linalg.norm(seg, axis=1, keepdims=True)
        seg = np.divide(seg, norms, out=np.zeros_like(seg), where=norms > 0)

        own_max = np.empty((len(seg), len(self._universe)), dtype=np.float32)
        for start in range(0, len(seg), chunk_size):
            block = seg[start:start + chunk_size]
            # Max within each HPO's contiguous block. Every HPO contributes at least one
            # sentence, so no group is empty and reduceat is safe.
            own_max[start:start + len(block)] = np.maximum.reduceat(
                block @ self._ctx_matrix.T, self._group_starts, axis=1
            )
        self._own_max = own_max

    def query_sims(self, hpo_id: str) -> np.ndarray:
        """Union similarity of every segment of the scored patient to ``hpo_id``."""
        if self._own_max is None:
            raise RuntimeError("call score_patient() before query_sims()")
        cols = self._cols[hpo_id]
        if len(cols) == 0:
            return np.zeros(self._own_max.shape[0], dtype=np.float32)
        return self._own_max[:, cols].max(axis=1)

    def has_context(self, hpo_id: str) -> bool:
        """True if ``hpo_id`` is a column of the ``own_max`` matrix (has context embeddings)."""
        return hpo_id in self._col_of

    def columns_for(self, context_hpos) -> np.ndarray:
        """Column indices of the given context HPOs, silently dropping any not in the universe.

        Used by the live traversal (the earlier runs) to turn a node's descendant set into the columns whose
        ``own_max`` a union score maxes over. Cache the result per node, it is patient-independent.
        """
        return np.fromiter(
            (self._col_of[h] for h in context_hpos if h in self._col_of), dtype=np.int64
        )

    def union_over_columns(self, cols: np.ndarray) -> np.ndarray:
        """Union similarity of every segment to the context HPOs at ``cols`` (a column-max).

        The lazy counterpart to :meth:`query_sims`: the traversal supplies the columns itself
        (from :meth:`columns_for`), not pre-registering every node as a query, so the
        scorer can be built once over the whole ontology and queried for any node on demand.
        """
        if self._own_max is None:
            raise RuntimeError("call score_patient() before union_over_columns()")
        cols = np.asarray(cols, dtype=np.int64)
        if cols.size == 0:
            return np.zeros(self._own_max.shape[0], dtype=np.float32)
        return self._own_max[:, cols].max(axis=1)

    def own_sims(self, hpo_id: str) -> np.ndarray:
        """Own-context similarity of every segment to ``hpo_id`` alone (its own column).

        ``own_max[:, col_of[hpo_id]]`` is by design the max cosine of each segment against
        the node's *own* synthetic-sentence sentences, the accept-score retrieval in the
        ``union_prune_own_accept`` mode, obtained for free from the same matrix.
        """
        if self._own_max is None:
            raise RuntimeError("call score_patient() before own_sims()")
        col = self._col_of.get(hpo_id)
        if col is None:
            return np.zeros(self._own_max.shape[0], dtype=np.float32)
        return self._own_max[:, col]

    def per_hpo_max(self) -> np.ndarray:
        """Per-context-HPO best-segment similarity for the scored patient (aligned to ``universe``).

        ``own_max.max(axis=0)``, for each context HPO, the similarity of the single most similar
        report segment to that HPO's own synthetic sentences. The flat top-M baseline (an earlier exploratory run) ranks the
        whole ontology by this vector in one numpy reduction, no per-node loop.
        """
        if self._own_max is None:
            raise RuntimeError("call score_patient() before per_hpo_max()")
        return self._own_max.max(axis=0)

    def per_segment_top(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-segment best-matching context HPOs, the other reduction of the same matrix.

        Returns ``(idx, sims)``, both ``(n_segments, m)``, highest similarity first. ``idx`` indexes
        :attr:`universe`, so ``universe[idx[i, r]]`` is the rank-``r`` candidate term for segment
        ``i``.

        :meth:`per_hpo_max` collapses ``own_max`` along the *segment* axis to rank the ontology for
        a whole report (an earlier exploratory run). Constrained decoding (the earlier runs) needs the candidate set of one
        *sentence*, which is the same matrix collapsed along the other axis, so this is a transpose
        of an existing computation, not a second retrieval pass.

        ``argpartition`` first, because m is ~50 against a universe of ~18k: a full argsort per
        segment would sort 18k entries to keep 50 of them, on every segment of every report.
        """
        if self._own_max is None:
            raise RuntimeError("call score_patient() before per_segment_top()")
        n_hpo = self._own_max.shape[1]
        m = max(1, min(int(m), n_hpo))
        # Partition to the m best, then sort only those m.
        part = np.argpartition(-self._own_max, m - 1, axis=1)[:, :m]
        sims = np.take_along_axis(self._own_max, part, axis=1)
        order = np.argsort(-sims, axis=1, kind="stable")
        idx = np.take_along_axis(part, order, axis=1)
        return idx, np.take_along_axis(sims, order, axis=1)

    def nbytes(self) -> int:
        """Memory of the current report's similarity matrix, in bytes."""
        return 0 if self._own_max is None else self._own_max.nbytes


def top_sentences(
    sims: np.ndarray, sentences: list[str], top_n: int = 5
) -> dict[str, list]:
    """Top-``top_n`` distinct sentences by similarity, highest first.

    Deduplicates on sentence *text* (not index) to match ``_lazy_retrieve`` in the earlier runs/exp10
    runners, a repeated sentence must not occupy two of the S slots. Returns the same keys those
    runners use, so downstream prompt construction is unchanged.
    """
    order = np.argsort(sims)[::-1]
    top_sents: list[str] = []
    top_scores: list[float] = []
    top_indices: list[int] = []
    for idx in order:
        if len(top_sents) >= top_n:
            break
        cand = sentences[idx]
        if cand not in top_sents:
            top_sents.append(cand)
            top_scores.append(float(sims[idx]))
            top_indices.append(int(idx))
    return {"top_sents": top_sents, "top_scores": top_scores, "top_indices": top_indices}
