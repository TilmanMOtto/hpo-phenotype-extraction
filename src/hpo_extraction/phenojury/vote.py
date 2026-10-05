"""Voting of the PhenoJury jurors: how per-sentence term sets become one prediction per report.

These functions are shared by the thesis protocol (``experiments/05_phenojury/protocol``) and the
PhenoJury application (:mod:`hpo_extraction.phenojury.pipeline`), so both count votes in one way.
They were moved here from the protocol script without changes to their bodies.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Hashable, Iterable, Mapping, NamedTuple, Sequence

logger = logging.getLogger(__name__)


#: The three matching units S5 sweeps. ``report`` is what every shipped ensemble number uses.
UNITS: tuple[str, ...] = ("report", "segment", "window")


#: The vote rules. ``closure`` is kept unreduced for completeness, ``an earlier exploratory run`` measured it at
#: µF1 0.234 because the accepted set is ancestor-closed by design and emits 99 predictions
#: per report at k=1, and ``closure_reduced`` is the form that rule is actually proposed in.
RULES: tuple[str, ...] = ("exact", "closure", "closure_reduced")


class Juror(NamedTuple):
    """One (model, prompt, normaliser) cell of the pool.

    A juror is the triple and not the model because S2 through S6 differ only in which axes they
    hold fixed. ``key`` is what appears in every table, so it is stable and sortable.
    """

    model: str
    prompt: str
    normaliser: str

    @property
    def key(self) -> str:
        """``model|prompt|normaliser``, the juror's name in every table."""
        return f"{self.model}|{self.prompt}|{self.normaliser}"

    @classmethod
    def parse(cls, key: str) -> "Juror":
        """The juror named by a key of the form ``model|prompt|normaliser``."""
        model, prompt, normaliser = key.split("|")
        return cls(model, prompt, normaliser)


Packed = dict[str, dict[Hashable, dict[int, frozenset]]]


def _closure_cached(terms: frozenset, view, cache: dict) -> frozenset:
    """``ancestors_of_set`` memoised on the exact input set.

    Juror sets repeat heavily, most sentences carry one or two terms, and eight jurors on four
    prompts name the same handful, so a cache keyed on the set turns what ``an earlier exploratory run`` measured as
    millions of calls into thousands. The closure convention itself stays
    ``OntologyView.ancestors_of_set``, i.e. The package's single definition.
    """
    hit = cache.get(terms)
    if hit is None:
        hit = frozenset(view.ancestors_of_set(terms))
        cache[terms] = hit
    return hit


def pack(
    jurors: Sequence[Juror],
    per_juror: Mapping[Juror, dict],
    report_ids: Sequence[str],
    unit: str,
    rule: str,
    view,
) -> Packed:
    """``{report: {unit_key: {juror_index: frozenset(terms)}}}`` for one ``(unit, rule)``.

    Identifiers are resolved through ``OntologyView.resolve`` here and nowhere else, so an obsolete
    id is mapped once rather than at every threshold.

    ``window`` is built from the segment packing by unioning each juror's sets over ``s-1, s, s+1``.
    Expressing it that way, not as a third counting rule is what keeps :func:`max_counts`
    the single definition of what a vote is.
    """
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}; expected one of {UNITS}")
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")

    closure_cache: dict = {}
    want_closure = rule != "exact"
    packed: Packed = {rid: {} for rid in report_ids}

    # The window centres are the report's sentences across EVERY juror, not each juror's own. A
    # juror that named a term only at sentence 5 and one that named it only at 6 agree under a +/-1
    # window, and they can only be seen to agree if both are asked about a centre they share. Making
    # each juror's own sentences the centres would give them disjoint buckets and silently turn the
    # window unit back into the segment unit.
    centres: dict[str, set[int]] = {rid: set() for rid in report_ids}
    if unit == "window":
        for juror in jurors:
            data = per_juror.get(juror) or {}
            for rid in report_ids:
                sentences = data.get(rid)
                if sentences:
                    centres[rid].update(int(s) for s in sentences)

    for index, juror in enumerate(jurors):
        data = per_juror.get(juror) or {}
        for rid in report_ids:
            sentences = data.get(rid)
            if not sentences:
                continue
            by_sentence = {int(s): t for s, t in sentences.items()}
            if unit == "report":
                merged: set[str] = set()
                for terms in by_sentence.values():
                    merged |= terms
                buckets: dict[Hashable, set[str]] = {0: merged}
            elif unit == "segment":
                buckets = {sent: set(terms) for sent, terms in by_sentence.items()}
            else:                                              # window +/-1
                buckets = {}
                for centre in centres[rid]:
                    merged = set()
                    for neighbour in (centre - 1, centre, centre + 1):
                        merged |= by_sentence.get(neighbour, set())
                    if merged:
                        buckets[centre] = merged
            for unit_key, terms in buckets.items():
                resolved = {t for t in (view.resolve(x) for x in terms) if t is not None}
                if not resolved:
                    continue
                frozen = frozenset(resolved)
                if want_closure:
                    frozen = _closure_cached(frozen, view, closure_cache)
                packed[rid].setdefault(unit_key, {})[index] = frozen
    return packed


def max_counts(
    packed: Packed, report_ids: Iterable[str], subset: frozenset[int]
) -> dict[str, dict[str, int]]:
    """``{report: {term: votes}}``, the largest number of *subset* jurors agreeing in any one unit.

    The maximum, not the sum, and that is the whole meaning of a non-report unit: under a
    segment unit a term is accepted when *k* jurors agree on it **in some one segment**. Summing
    across segments would let two jurors in two different sentences manufacture a vote of two that
    no segment ever cast, which is the report unit wearing a segment's name.

    Every threshold reads off this one structure, which is why it is computed per
    ``(rule, unit, subset, fold)``, not per ``k``.
    """
    out: dict[str, dict[str, int]] = {}
    for rid in report_ids:
        best: dict[str, int] = {}
        for jurors in packed.get(rid, {}).values():
            tally: Counter = Counter()
            for index in subset:
                terms = jurors.get(index)
                if terms:
                    tally.update(terms)
            for term, votes in tally.items():
                if votes > best.get(term, 0):
                    best[term] = votes
        out[rid] = best
    return out


def sets_at_k(counts: Mapping[str, Mapping[str, int]], k: int) -> dict[str, set[str]]:
    """``{report: set}`` of the terms at least *k* jurors supported."""
    return {rid: {t for t, n in row.items() if n >= k} for rid, row in counts.items()}


def reduce_specific(terms: set[str], view) -> set[str]:
    """Most specific members of a set, part of the closure rule's OUTPUT, not a scoring rule.

    Copied in behaviour from ``an earlier exploratory run.reduce_specific``: the two experiments must
    agree here or their closure rows are not comparable.
    """
    strict: set[str] = set()
    for term in terms:
        strict |= (set(view.ancestors(term)) - {term})
    return terms - strict


def predict_all_k(
    packed: Packed, report_ids: Sequence[str], subset: frozenset[int], rule: str, view,
    max_k: int | None = None,
) -> dict[int, dict[str, set[str]]]:
    """``{k: {report: set}}`` for every threshold the subset admits, from one counting pass."""
    counts = max_counts(packed, report_ids, subset)
    top = max_k or len(subset)
    out: dict[int, dict[str, set[str]]] = {}
    for k in range(1, top + 1):
        pred = sets_at_k(counts, k)
        if rule == "closure_reduced":
            pred = {rid: reduce_specific(terms, view) for rid, terms in pred.items()}
        out[k] = pred
    return out


def packings_for(jurors, per_juror, report_ids, rules, units, view) -> dict:
    """``{(rule, unit): Packed}``, every combination the selection may ask for, built once.

    Packing is the expensive half (it resolves identifiers and, for a closure rule, takes ancestor
    closures), and the selection loop asks for the same combination thousands of times. Building
    them up front is what makes the unit axis affordable.
    """
    out = {}
    for rule in rules:
        for unit in units:
            out[(rule, unit)] = pack(jurors, per_juror, report_ids, unit, rule, view)
            logger.debug("packed rule=%s unit=%s", rule, unit)
    return out


def predict(packings, rule, unit, report_ids, subset, k, view):
    """One ``(rule, unit, k)`` prediction for one subset."""
    return predict_all_k(packings[(rule, unit)], report_ids, subset, rule, view, max_k=k)[k]
