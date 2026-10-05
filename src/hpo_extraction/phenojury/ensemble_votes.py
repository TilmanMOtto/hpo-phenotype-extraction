"""Vote-mask primitives for the SLM ensemble, the pure half of the aggregation re-run.

Every aggregation rule the ensemble driver sweeps is a function of *which models detected which
term*, so once that is packed into an integer bitmask per ``(report, term)`` the whole decision
layer re-runs offline at no GPU cost. ``slm_ensemble_experiment`` states the same invariant from
the other side ("score once, re-run the decision logic"). These are the functions that make it
usable outside the driver.

They live in ``src/`` because two things need them and neither should own them: ``app/phenojury_generation_free_listing``
drives its live k-slider and model checklist off them, and ``an earlier exploratory run`` uses
them to ask what the vote rule *costs*, which terms a higher k discards, and who was essential
for the ones it keeps. Only the mask arithmetic is here. Anything that needs the app's bundle, or
the driver's scorer, stays where it was.

**Masks.** A mask is an int whose bit *i* is set when ``models[i]`` detected the term. Bit order is
the caller's model list, so a mask is only meaningful beside the list it was built with, which is
why :func:`mask_of` and :func:`bits` take that list explicitly rather than a module-level default.

**Decisiveness.** For a k-of-N rule, model *m* is decisive for a term when the term is predicted
with the full voter set and is not predicted without *m*. Because the rule is a threshold on a
count, that reduces to "the term has k votes and *m* is one of them": every voter of a term
sitting on the threshold is essential, and no voter of a term above it is. See
:func:`decisive_mask`, ``test_exp13_06_ui`` cross-checks it against brute-force leave-one-out over
all 2^n subsets.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

#: Popcount for the mask range an 8-model ensemble can produce, which is every call this module
#: sees in practice. :func:`popcount` falls back to ``bin()`` above it, not capping.
_POPCOUNT: tuple[int, ...] = tuple(bin(i).count("1") for i in range(1 << 12))


def popcount(mask: int) -> int:
    """Number of models in *mask*."""
    return _POPCOUNT[mask] if 0 <= mask < len(_POPCOUNT) else bin(mask).count("1")


def mask_of(subset: Iterable[str], models: Sequence[str]) -> int:
    """The mask for a set of model names, indexed against *models*. Unknown names are ignored."""
    index = {name: i for i, name in enumerate(models)}
    mask = 0
    for name in subset:
        bit = index.get(name)
        if bit is not None:
            mask |= 1 << bit
    return mask


def bits(mask: int, models: Sequence[str]) -> list[str]:
    """The model names in *mask*, in *models* order, the inverse of :func:`mask_of`."""
    return [name for i, name in enumerate(models) if mask & (1 << i)]


def vote_k_from_masks(masks: Mapping[str, Mapping[str, int]], subset_mask: int, k: int) -> dict:
    """``{report: set(hpo)}`` keeping every term at least *k* of *subset_mask*'s models detected.

    Equivalent to the driver's ``vote_k_sets`` restricted to a subset, but a popcount per term
    instead of a re-tally per report, which is what makes exhaustive subset sweeps affordable.
    """
    return {
        report_id: {h for h, mask in row.items() if popcount(mask & subset_mask) >= k}
        for report_id, row in masks.items()
    }


def plurality_from_masks(sent_masks: Mapping[str, Mapping[int, Mapping[str, int]]],
                         subset_mask: int) -> dict:
    """``{report: set(hpo)}``, the per-sentence plurality restricted to a subset.

    Per sentence each model votes once per term it detected. Models that detected nothing do not
    vote. The top-voted term(s) win and ties keep every winner. A report's set is the union over
    its sentences. Same rule as the driver's ``plurality_sets``, expressed over masks.
    """
    out: dict[str, set] = {}
    for report_id, sentences in sent_masks.items():
        selected: set = set()
        for hpos in sentences.values():
            tally = {h: popcount(mask & subset_mask) for h, mask in hpos.items()}
            tally = {h: v for h, v in tally.items() if v > 0}
            if not tally:
                continue
            best = max(tally.values())
            selected |= {h for h, v in tally.items() if v == best}
        out[report_id] = selected
    return out


# ── Ontology-aware voting: counting over ancestor closures ───────────────────

def closure_counts(
    masks: Mapping[str, Mapping[str, int]],
    subset_mask: int,
    models: Sequence[str],
    view,
) -> dict[str, dict[str, int]]:
    """``{report: {term: n_up}}`` where ``n_up(v)`` counts jurors naming ``v`` *or a descendant*.

    Exact voting asks every juror for the same identifier. Jurors routinely agree that a finding is
    present and disagree only on how specific to be, one writes *Global developmental delay*, the
    next *Neurodevelopmental delay*, and exact counting splits those votes, so a term two jurors
    support in substance can fail a threshold of two. Counting over ancestor closures instead lets
    the disagreement back off to the most specific term the jury actually agrees on.

    Formally ``n_up(v) = sum_g 1[v in closure(Z_g)]`` with ``closure`` the reflexive ancestor
    closure. Each juror contributes **at most one** vote to any term, however many of its
    descendants that juror named, the count is over jurors, not over mentions, which is what keeps
    it comparable to the exact count and on the same ``0..J`` scale.

    Two properties follow, and ``tests/unit/test_ensemble_closure_votes.py`` pins both:

    * **Monotone along the hierarchy**: ``n_up(u) >= n_up(v)`` whenever ``u`` is an ancestor of
      ``v``, since every closure containing ``v`` contains ``u``. So the accepted set is closed
      under ancestors, and reducing it to its most specific members is well defined.
    * **Agrees with exact voting at k = 1**: the union over jurors is the same set either way, up
      to the ancestors the closure adds, so the two rules differ only in what a *threshold* does.

    ``view`` is an :class:`~hpo_extraction.evaluation.metrics.ontology.OntologyView`. Only
    ``ancestors_of_set`` is used, so the closure convention (universal nodes excluded) is the
    package's single definition, not a second one written here.
    """
    out: dict[str, dict[str, int]] = {}
    n_models = len(models)
    for report_id, row in masks.items():
        counts: dict[str, int] = {}
        for bit in range(n_models):
            flag = 1 << bit
            if not (subset_mask & flag):
                continue
            named = [h for h, mask in row.items() if mask & flag]
            if not named:
                continue
            for term in view.ancestors_of_set(named):
                counts[term] = counts.get(term, 0) + 1
        out[report_id] = counts
    return out


def closure_vote_from_masks(
    masks: Mapping[str, Mapping[str, int]],
    subset_mask: int,
    models: Sequence[str],
    view,
    k: int,
) -> dict[str, set]:
    """``{report: set(hpo)}``, the terms at least *k* jurors support under the closure rule.

    Returned **unreduced**, like :func:`vote_k_from_masks`, so the caller applies whatever
    specificity convention the experiment uses, not having one baked in here. Note that the
    raw accepted set is ancestor-closed by design, so it is much larger than the exact rule's
    at the same *k*. The interesting comparison is after both are reduced.
    """
    counts = closure_counts(masks, subset_mask, models, view)
    return {report_id: {h for h, n in row.items() if n >= k} for report_id, row in counts.items()}


def decisive_mask(mask: int, subset_mask: int, k: int) -> int:
    """The voters without whom this term would fall below *k*, empty unless it sits on k.

    A term with more than *k* votes survives the loss of any single model, so none of its voters is
    decisive. A term with *k* loses on the removal of any one of them, so all of them are.
    """
    voters = mask & subset_mask
    return voters if popcount(voters) == k else 0
