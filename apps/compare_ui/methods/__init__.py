"""One adapter per column: read that method's own records, place them, explain them.

Every adapter answers the same two questions about one report and is free to answer them from
completely different evidence, because the five methods kept completely different evidence:

``predictions()``  -- what did this method emit for this report?
``evidence()``     -- for each term, where on the report does it sit, and *why* did this method
                      decide it?

The second is the whole point of the app and the reason an adapter is a module rather than a dict
of paths. PhenoBERT's "why" is a matched phrase at a character offset; AutoPCR's is which of three
linking routes fired; RAG-HPO's is a ranked candidate menu and a Yes; PhenoJury's is eight models'
verbatim generations and a vote count; TreePhenoRAG's is a pair of scores against a pair of
thresholds and, for a miss, the ancestor that blocked it. Those are not variations on a shape --
they are five different shapes, and :mod:`apps.compare_ui.views.reasons` renders each on its own
terms, not flattening them into a lowest common denominator that says nothing.

The shared part is thin: :class:`base.Adapter` fixes the lifecycle (``load`` once per
cohort, ``evidence`` once per report) and :mod:`base` supplies the placement helpers, because
placement is the one thing all five must agree about.
"""

from __future__ import annotations

from apps.compare_ui.methods import autopcr, base, phenobert, phenojury, raghpo, tree

#: ``{roster key: adapter class}``. The roster decides which columns exist and in what order. This
#: decides how each one is read. A key in one and not the other is a configuration error, and
#: :func:`adapter_for` says so, not silently dropping a column.
ADAPTERS = {
    "phenobert": phenobert.PhenoBertAdapter,
    "autopcr_70b": autopcr.AutoPcrAdapter,
    "raghpo_70b": raghpo.RagHpoAdapter,
    "phenojury": phenojury.PhenoJuryAdapter,
    "treephenorag": tree.TreeAdapter,
}

__all__ = ["ADAPTERS", "adapter_for", "base"]


def adapter_for(key: str):
    """A new adapter instance for the method *key*."""
    try:
        return ADAPTERS[key]
    except KeyError:
        raise KeyError(
            "no adapter for roster key {!r}; apps/compare_ui/roster.py and "
            "apps/compare_ui/methods/__init__.py have drifted apart".format(key)) from None
