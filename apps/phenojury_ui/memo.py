"""Per-bundle memo for the derived tables the views recompute.

The views share their inputs far more than the module boundaries suggest. One render of the deep
dive asks for ``predicted_sets`` three times and the whole-cohort autopsy once. The scorecard sweeps
k twice. The false-positive view used to walk the ontology once per k. All of it is a pure function
of ``(bundle, configuration)``, and the configuration changes only when the reader moves a control.

So the results are cached on the bundle itself. That scoping is the point: a bundle is evicted from
the registry's LRU when a cohort is dropped, and rebuilt when the run directory's mtime changes, so
the memo cannot outlive the data it was computed from, which a module-level cache keyed by run id
could.

Every entry is keyed by :func:`config_key`, so two configurations that differ only in the order the
model checkboxes were clicked hit the same entry, and a k that a rule does not use cannot split it.
"""

from __future__ import annotations

from collections import OrderedDict

#: Entries per bundle. A reader sweeping the k slider across an eight-model ensemble touches eight
#: configurations per named table. A few of those at once is the working set worth keeping.
MAX_ENTRIES = 48


def new_store() -> OrderedDict:
    """An empty memo store (least recently used first)."""
    return OrderedDict()


def config_key(config: dict) -> tuple:
    """A hashable identity for a configuration, the five fields every derived table depends on.

    ``models`` is sorted: the subset is a set, and keying on click order would miss every hit.

    ``subset`` is here because it changes *which reports* a table is computed over, which is as
    much a part of the answer as k is. Leaving it out would be the worst bug this cache can have:
    every number would be right for the whole cohort and served under a banner saying it was
    computed over twenty reports, with nothing on screen to show the difference.
    """
    return (
        config.get("rule") or "vote_k",
        int(config.get("k") or 1),
        int(config.get("min_count") or 1),
        tuple(sorted(config.get("models") or ())),
        config.get("subset") or "all",
    )


def cached(bundle: dict, key, build):
    """``build()``, remembered on *bundle* under *key*.

    Falls through to a plain call when the bundle carries no store, so the pure functions stay
    usable on the hand-built dicts the unit tests pass them.
    """
    store = bundle.get("memo")
    if store is None:
        return build()
    if key in store:
        store.move_to_end(key)
        return store[key]
    value = build()
    store[key] = value
    while len(store) > MAX_ENTRIES:
        store.popitem(last=False)
    return value


def own_gold(bundle: dict, gold: dict) -> bool:
    """True when *ground truth* is the bundle's own annotation set.

    ``drivers.py`` scores a cohort against an *external* ground-truth file, and those results must
    never be served from, or written into, a cache keyed only by the configuration. Identity, not
    equality: cheap, and it cannot be fooled by two files that happen to agree on this cohort.
    """
    return gold is bundle.get("gold")
