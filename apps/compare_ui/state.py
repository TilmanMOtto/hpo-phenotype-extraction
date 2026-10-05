"""Process-global handles. Everything a ``dcc.Store`` must not hold.

There is one thing here and it is the registries -- one per cohort, each owning a bundle
directory, its index and a small LRU of parsed bundles. A ``dcc.Store`` cannot hold them for the
usual reason (browser state is JSON and crosses the wire on every callback) and for one specific
to this app: the HCY bundles carry patient report text, and the rule for HCY is that it stays on
the machine the data is on. What the browser receives is the one document the reader asked for,
rendered.

The **cohort key** is the part that does cross the wire, because it has to: it is a dropdown value
and every render callback needs it to know which registry to read. It is a short string naming a
build, which is the kind of thing browser state is for.

:func:`registry` is total. A callback firing with a stale or absent cohort -- the
first render before the dropdown has a value, or a dataset that has since been rebuilt away --
gets the default registry rather than ``None``, because a view handed ``None`` renders an empty
page that reads as "this build found nothing".
"""

from __future__ import annotations

REGISTRIES: dict = {}
DEFAULT_KEY: str = ""


def set_registries(registries: dict, default_key: str = "") -> None:
    """Install the discovered registries. *default_key* is what an unset cohort resolves to."""
    global REGISTRIES, DEFAULT_KEY
    REGISTRIES = dict(registries or {})
    DEFAULT_KEY = str(default_key or next(iter(REGISTRIES), ""))


def set_registry(registry) -> None:
    """Install a single registry under its own cohort key -- the one-cohort shorthand."""
    if registry is None:
        set_registries({}, "")
        return
    key = registry.cohort.key
    set_registries({key: registry}, key)


def keys() -> list:
    """Keys of the loaded datasets."""
    return list(REGISTRIES)


def registry(cohort: str = ""):
    """The registry for *cohort*, falling back to the default. ``None`` only when none exist."""
    if cohort and cohort in REGISTRIES:
        return REGISTRIES[cohort]
    return REGISTRIES.get(DEFAULT_KEY)


def default_key() -> str:
    """Key of the dataset shown first."""
    return DEFAULT_KEY
