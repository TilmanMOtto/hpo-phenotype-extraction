"""The process-wide Registry handle.

Views import ``get_registry()`` rather than receiving the registry through callback plumbing, Dash callbacks can only carry JSON, and the registry holds an HPOTree, an OntologyView and
several DataFrames that must never be serialised. This is the seam that keeps the heavy objects
server-side.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from .registry import Registry

_REGISTRY = None


def set_registry(registry) -> None:
    """Install *registry* as the process-wide registry the callbacks read."""
    global _REGISTRY
    _REGISTRY = registry


def get_registry():
    """The registry installed by :func:`set_registry`. Raises when none is installed."""
    if _REGISTRY is None:
        raise RuntimeError("Registry not initialised, app.py must call set_registry() at startup")
    return _REGISTRY
