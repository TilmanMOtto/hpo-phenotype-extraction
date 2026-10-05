"""The process-wide Registry handle.

Views call :func:`get_registry` rather than receiving the registry through callback plumbing: Dash
callbacks can only carry JSON, and the registry holds an ``HPOTree``, the detection tensor and
every raw generation in the cohort. This is the seam that keeps all of that server-side, so what
crosses an SSH tunnel is a finished figure or one page of rows.
"""

from __future__ import annotations

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


def has_registry() -> bool:
    """True when a registry is installed."""
    return _REGISTRY is not None
