"""Render only the tab the user is looking at.

Every view's content callback listens on ``run-a`` / ``op-a`` / ``store-theme``, so without this
a single dropdown change fires all six of them and five sixths of the resulting work is thrown
away, including the two that classify every false positive against the ontology and sweep the
risk--coverage curve. That was the difference between a page appearing in about a second and in
about half a minute.

Dash has no built-in "only if visible": the whole layout exists from the start (which is what
lets a callback target a component on a tab that has never been opened), so the guard has to be
explicit. Each content callback takes ``Input("tabs", "value")`` and opens with::

    active.guard(tab, "scorecard")

which raises ``PreventUpdate`` when that view is not on screen. Two consequences worth knowing:

* Switching *to* a tab now fires its callback, because ``tabs.value`` is one of its inputs. That
  is what makes the deferral work rather than merely postponing the cost to a place where it is
  invisible.
* A view that was rendered, then hidden, then shown again re-renders from the cached report,
  not serving stale children. The reports are cached on disk by artifact mtime, so this
  is a dictionary lookup, not a recomputation.
"""

from __future__ import annotations

from dash import Input
from dash.exceptions import PreventUpdate


def guard(active_tab, view_id: str) -> None:
    """Raise ``PreventUpdate`` unless ``view_id`` is the tab currently on screen.

    ``active_tab`` of ``None`` is treated as "not this view": it only occurs before the tabs
    component has reported a value, and rendering everything in that window would give back the
    startup stall the guard exists to remove.
    """
    if active_tab != view_id:
        raise PreventUpdate


#: The input every guarded callback must declare, so the guard has something to read.
TAB_INPUT = Input("tabs", "value")
