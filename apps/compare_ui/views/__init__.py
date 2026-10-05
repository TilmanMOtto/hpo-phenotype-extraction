"""Three tabs, one contract.

Each module exposes ``VIEW_ID``, ``layout()``, ``register(app)`` and pure ``render_*`` functions.
``app.py`` wires them once; ``selftest.py`` drives the ``render_*`` functions directly, with no
browser and no server.

That split is the reason the selftest is worth anything. A Dash callback that raises does not crash
the process -- it returns a 500 for one component and the page renders around the hole. On this app
that hole would be a method column that silently failed to draw, next to four that did, which reads
as "this method found nothing here" rather than as an error. Keeping the rendering out of the
callback bodies is what lets a test call it and see the exception.
"""

from __future__ import annotations

from apps.compare_ui.views import (  # noqa: F401
    common, provenance, reader, reasons, scorecard,
)

__all__ = ["common", "provenance", "reader", "reasons", "scorecard"]
