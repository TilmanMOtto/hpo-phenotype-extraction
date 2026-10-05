"""One module per panel. Each exposes ``layout()``, ``register(app)`` and pure ``render_*``
functions; ``app.py`` wires them once and ``selftest.py`` drives the ``render_*`` functions
directly, without a browser.
"""
