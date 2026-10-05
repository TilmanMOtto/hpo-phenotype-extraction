"""One module per tab. Each exposes ``layout()``, ``register(app)`` and pure ``render_*``
functions; ``app.py`` lists them once in ``VIEWS`` and ``selftest.py`` drives the ``render_*``
functions directly, without a browser.
"""
