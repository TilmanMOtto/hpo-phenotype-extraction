"""View modules. Each exposes ``layout()`` and ``register(app)``.

``layout()`` is called once at app construction and returns the static shell, filter controls
and empty containers. Everything inside is filled by callbacks, which read the registry directly
through :mod:`apps.treephenorag_ui.state` rather than receiving data through Dash plumbing: callbacks can
only carry JSON, and the node tables are gigabytes.

Figure builders are module-level functions of the form ``_x_figure(report, mode)``. They are
private by naming and public by use, the tests call them headlessly on a fixture report, which
is what makes the charts testable without a browser.
"""
