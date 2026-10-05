"""The pieces every tab needs: the render gate, the failure guard, and the term chip.

Two of these are essential rather than convenient.

:func:`should_render` -- all three tabs live in the layout at once, toggled by ``display``, so
every callback bound to the shared stores would otherwise fire on every control move: three whole
re-renders and three payloads for the one tab the reader can see. Over an ``ssh -L`` tunnel that is
the difference between a click and a wait.

:func:`guard` -- a callback that raises returns a 500 for its component and the page renders around
the hole. Here that hole is a column, and a missing column reads as a method that found nothing.
The guard turns it into a visible red note *and* records it, so :mod:`apps.compare_ui.selftest` can
drain the record and fail: a panel that failed inside the guard returned normally, so a render
matrix would otherwise see a ``Div`` and call it a pass.
"""

from __future__ import annotations

import json
import traceback

from dash import dcc, html

from apps.compare_ui import theme

#: ``"view_id:panel" -> traceback``. Drained by the selftest. See the module docstring.
_FAILED: dict = {}


def guard(view_id: str, panel: str, fn, *args, **kwargs):
    """Run a renderer, or return a visible red note and record why."""
    try:
        return fn(*args, **kwargs)
    except Exception:                                    # noqa: BLE001 - that is the whole job
        key = "{}:{}".format(view_id, panel)
        _FAILED[key] = traceback.format_exc()
        return theme.note("{} failed to render. See the server log.".format(panel), "danger")


def failures() -> dict:
    """``{view: error message}`` of the views that failed to render since the last clear."""
    return dict(_FAILED)


def clear_failures() -> None:
    """Forget the recorded render failures."""
    _FAILED.clear()


def render_key(*parts) -> str:
    """A signature for "what this panel last drew".

    JSON, not a tuple because it round-trips through a ``dcc.Store`` unchanged -- a tuple
    comes back as a list and would never compare equal to the one just built, so every panel would
    redraw on every callback and the gate would do nothing.
    """
    return json.dumps(parts, sort_keys=True, default=str)


def should_render(view_id: str, active_tab: str, key: str, previous) -> bool:
    """True when *view_id* is the active tab and the selection changed since the last render."""
    return active_tab == view_id and key != previous


# -- chips -------------------------------------------------------------------

def term_chip(bundle, hpo_id: str, outcome: str = "", method: str = "", *, prefix: str = "",
              clickable: bool = True, extra: str = ""):
    """One phenotype, as the reader meets it everywhere in this app.

    The hover text is the bundle's own ``terms[id].tooltip`` -- the same wording
    ``apps.treephenorag_ui.terms.tooltip`` produces, baked in at build time. So hovering a term here says
    what hovering it in the tree deep-dive says, and the app needs no ontology to do it.

    The outcome is carried twice, as a tone *and* as a glyph, because colour alone is not a channel
    everyone has.
    """
    entry = (bundle.get("terms") or {}).get(hpo_id) or {}
    label = entry.get("label") or hpo_id
    title = entry.get("tooltip") or hpo_id
    if outcome:
        title = "{}\n\n{}".format(theme.OUTCOME_HELP.get(outcome, outcome), title)

    children = []
    if outcome:
        children.append(html.Span(theme.OUTCOME_GLYPH.get(outcome, "?"), className="cmp-glyph"))
    children.append(html.Span(prefix + label, className="cmp-chip-label"))
    if extra:
        children.append(html.Span(extra, className="cmp-chip-extra"))

    props = {
        "className": "cmp-chip",
        "title": title,
        "n_clicks": 0,
    }
    if outcome:
        props["data-outcome"] = outcome
    if clickable:
        props["id"] = {"type": "cmp-chip", "method": method or "gold", "hpo": hpo_id}
    return html.Button(children, **props)


def qualifier_chips(qualifiers):
    """The curator's own difficulty labels -- why this term was hard, in their words."""
    return [html.Span(q.replace("_", " "), className="cmp-qual",
                      title="a curation qualifier recorded on this annotation")
            for q in qualifiers or ()]


def outcome_legend():
    """The legend of the tp, fp and fn colours."""
    return html.Div(
        [html.Span([html.Span(theme.OUTCOME_GLYPH[o], className="cmp-glyph"),
                    html.Span(theme.OUTCOME_HELP[o])],
                   className="cmp-chip cmp-legend", **{"data-outcome": o})
         for o in ("tp", "fp", "fn")],
        className="cmp-legend-row")


# -- small tables ------------------------------------------------------------

def table(headers, rows, *, className: str = "tbl"):
    """A plain ``html.Table``. It ships the rows drawn and nothing else."""
    return html.Div(
        html.Table([
            html.Thead(html.Tr([html.Th(h) for h in headers])),
            html.Tbody([html.Tr([html.Td(cell) for cell in row]) for row in rows]),
        ], className=className),
        className="tbl-scroll")


def md_table(headers, rows) -> str:
    """The same table as Markdown, for the panel's Copy button."""
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_plain(c) for c in row) + " |")
    return "\n".join(lines)


def _plain(cell) -> str:
    """Markdown cannot carry a Dash component, so a component degrades to its text."""
    if isinstance(cell, str):
        return cell
    if isinstance(cell, (int, float)):
        return fmt(cell)
    children = getattr(cell, "children", None)
    if isinstance(children, str):
        return children
    if isinstance(children, (list, tuple)):
        return " ".join(_plain(c) for c in children)
    return ""


def fmt(value, digits: int = 3) -> str:
    """A number for a table cell. ``None`` prints as an em dash, never as zero.

    Zero would be a claim. A report with no ground truth has no recall, and printing 0.000 there says the
    method failed at something it was never asked to do.
    """
    if value is None or value == "":
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "—"
    if abs(number) >= 0.001 or number == 0:
        return "{:.{}f}".format(number, digits)
    return "{:.2e}".format(number)


def copy_target(panel_id: str, markdown: str):
    """A Copy-Markdown clipboard with no panel chrome, for places a card would be too heavy."""
    return dcc.Clipboard(id={"type": "copy-md", "panel": panel_id}, content=markdown or " ",
                         title="Copy this as Markdown", className="copy-btn copy-clip")
