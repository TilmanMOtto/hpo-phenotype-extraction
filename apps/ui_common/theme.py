"""Palette, Plotly template, and the shared UI components.

The categorical hues are a validated set (CVD-separated in both light and dark, chroma and
lightness in band). Two light-mode slots sit below 3:1 contrast on the light surface, which
obligates the relief rule, so every categorical chart here direct-labels its segments and
every one of them has a table view behind it (the error explorer). Do not add hues by hand:
re-run the palette validator if the set ever changes.

Colour carries meaning by *job*, not by decoration:

    categorical   identity, error classes, runs A vs B
    status        state, TP/FP/FN/TN, a GT term's fate
    sequential    magnitude, the τ heatmap (one hue, light→dark)
    diverging     polarity, A/B deltas (blue↔red, neutral gray at zero)
"""

from __future__ import annotations

from dash import dcc, html

# ── palette ──────────────────────────────────────────────────────────────────

LIGHT = {
    "surface": "#fcfcfb",
    "surface_2": "#f4f4f2",
    "border": "#dededa",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "text_muted": "#82817c",
    "grid": "#e8e8e4",
    "categorical": ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"],
    "sequential": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    "diverging_pos": "#2a78d6",
    "diverging_neg": "#d03b3b",
    "diverging_mid": "#f0efec",
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
    "neutral": "#a8a79f",
}

DARK = {
    "surface": "#1a1a19",
    "surface_2": "#242423",
    "border": "#3a3a37",
    "text": "#ffffff",
    "text_secondary": "#c3c2b7",
    "text_muted": "#8e8d85",
    "grid": "#333331",
    "categorical": ["#3987e5", "#199e70", "#c98500", "#008300", "#9085e9", "#e66767", "#d55181", "#d95926"],
    "sequential": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    "diverging_pos": "#3987e5",
    "diverging_neg": "#d03b3b",
    "diverging_mid": "#383835",
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
    "neutral": "#6e6d67",
}


def palette(mode: str = "light") -> dict:
    """Colour tokens of the light or dark theme."""
    return DARK if mode == "dark" else LIGHT


# The six FP relationship classes, ordered near-miss → hallucination. Order is the severity
# encoding (colour is identity only), so charts must keep it.
ERROR_ORDER = ("ancestor", "descendant", "sibling", "same_system", "unrelated", "no_gt")
ERROR_HELP = {
    "ancestor": "predicted a parent of the truth, too general",
    "descendant": "predicted a child of the truth, too specific",
    "sibling": "same neighbourhood, wrong branch",
    "same_system": "right organ system, nothing closer",
    "unrelated": "shares only the ontology root",
    "no_gt": "patient has no ground truth at all, pure hallucination",
}


def error_colors(mode: str = "light") -> dict:
    """``{false-positive category: colour}``."""
    cat = palette(mode)["categorical"]
    # Slots 1,2,3,5,6,8, the validated ordering for six adjacent categories.
    slots = [cat[0], cat[1], cat[2], cat[4], cat[5], cat[7]]
    return dict(zip(ERROR_ORDER, slots))


def outcome_colors(mode: str = "light") -> dict:
    """``{tp, fp, fn: colour}``."""
    p = palette(mode)
    return {"TP": p["good"], "FP": p["critical"], "FN": p["warning"], "TN": p["neutral"]}


# A ground-truth term's cause of death, best → worst. Status colours: each ships with a label.
FATE_ORDER = ("found", "rejected", "pruned", "skipped", "not_traversed")
FATE_HELP = {
    "found": "predicted correctly",
    "rejected": "the model was asked and said No, a model error",
    "pruned": "an ancestor read as confidently absent, so it was never asked, a pruning cost",
    "skipped": "below the depth cutoff, never asked, a config cost",
    "not_traversed": "never in the target set at all, a coverage gap",
}


def fate_colors(mode: str = "light") -> dict:
    """``{traversal fate of an annotated term: colour}``."""
    p = palette(mode)
    return {
        "found": p["good"],
        "rejected": p["warning"],
        "pruned": p["critical"],
        "skipped": p["serious"],
        "not_traversed": p["neutral"],
    }


def plotly_layout(mode: str = "light", **overrides) -> dict:
    """Recessive axes, no chart-junk, hover on by default. Fed to every figure.

    Axis overrides are *merged* into the themed axis defaults rather than replacing them: a caller
    that only wants to set a tick format must not silently lose the themed grid and zero-line
    colours and fall back to plotly's (which are built for a white surface and glare in dark mode).
    """
    p = palette(mode)
    axis_base = {
        "gridcolor": p["grid"],
        "zerolinecolor": p["grid"],
        "linecolor": p["border"],
        "tickfont": {"color": p["text_muted"]},
    }
    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif",
                 "size": 13, "color": p["text_secondary"]},
        "margin": {"l": 56, "r": 20, "t": 28, "b": 44},
        "xaxis": dict(axis_base),
        "yaxis": dict(axis_base),
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0,
                   "font": {"color": p["text_secondary"]}, "bgcolor": "rgba(0,0,0,0)"},
        "hoverlabel": {"font": {"size": 12}},
        "colorway": p["categorical"],
        "bargap": 0.28,
    }
    for key, value in overrides.items():
        if key in ("xaxis", "yaxis") and isinstance(value, dict):
            merged = dict(axis_base)
            merged.update(value)
            layout[key] = merged
        else:
            layout[key] = value
    return layout


GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}


# ── components ───────────────────────────────────────────────────────────────

def stat(label: str, value: str, sub: str = "", tone: str | None = None, help_text: str = "") -> html.Div:
    """One number, told plainly. The hero unit of the scorecard."""
    return html.Div(
        [
            html.Div(label, className="stat-label"),
            html.Div(value, className="stat-value", **({"data-tone": tone} if tone else {})),
            html.Div(sub, className="stat-sub") if sub else None,
        ],
        className="stat", title=help_text,
    )


def stat_row(stats: list) -> html.Div:
    """A row of stat tiles."""
    return html.Div(stats, className="stat-row")


def panel(
    title: str,
    body,
    *,
    panel_id: str,
    subtitle: str = "",
    graph_id: str | None = None,
    markdown: str = "",
) -> html.Div:
    """A titled card with a copy toolbar.

    ``graph_id`` enables Copy PNG (a clientside Plotly.toImage → clipboard). ``markdown``
    fills the Copy Markdown clipboard with the numbers behind the panel, so a figure can go
    into a slide and its data into a findings file without retyping either.
    """
    tools = []
    if graph_id:
        tools.append(html.Button(
            "PNG", id={"type": "copy-png", "graph": graph_id},
            className="copy-btn", title="Copy this chart to the clipboard as a PNG",
        ))
        # The clientside copy callback needs somewhere to land that isn't the button's own
        # n_clicks (that would be a circular dependency).
        tools.append(html.Div(id={"type": "copy-sink", "graph": graph_id},
                              style={"display": "none"}))
    tools.append(dcc.Clipboard(
        id={"type": "copy-md", "panel": panel_id}, content=markdown or " ",
        title="Copy the numbers behind this panel as Markdown", className="copy-btn copy-clip",
    ))

    return html.Div(
        [
            html.Div([
                html.Div([
                    html.H3(title, className="panel-title"),
                    html.Div(subtitle, className="panel-subtitle") if subtitle else None,
                ]),
                html.Div(tools, className="panel-tools"),
            ], className="panel-head"),
            html.Div(body, className="panel-body"),
        ],
        className="panel", id=f"panel-{panel_id}",
    )


def note(text: str, tone: str = "info") -> html.Div:
    """A note the reader must not miss, e.g. why τ_prune is locked on this run."""
    return html.Div(text, className=f"note note-{tone}")


def empty(message: str) -> html.Div:
    """A placeholder panel that shows *message*."""
    return html.Div(message, className="empty")
