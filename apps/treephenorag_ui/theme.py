"""Palette, Plotly template, and the shared UI components.

The palette is the validated set from ``apps/ui_common/theme.py``, CVD-separated in both light and
dark, chroma and lightness in band. Two light-mode slots sit below 3:1 contrast on the light
surface, which obligates the relief rule: every categorical chart here direct-labels its segments
and has a table view behind it. Do not add hues by hand.

Colour carries meaning by *job*, not by decoration:

    categorical   identity, FP buckets, methods A vs B
    status        state, TP/FP/FN/TN, an annotated term's fate
    sequential    magnitude, the τ frontier
    diverging     polarity, A/B deltas (blue↔red, neutral gray at zero)

The vocabularies below are the earlier runs', not tree_ui's. The FP buckets are the Garcia et al. taxonomy
implemented in ``hpo_extraction.evaluation.metrics.errors``. The ground truth-term fates are the four outcomes a
traversal can produce, which are *not* the same five ``app/tree_ui`` uses (earlier has no ``skipped``
depth cutoff, and its ``blocked`` fate is attributable to a specific severing node).
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


# ── vocabularies ─────────────────────────────────────────────────────────────

#: False-positive buckets, ordered near-miss → hallucination → hygiene. Mirrors
#: ``errors.GARCIA_BUCKETS + errors.HYGIENE_BUCKETS``; ``assert_vocabularies_match`` pins it.
FP_ORDER = ("ancestor", "descendant", "sibling", "unrelated", "hallucination",
            "out_of_subtree", "no_gold")
FP_HELP = {
    "ancestor": "a parent of the nearest annotated term, the system was too general",
    "descendant": "a child of the nearest annotated term, too specific",
    "sibling": "shares a direct is_a parent with the nearest annotated term",
    "unrelated": "a real phenotype, but nowhere near anything annotated",
    "hallucination": "the code names no term the ontology knows",
    "out_of_subtree": "a real HPO term outside phenotypic abnormality (inheritance, modifiers)",
    "no_gold": "the report has no annotated terms at all, so there is no reference to be near",
}


def fp_colors(mode: str = "light") -> dict:
    """``{false-positive category: colour}``."""
    cat = palette(mode)["categorical"]
    p = palette(mode)
    return {
        "ancestor": cat[0], "descendant": cat[1], "sibling": cat[2],
        "unrelated": cat[4], "hallucination": cat[5],
        "out_of_subtree": cat[7], "no_gold": p["neutral"],
    }


#: What was predicted *around* a missed annotated term. Mirrors ``errors.FN_BUCKETS``.
FN_ORDER = ("ancestor_predicted", "descendant_predicted", "sibling_predicted", "nothing_near")
FN_HELP = {
    "ancestor_predicted": "a parent was predicted, a granularity failure, not a detection failure",
    "descendant_predicted": "a child was predicted, the region was found, the level was not",
    "sibling_predicted": "a sibling was predicted, right neighbourhood, wrong term",
    "nothing_near": "nothing in this region was predicted, a detection failure",
}


def fn_colors(mode: str = "light") -> dict:
    """``{false-negative category: colour}``."""
    cat = palette(mode)["categorical"]
    return {"ancestor_predicted": cat[0], "descendant_predicted": cat[1],
            "sibling_predicted": cat[2], "nothing_near": cat[5]}


def outcome_colors(mode: str = "light") -> dict:
    """``{tp, fp, fn: colour}``."""
    p = palette(mode)
    return {"TP": p["good"], "FP": p["critical"], "FN": p["warning"], "TN": p["neutral"]}


#: An annotated term's cause of death in a tree run, best → worst. Every fate ships with a label,
#: because status colour alone cannot carry a five-way distinction.
FATE_ORDER = ("found", "rejected", "blocked", "outside_graph")
FATE_HELP = {
    "found": "accepted, the traversal reached it and the SLM said yes",
    "rejected": "reached, asked, and scored below τ_accept, an identification failure",
    "blocked": "an ancestor scored below τ_prune, so it was never reached, a pruning failure",
    "outside_graph": "not in the traversal graph at all, an ontology coverage gap",
}


def fate_colors(mode: str = "light") -> dict:
    """``{traversal fate of an annotated term: colour}``."""
    p = palette(mode)
    return {
        "found": p["good"],
        "rejected": p["warning"],
        "blocked": p["critical"],
        "outside_graph": p["neutral"],
    }


def assert_vocabularies_match() -> None:
    """Fix the display vocabularies to the metric library's own bucket tuples.

    The UI hard-codes bucket order (order is the severity encoding, so charts must keep it) and
    a colour per bucket. If ``thesis_metrics.errors`` ever gains or renames a bucket, an unpinned
    UI would silently drop that bucket's rows from every chart, the failure would look like a
    smaller error count, which is the direction a reader would not question.
    """
    from hpo_extraction.evaluation.metrics import errors

    expected_fp = tuple(errors.GARCIA_BUCKETS) + tuple(errors.HYGIENE_BUCKETS)
    assert FP_ORDER == expected_fp, f"FP_ORDER drifted from errors.py: {FP_ORDER} != {expected_fp}"
    assert FN_ORDER == tuple(errors.FN_BUCKETS), \
        f"FN_ORDER drifted from errors.py: {FN_ORDER} != {tuple(errors.FN_BUCKETS)}"
    for bucket in FP_ORDER:
        assert bucket in FP_HELP and bucket in fp_colors(), f"FP bucket {bucket} has no help/colour"
    for bucket in FN_ORDER:
        assert bucket in FN_HELP and bucket in fn_colors(), f"FN bucket {bucket} has no help/colour"


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

def stat(label: str, value: str, sub: str = "", tone: str | None = None,
         help_text: str = "", tint: str | None = None) -> html.Div:
    """One number, told plainly. The hero unit of the scorecard.

    ``tone`` colours the number; ``tint`` colours the tile it sits in. They are separate because
    they answer different questions: a tone says how to read the value, a tint says something
    about the thing the value describes, whether a node was accepted, whether it was expanded, which is legible before the number itself is read. Both take ``good``/``warning``/``critical``
    (``tone`` also takes ``accent``).
    """
    return html.Div(
        [
            html.Div(label, className="stat-label"),
            html.Div(value, className="stat-value", **({"data-tone": tone} if tone else {})),
            html.Div(sub, className="stat-sub") if sub else None,
        ],
        className="stat", title=help_text, **({"data-tint": tint} if tint else {}),
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


def note(text, tone: str = "info") -> html.Div:
    """A note the reader must not miss, e.g. which ground truth file this cell was scored against."""
    return html.Div(text, className=f"note note-{tone}")


def empty(message: str) -> html.Div:
    """A placeholder panel that shows *message*."""
    return html.Div(message, className="empty")


def table(headers, rows, *, align=None, className="tbl") -> html.Div:
    """A plain HTML table in a horizontally scrollable box.

    Wide tables must scroll inside their own container, not widening the page, the
    culprit leaderboard and the flip table are both wider than a sidebar-constrained main column.
    """
    align = align or {}
    head = html.Thead(html.Tr([html.Th(h) for h in headers]))
    body = html.Tbody([
        html.Tr([
            html.Td(cell, className="num" if align.get(i) == "num" else None)
            for i, cell in enumerate(row)
        ])
        for row in rows
    ])
    return html.Div(html.Table([head, body], className=className), className="tbl-scroll")
