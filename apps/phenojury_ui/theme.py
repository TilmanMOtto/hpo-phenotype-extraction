"""Chrome, re-exported from ``app/tree_ui`` plus the colour scales this app owns.

The palette, the Plotly template and the panel/stat components are shared with the other two
analysis apps: three UIs that look and read the same are one UI a reader has to learn
once, and the palette is a validated CVD-separated set that must not be extended by hand.

What this module adds is the four categorical scales specific to the ensemble, each keyed by *job*:

``model_colors``     identity, eight models, hue fixed by position in the **full** roster so
                     unchecking one in the designer never repaints the others
``fate_colors``      status, a missed term's cause of death, best → worst
``relation_colors``  identity with a severity *order*, the six FP classes
``outcome_colors``   status, TP/FP/FN
"""

from __future__ import annotations

from apps.ui_common.theme import (  # noqa: F401 - re-exported as this app's chrome API
    DARK,
    GRAPH_CONFIG,
    LIGHT,
    empty,
    note,
    palette,
    panel,
    plotly_layout,
    stat,
    stat_row,
)

from .autopsy import FATE_HELP, FATE_ORDER  # noqa: F401 - re-exported for the views
from .relations import RELATION_HELP, RELATION_ORDER  # noqa: F401


def model_colors(all_models: list[str], mode: str = "light") -> dict:
    """``{model: colour}``, hue keyed by position in *all_models*.

    Callers must pass the run's full roster, not the currently-selected subset: keying on the
    subset would repaint every series the moment a checkbox moves, and a reader comparing two
    configurations would be reading two different colour languages.
    """
    cat = palette(mode)["categorical"]
    return {model: cat[i % len(cat)] for i, model in enumerate(all_models)}


def fate_colors(mode: str = "light") -> dict:
    """Status colours for the recall autopsy, recoverable at the good end, dead at the muted end."""
    p = palette(mode)
    return {
        "below_votes": p["warning"],      # The k slider gets these back
        "below_min_count": p["serious"],
        "negated": p["diverging_pos"],    # a disagreement, not a failure, not red
        "not_linked": p["critical"],
        "not_written": p["neutral"],
        "no_evidence": p["diverging_neg"],
    }


def relation_colors(mode: str = "light") -> dict:
    """Identity colours for the six FP classes, in :data:`RELATION_ORDER`.

    Slots 1,2,3,5,6,8 of the categorical scale, the validated ordering for six adjacent
    categories, the same assignment ``app/tree_ui`` uses for its error classes.
    """
    cat = palette(mode)["categorical"]
    slots = [cat[0], cat[1], cat[2], cat[4], cat[5], cat[7]]
    return dict(zip(RELATION_ORDER, slots))


def outcome_colors(mode: str = "light") -> dict:
    """``{tp, fp, fn: colour}``."""
    p = palette(mode)
    return {"TP": p["good"], "FP": p["critical"], "FN": p["warning"], "TN": p["neutral"]}


def gate_colors(mode: str = "light") -> dict:
    """``{pass, fail, skip: colour}``."""
    p = palette(mode)
    return {"pass": p["good"], "fail": p["critical"], "skip": p["neutral"]}
