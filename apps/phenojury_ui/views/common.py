"""Helpers every view shares. Not a view, it has no ``layout`` and registers no callbacks.

The important one is :func:`resolve`. ``store-config`` is browser state and can outlive the run it
was chosen for: switch cohort and a model subset may name models this cohort never ran, or a k
above its ensemble size. Every view resolves the stored config against the bundle before using it,
so a stale store degrades to the nearest valid configuration instead of producing a silently empty
screen.
"""

from __future__ import annotations

import json
import logging
import traceback

from dash import html

from .. import frame as frame_mod
from .. import state, theme

logger = logging.getLogger(__name__)

#: What a view gets when there is nothing to show yet.
NO_RUN = "Pick a cohort in the sidebar."

#: ``{run_id: traceback}`` for cohorts whose bundle could not be built.
#:
#: Two jobs. It is what :func:`load_failure` renders on screen, so a broken cohort reports its own
#: reason instead of impersonating "nothing selected yet". And it stops the retry storm: without it
#: a failing build is never cached, so all nine callbacks rebuild it on every interaction and the
#: log fills with "Building bundle" while the page stays blank.
_FAILED: dict[str, str] = {}


def bundle_for(run_id: str | None):
    """The bundle for *run_id*, or ``None``.

    Never raises, a Dash callback that propagates returns a 500 for one component and the page
    renders around the hole, which is strictly less useful than an error panel. But the traceback
    is logged and kept: swallowing it silently is what turns a one-line bug into an afternoon.
    """
    if not run_id or not state.has_registry():
        return None
    if run_id in _FAILED:
        return None
    try:
        return state.get_registry().get_bundle(run_id)
    except Exception as exc:  # noqa: BLE001 - surfaced by load_failure(), not hidden
        _FAILED[run_id] = traceback.format_exc()
        logger.error("Could not build the bundle for %r: %s\n%s",
                     run_id, exc, _FAILED[run_id])
        return None


def load_failure(run_id: str | None) -> str | None:
    """The traceback for a cohort that failed to load, if it did."""
    return _FAILED.get(run_id) if run_id else None


def clear_failures() -> None:
    """Forget cached failures, called when the data source is repointed."""
    _FAILED.clear()


def no_data(run_id: str | None, title: str = "Nothing to show"):
    """The empty state every view opens with: either 'pick a cohort' or the reason it broke."""
    failure = load_failure(run_id)
    if failure is None:
        return theme.empty(NO_RUN)
    return theme.panel(
        title,
        html.Div([
            theme.note(f"Cohort {run_id!r} could not be loaded. The full traceback is in the "
                       "server log; the tail is below.", "danger"),
            html.Pre(failure[-2000:], className="prompt-box"),
        ]),
        panel_id="load-failure")


# ── rendering only what is on screen ─────────────────────────────────────────
#
# All seven views live in the layout at once, toggled by ``display``, so every callback bound to
# ``store-config`` used to fire on every control move, seven whole-cohort recomputations and seven
# payloads for the one panel the reader can actually see. Over an SSH tunnel that is the difference
# between a slider and a wait.
#
# So each view's render callback takes the open tab as an input and asks :func:`should_render`
# first. Two things stop:
#
# * a view that is not on screen does not compute or ship anything, it renders when the reader
#   opens its tab, with whatever configuration is current by then;
# * a view whose inputs have not changed since it last drew does not redraw, so switching tabs back
#   and forth is free rather than a rebuild each way.
#
# The "since it last drew" signature is a ``dcc.Store`` per view, i.e. browser state, so two open
# browsers cannot suppress each other's renders the way one process-global would.


def render_key(*parts) -> str:
    """A stable string identity for everything a view's output depends on.

    JSON, not a tuple because it round-trips through a ``dcc.Store`` unchanged, a tuple
    comes back as a list and would never compare equal to the one just built.
    """
    return json.dumps(parts, sort_keys=True, default=str)


def should_render(view_id: str, active_tab: str | None, key: str, previous: str | None) -> bool:
    """True when *view_id* is the open tab **and** its inputs differ from the last render."""
    return active_tab == view_id and key != previous


def resolve(config: dict | None, bundle: dict) -> dict:
    """A valid configuration for *bundle*, starting from whatever the store holds."""
    default = bundle["default_config"]
    if not config:
        return {**default, "subset": frame_mod.ALL}

    models = [m for m in bundle["models"] if m in set(config.get("models") or ())]
    if not models:
        models = list(default["models"])
    rule = config.get("rule") if config.get("rule") in ("vote_k", "plurality") else default["rule"]
    k = int(config.get("k") or default["k"])
    return {
        "models": models,
        "k": max(1, min(k, len(models))),
        "rule": rule,
        "min_count": max(1, int(config.get("min_count") or default["min_count"])),
        "subset": resolve_subset(config.get("subset"), bundle),
    }


# ── the deep-dive subset ─────────────────────────────────────────────────────

def frame_or_none() -> dict | None:
    """The loaded sampling frame, or ``None``. Registry-level, so it outlives a cohort switch."""
    if not state.has_registry():
        return None
    return getattr(state.get_registry(), "frame", None)


def resolve_subset(subset: str | None, bundle: dict) -> str:
    """A subset value this cohort can actually honour, degrading to ``"all"``.

    ``store-config`` is browser state and outlives the cohort it was chosen for. Switch from HCY to
    GSC+ and a cell drawn on HCY names no report here at all, so it degrades, not emptying
    every screen, as a stale model subset or an out-of-range k already does.
    """
    frame = frame_or_none()
    subset = subset or frame_mod.ALL
    if subset == frame_mod.ALL or not frame_mod.applies_to(frame, bundle["report_ids"]):
        return frame_mod.ALL
    if subset == frame_mod.SAMPLE:
        return frame_mod.SAMPLE
    if subset in frame_mod.cells_present(frame, bundle["report_ids"]):
        return subset
    return frame_mod.ALL


def report_ids(bundle: dict, config: dict | None = None) -> list[str]:
    """The reports a view may show, **the only way a view should obtain them.**

    Reading ``bundle["report_ids"]`` directly bypasses the subset filter, so a view that does it
    silently reports over the whole cohort while the banner above it says otherwise. The one place
    that is *correct* is :mod:`verify`: the gates compare recomputed values against the shipped
    full-cohort artifacts, and filtering them would turn "the app agrees with the driver" into a
    guaranteed failure.
    """
    subset = (config or {}).get("subset") or frame_mod.ALL
    if subset == frame_mod.ALL:
        return list(bundle["report_ids"])
    return frame_mod.select(frame_or_none(), subset, bundle["report_ids"])


def subset_label(subset: str | None) -> str:
    """How a subset is named on screen and in a Copy-Markdown provenance line."""
    if not subset or subset == frame_mod.ALL:
        return "all reports"
    if subset == frame_mod.SAMPLE:
        return "deep-dive sample"
    return f"deep-dive cell {subset}"


def cell_chip(report_id: str, *, className: str = "pill pill-cell"):
    """The badge naming the deep-dive cell a report was drawn into, or ``None``.

    Drawn whenever a frame is loaded, including at ``subset="all"``, while browsing the whole
    cohort, "this one is in the sample" is the thing worth knowing.
    """
    frame = frame_or_none()
    cell = frame_mod.cell_of(frame, report_id)
    if not cell:
        return None
    return html.Span(cell, className=className,
                     title=frame_mod.describe_report(frame, report_id))


def describe(config: dict) -> str:
    """The provenance line every Copy-Markdown block carries. Numbers without it are unreadable.

    The subset is named here, not only in the banner because this line is what gets pasted
    into a findings file, where the banner is not. A table over 20 purposively drawn reports that
    arrives somewhere else labelled only by its configuration is a number without its note.
    """
    rule = ("per-sentence plurality" if config["rule"] == "plurality"
            else f"vote_k{config['k']} ({config['k']} of {len(config['models'])})")
    line = (f"{rule} · models: {', '.join(config['models'])} · "
            f"min_detection_count={config['min_count']}")
    subset = config.get("subset")
    if subset and subset != frame_mod.ALL:
        line += (f" · **{subset_label(subset)}, purposive, no aggregate here may be quoted**")
    return line


def is_shipped(config: dict, bundle: dict) -> bool:
    """True when this configuration is one the driver actually wrote a rule directory for."""
    if set(config["models"]) != set(bundle["models"]) or config["min_count"] != bundle["min_count"]:
        return False
    rule = "agg_plurality" if config["rule"] == "plurality" else f"vote_k{config['k']}"
    return rule in bundle["shipped_sets"]


# ── formatting ───────────────────────────────────────────────────────────────

def pct(x: float | None, digits: int = 1) -> str:
    """*x* (a share between 0 and 1) as a percentage, or a dash for None."""
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def num(x: float | None, digits: int = 3) -> str:
    """*x* to *digits* decimals, or a dash for None."""
    return "—" if x is None else f"{x:.{digits}f}"


def signed(x: float | None, digits: int = 3) -> str:
    """*x* to *digits* decimals with an explicit sign, or a dash for None."""
    return "—" if x is None else f"{x:+.{digits}f}"


def term(bundle: dict, hpo_id: str) -> str:
    """``HP:0001250 · Seizure`` where the ontology is available, the bare id where it is not."""
    tree = bundle.get("tree")
    if tree is None:
        return hpo_id
    from .. import relations

    name = relations.label(tree, hpo_id)
    return hpo_id if name == hpo_id else f"{hpo_id} · {name}"


def hpo_tooltip(bundle: dict, hpo_id: str) -> str:
    """What an HPO id means, as plain text for a native ``title=`` hover."""
    from .. import relations

    return relations.tooltip(bundle.get("tree"), hpo_id)


def info_dot(bundle: dict, hpo_id: str):
    """The ⓘ beside a term: hover for its definition and synonyms.

    An HPO id and a three-word label are not enough to judge whether a prediction is right, "Abnormality of the cardiovascular system" and "Abnormal heart morphology" are different claims
    and read the same at a glance. The definition is the thing that settles it, and it belongs
    where the term is, not on another screen.
    """
    return html.Span("ⓘ", className="info-dot", title=hpo_tooltip(bundle, hpo_id))


def term_with_info(bundle: dict, hpo_id: str):
    """``HP:0001250 · Seizure ⓘ``, the labelled term plus its hover."""
    return html.Span([term(bundle, hpo_id), info_dot(bundle, hpo_id)])


def term_chip(bundle: dict, hpo_id: str, *, style: dict | None = None, suffix: str = "",
              className: str = "pill pill-term"):
    """A term as a pill with its ⓘ, for the chip rows under the report.

    ``pill-term`` carries the colours. The bare ``.pill`` class is written for *coloured* chips, it sets white text and expects the caller to supply a background, so a chip drawn without one
    is white on white. Callers that pass their own ``style`` override this anyway. Those that do
    not get a readable chip in both themes.
    """
    return html.Span(
        [f"{term(bundle, hpo_id)}{suffix}", info_dot(bundle, hpo_id)],
        className=className, style=style or {})


def table(headers: list[str], rows: list[list], *, numeric_from: int = 1,
          scroll: bool = True) -> html.Div:
    """A plain HTML table, it ships the rows drawn and nothing else."""
    head = html.Tr([html.Th(h) for h in headers])
    body = [
        html.Tr([html.Td(cell, className="num" if i >= numeric_from else "")
                 for i, cell in enumerate(row)])
        for row in rows
    ]
    return html.Div(html.Table([html.Thead(head), html.Tbody(body)], className="tbl"),
                    className="tbl-scroll" if scroll else "")


def md_table(headers: list[str], rows: list[list]) -> str:
    """The same table as Markdown, for the Copy button, findings files want text, not a PNG."""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def model_chips(models: list[str], colors: dict, muted: list[str] | None = None) -> html.Div:
    """Coloured model pills. Anything in *muted* is drawn hollow, present but not voting."""
    muted = set(muted or ())
    return html.Div(
        [
            html.Span(
                m, className="pill",
                style=({"background": "transparent", "border": f"1px solid {colors.get(m, '#888')}",
                        "color": colors.get(m, "#888")} if m in muted
                       else {"background": colors.get(m, "#888")}),
            )
            for m in models
        ],
        style={"display": "flex", "flexWrap": "wrap", "gap": "4px"},
    )


def vote_chips(voters: int, decisive: int, models: list[str], colors: dict) -> html.Div:
    """One chip per model: filled when it voted, ringed when its vote was essential."""
    chips = []
    for i, model in enumerate(models):
        voted = voters >> i & 1
        is_decisive = decisive >> i & 1
        style = {"background": colors.get(model, "#888") if voted else "transparent",
                 "border": f"1px solid {colors.get(model, '#888')}",
                 "opacity": "1" if voted else "0.35"}
        if is_decisive:
            style["outline"] = f"2px solid {colors.get(model, '#888')}"
            style["outlineOffset"] = "1px"
        chips.append(html.Span(model[:4], className="pill", style=style,
                               title=f"{model}: " + ("decisive" if is_decisive
                                                     else "voted" if voted else "did not vote")))
    return html.Div(chips, style={"display": "flex", "gap": "3px"})


def empty_panel(message: str, panel_id: str, title: str):
    """A titled panel that shows *message*."""
    return theme.panel(title, theme.empty(message), panel_id=panel_id)
