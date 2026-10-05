"""Headless acceptance check: the gates, then every view, then the HTTP surface.

    python apps/phenojury_ui/app.py --selftest [--output-base <dir>]

Exits 0 only if all three pass. Run it on the cluster before trusting a screen, and in CI against
the synthetic fixture.

Why a render harness at all: a Dash callback that raises does not crash the process, it returns a
500 for one component and the page renders around the hole. The failure modes that costs are
the ones a reviewer will not notice (a panel silently missing, a table quietly empty), so
they have to be provoked. Each view exposes pure ``render_*`` functions and the
callbacks are thin wrappers over them, which is what makes driving them without a browser possible.

Three layers, because they catch different things:

**Gates**, the arithmetic (:mod:`apps.phenojury_ui.verify`). Recomputed == shipped.

**Render matrix**, every view at every interesting configuration, including the degenerate ones a
happy-path click-through never reaches: k=1 and k=N, plurality, a single-model subset, a report
with no annotations, a report where every model wrote nothing, and a cohort with a model missing.
A view must return a non-empty component for all of them.

**HTTP**, ``/``, ``/_dash-layout`` and ``/_dash-dependencies`` through Flask's test client. These
are what catch duplicate component ids and callbacks wired to targets that do not exist, neither of
which any amount of calling ``render_*`` will reveal.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback

from . import frame as frame_mod

logger = logging.getLogger(__name__)


def _versions() -> str:
    import dash
    import pandas
    import plotly

    return (f"dash {dash.__version__}, plotly {plotly.__version__}, "
            f"pandas {pandas.__version__}, python {sys.version.split()[0]}")


#: Two cells the selftest adds to whatever frame is loaded.
#:
#: ``selftest_some`` filters to a couple of real reports; ``selftest_none`` names a report that
#: cannot exist, so the subset resolves to nothing. Both paths must be swept **wherever this runs**,
#: and the real frame cannot supply them: it is drawn on HCY, which is patient data, so on GSC+ or
#: on a fixture there is no frame on disk at all. Building the cells from the cohort's own ids
#: exercises the code that counts, the filter, the empty result, and every view under both.
SELFTEST_SOME = "selftest_some"
SELFTEST_NONE = "selftest_none"


def selftest_frame(bundle: dict, real: dict | None = None) -> dict:
    """*real* (or an empty frame) with the two selftest cells added. Never written to disk."""
    real = real or {}
    ids = list(bundle["report_ids"])
    picks = dict(real.get("picks") or {})
    picks[SELFTEST_SOME] = ids[:2]
    picks[SELFTEST_NONE] = ["__no_such_report__"]
    return {
        "dir": real.get("dir", ""),
        "generated": real.get("generated", ""),
        "params": real.get("params") or {},
        "inputs": real.get("inputs") or {},
        "cell_sizes": real.get("cell_sizes") or {},
        "cells": list(picks),
        "picks": picks,
        "pools": real.get("pools") or {},
        "selected": sorted({r for group in picks.values() for r in group}),
        "cell_of": {r: cell for cell, group in picks.items() for r in group},
        "scored": real.get("scored") or {},
    }


def state_matrix(bundle: dict, frame: dict) -> list[tuple[str, dict, str]]:
    """``(name, raw config, subset)``, every state the views are rendered at.

    Two passes rather than a product. The configuration and the report subset are orthogonal, one chooses how the ensemble decides, the other which reports are on screen, so sweeping every
    combination would multiply the matrix by the cell count to re-prove that. The first pass is the
    existing configuration sweep, unfiltered. The second holds the shipped configuration and moves
    the subset, ending on a cell that selects nothing.
    """
    default = dict(bundle["default_config"])
    out = [(label, cfg, frame_mod.ALL) for label, cfg in config_matrix(bundle)]
    subsets = [("deep-dive sample", frame_mod.SAMPLE)]
    subsets += [(f"cell {cell}", cell)
                for cell in frame_mod.cells_present(frame, bundle["report_ids"])]
    subsets.append((f"cell {SELFTEST_NONE} (selects nothing)", SELFTEST_NONE))
    return out + [(f"shipped default · {name}", default, subset) for name, subset in subsets]


def config_matrix(bundle: dict) -> list[tuple[str, dict]]:
    """The configurations every view is rendered at. Named, so a failure says which one."""
    models = bundle["models"]
    n = len(models)
    if not n:
        return [("empty ensemble", {"models": [], "k": 1, "rule": "vote_k", "min_count": 1})]

    out = [
        ("shipped default", dict(bundle["default_config"])),
        ("k=1 (union)", {"models": models, "k": 1, "rule": "vote_k", "min_count": 1}),
        ("k=N (unanimity)", {"models": models, "k": n, "rule": "vote_k", "min_count": 1}),
        ("plurality", {"models": models, "k": 1, "rule": "plurality", "min_count": 1}),
        ("single model", {"models": models[:1], "k": 1, "rule": "vote_k", "min_count": 1}),
        # min_count above anything the data supports: every mask empties, so every view must cope
        # with predicting nothing at all.
        ("min_count=3", {"models": models, "k": 1, "rule": "vote_k", "min_count": 3}),
    ]
    if n > 2:
        out.append(("half the ensemble",
                    {"models": models[: n // 2], "k": 2, "rule": "vote_k", "min_count": 1}))
    return out


def _interesting_reports(bundle: dict) -> list[str]:
    """Reports worth rendering: the worst, one with no annotations, one nobody wrote about."""
    from .views.patient import rank_reports

    picks: list[str] = []
    ranked = rank_reports(bundle, bundle["default_config"])
    if ranked:
        picks.append(ranked[0][0])
        picks.append(ranked[-1][0])
    no_gold = [r for r in bundle["report_ids"] if not bundle["gold"].get(r)]
    silent = [r for r, wrote in bundle["wrote_any_by_report"].items() if not wrote]
    picks += no_gold[:1] + silent[:1]
    seen: list[str] = []
    for report_id in picks:
        if report_id and report_id not in seen:
            seen.append(report_id)
    return seen or bundle["report_ids"][:1]


def _nonempty(component) -> bool:
    """A rendered component that actually carries something. ``None`` and ``[]`` are failures."""
    if component is None:
        return False
    if isinstance(component, (list, tuple)):
        return bool(component)
    return True


def render_matrix(registry, run_ids: list[str]) -> tuple[list[str], int]:
    """Call every view's ``render_*`` over the state matrix. Returns ``(failures, n_attempts)``."""
    from .views import (
        annotate,
        compare_pb,
        falsepos,
        missed,
        models as models_view,
        patient,
        scorecard,
        terms,
        voters,
    )
    from .views.common import resolve

    failures: list[str] = []
    attempts = 0

    def attempt(what: str, fn, *args):
        nonlocal attempts
        attempts += 1
        try:
            result = fn(*args)
        except Exception as exc:  # noqa: BLE001 - collecting, not propagating
            failures.append(f"{what}: {type(exc).__name__}: {exc}\n"
                            + "".join(traceback.format_exc(limit=6)))
            return
        if not _nonempty(result):
            failures.append(f"{what}: rendered nothing")

    def attempt_data(what: str, fn, *args):
        """Like :func:`attempt`, but an empty result is a legitimate answer.

        A view *component* must always carry something, an empty panel with an explanation, not a
        hole in the page. A data function may legitimately return no rows, and asserting otherwise
        would make "this cohort predicted nothing" indistinguishable from a crash.
        """
        nonlocal attempts
        attempts += 1
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{what}: {type(exc).__name__}: {exc}\n"
                            + "".join(traceback.format_exc(limit=6)))
            return None

    for run_id in run_ids:
        bundle = registry.get_bundle(run_id)
        reports = _interesting_reports(bundle)
        # Install a frame for the duration of this cohort's sweep so the subset code is exercised
        # even where none is on disk, and put the real one back afterwards, the registry is
        # process-wide, and a selftest that left its fixture behind would go on filtering the
        # reader's screens.
        real_frame = getattr(registry, "frame", None)
        registry.frame = selftest_frame(bundle, real_frame)
        for label, raw_config, subset in state_matrix(bundle, registry.frame):
            # resolve() degrades a cell this cohort does not carry back to "all", which is right in
            # The app and wrong here: the empty subset is the case being tested.
            cfg = {**resolve(raw_config, bundle), "subset": subset}
            for mode in ("light", "dark"):
                where = f"{run_id} · {label} · {mode}"

                attempt(f"scorecard.gates [{where}]", scorecard.render_gates, bundle, mode)
                attempt(f"scorecard.stats [{where}]", scorecard.render_stats, bundle, cfg)
                attempt(f"scorecard.sweep [{where}]", scorecard.render_sweep, bundle, cfg, mode)
                attempt(f"scorecard.rules [{where}]", scorecard.render_rule_table, bundle, cfg)
                attempt(f"scorecard.spread [{where}]", scorecard.render_report_spread,
                        bundle, cfg, mode)

                attempt(f"models.funnel [{where}]", models_view.render_funnel, bundle, cfg, mode)
                attempt(f"models.archetypes [{where}]", models_view.render_archetypes,
                        bundle, cfg, mode)
                attempt(f"models.contribution [{where}]", models_view.render_contribution,
                        bundle, cfg, mode)
                attempt(f"models.behaviour [{where}]", models_view.render_behaviour, bundle, cfg)

                rows = _decisive_rows(bundle, cfg)
                attempt(f"voters.decisive [{where}]", voters.render_decisive,
                        bundle, cfg, rows, mode)
                attempt(f"voters.histogram [{where}]", voters.render_vote_histogram,
                        bundle, cfg, rows, mode)
                attempt(f"voters.coalitions [{where}]", voters.render_coalitions,
                        bundle, cfg, rows, mode)
                attempt(f"voters.agreement [{where}]", voters.render_agreement, bundle, cfg, mode)

                result = _autopsy(bundle, cfg)
                attempt(f"missed.fates [{where}]", missed.render_fates, bundle, cfg, result, mode)
                attempt(f"missed.blame [{where}]", missed.render_blame, bundle, cfg, result, mode)
                attempt(f"missed.recoverable [{where}]", missed.render_recoverable,
                        bundle, cfg, mode)

                attempt(f"missed.examples [{where}]", missed.render_examples, bundle, cfg, result)

                if bundle.get("tree") is not None:
                    fp_rows = falsepos.classify_all(bundle, cfg)
                    attempt(f"falsepos.mix [{where}]", falsepos.render_mix,
                            bundle, cfg, fp_rows, mode)
                    attempt(f"falsepos.distance [{where}]", falsepos.render_distance,
                            bundle, cfg, fp_rows, mode)
                    attempt(f"falsepos.by_model [{where}]", falsepos.render_by_model,
                            bundle, cfg, fp_rows, mode)
                    attempt(f"falsepos.across_k [{where}]", falsepos.render_across_k,
                            bundle, cfg, mode)
                    attempt(f"falsepos.examples [{where}]", falsepos.render_examples,
                            bundle, cfg, fp_rows)

                for report_id in reports:
                    for show_raw in (False, True):
                        for show_all in (False, True):
                            attempt(f"patient [{where} · {report_id} · raw={show_raw} · "
                                    f"all_sentences={show_all}]",
                                    patient.render_report, bundle, cfg, report_id, mode, show_raw,
                                    show_all)
                    # The two focus states are the page's new dimensions and its emptiest ones: a
                    # sentence nobody wrote about, and a term filter matching nothing. Both must
                    # still render a panel that explains itself.
                    for sent_focus in _focus_sentences(bundle, report_id):
                        attempt(f"patient [{where} · {report_id} · sentence={sent_focus}]",
                                patient.render_report, bundle, cfg, report_id, mode, False, False,
                                sent_focus, None)
                    for term_focus in _focus_terms(bundle, cfg, report_id):
                        attempt(f"patient [{where} · {report_id} · term={term_focus}]",
                                patient.render_report, bundle, cfg, report_id, mode, False, False,
                                None, term_focus)
                    attempt_data(f"patient.ordered [{where}]", patient.ordered_reports, bundle, cfg)

                # The control bodies. Their callbacks are driven by pattern-matching inputs, which
                # callback_smoke cannot dispatch, so this is the only place they are exercised.
                ids = [r for r, _ in patient.ordered_reports(bundle, cfg)]
                for trigger in ("pa-prev", "pa-next"):
                    for at in (ids[:1] + ids[-1:] + ["nope"]):
                        attempt_data(f"patient.step {trigger} [{where} · {at}]",
                                     patient.step_to, trigger, at, ids)
                for trigger in ({"type": "pa-term", "hpo": "HP:0001250"}, "pa-term-clear"):
                    for current in (None, "HP:0001250"):
                        attempt_data(f"patient.toggle {trigger} [{where}]", patient.toggle_focus,
                                     trigger, current, "hpo", ("pa-term-clear", "pa-report"))

                attempt(f"compare_pb.all [{where}]", compare_pb.render_all, bundle, cfg, mode,
                        "delta")
                if bundle.get("pb_standalone"):
                    attempt(f"compare_pb.totals [{where}]", compare_pb.render_totals, bundle, cfg)
                    attempt(f"compare_pb.agreement [{where}]", compare_pb.render_agreement,
                            bundle, cfg, mode)
                    for order in ("delta", "-delta", "report"):
                        attempt(f"compare_pb.reports [{where} · {order}]",
                                compare_pb.render_reports, bundle, cfg, order)
                    attempt(f"compare_pb.terms [{where}]", compare_pb.render_terms, bundle, cfg)
                    # Paging is server-side, so page 0 alone never exercises the slice the browser
                    # actually asks for.
                    for page in (0, 1, 99):
                        attempt_data(f"compare_pb.page {page} [{where}]", compare_pb._page,
                                     compare_pb.report_rows(bundle, cfg), page, ())
                else:
                    attempt(f"compare_pb.missing [{where}]", compare_pb.render_missing, bundle)

                term_rows = attempt_data(f"terms.rows [{where}]", terms.build_rows, bundle, cfg)
                for filters in (([], [], []), (["FP", "FN"], [], []),
                                (["TP"], cfg["models"][:1], []), (["FP"], [], ["unrelated"])):
                    filtered = terms.filter_rows(term_rows or [], *filters)
                    attempt(f"terms.table [{where} · {filters[0] or 'all'}]", terms.render_table,
                            filtered, cfg)
                    # Paging is server-side, so the page slice is code the browser depends on and
                    # render_table alone never exercises past page 0.
                    for page in (0, 1, 99):
                        attempt_data(f"terms.page {page} [{where}]", terms.page_of,
                                     terms.sort_rows(filtered, [{"column_id": "n_votes",
                                                                 "direction": "desc"}]), page)

                # Annotate, over an empty log and a synthetic one. The empty case is what a reader
                # opens on. The populated case is the only way the agreement panel and the verdict
                # block are reached at all, and neither is exercised by a log this selftest must
                # not write to a real run directory.
                for folded_label, folded in _annotation_states(bundle):
                    aw = f"{where} · {folded_label}"
                    for which in ("todo", "done", "all"):
                        rows = attempt_data(
                            f"annotate.queue [{aw} · {which}]", annotate.queue_rows,
                            bundle, cfg, folded, cfg["models"][:1], which)
                        attempt(f"annotate.render_queue [{aw} · {which}]",
                                annotate.render_queue, rows or [], None)
                    attempt_data(f"annotate.progress [{aw}]", annotate.render_progress,
                                 bundle, cfg, folded, cfg["models"][:1])
                    attempt(f"annotate.agreement [{aw}]", annotate.render_agreement,
                            bundle, cfg, folded, mode)
                    # No key, a key that does not parse, one naming a reply that does not exist,
                    # and every key the log actually holds.
                    keys = [None, "not-a-key", "run|nomodel|noreport|0"] + list(folded["replies"])
                    for key in keys:
                        attempt(f"annotate.reply [{aw} · {key}]", annotate.render_reply,
                                bundle, cfg, key, folded, mode)
                        attempt_data(f"annotate.parse [{aw} · {key}]", annotate.parse_key, key)
        registry.frame = real_frame
    return failures, attempts


def _annotation_states(bundle: dict) -> list[tuple[str, dict]]:
    """``(name, folded state)``, an empty annotation log, and one with a reply of each kind.

    Built in memory, not on disk. The log lives in the *run* directory by default, and a
    selftest that wrote into a real experiment's output would leave annotations nobody made.
    """
    from . import annotations as ann

    empty = {"replies": {}, "n_skipped": 0}
    reply = next(iter(bundle["built"]["replies"]), None)
    if reply is None:
        return [("no annotations", empty)]
    model, report_id, sent_num = reply
    base = {"run_id": bundle["run_id"], "cohort": bundle["cohort"],
            "prompt_key": bundle.get("prompt_key", ""), "model": model,
            "report_id": report_id, "sentence_number": sent_num,
            "note": "", "author": "selftest", "updated_at": ""}
    # A term PhenoBERT also found, one it did not, and a skipped reply, so agreement, a miss and
    # The empty-denominator path are all reached.
    found = sorted(ann.phenobert_terms(bundle, model, report_id, sent_num)["positive"])
    key = ann.key_of(bundle["run_id"], model, report_id, sent_num)
    annotated = {"replies": {key: {**base, "key": key, "status": "annotated",
                                   "hpo_codes": found[:1] + ["HP:0001250"]}}, "n_skipped": 0}
    skipped = {"replies": {key: {**base, "key": key, "status": "skipped", "hpo_codes": []}},
               "n_skipped": 0}
    return [("no annotations", empty), ("one annotated", annotated), ("one skipped", skipped)]


# Values fed to each callback input/state, keyed by ``(component id, property)``. A callback whose
# inputs are not all covered here is skipped and *reported*, so adding a control to the sidebar
# without teaching the selftest about it cannot silently drop that callback from the check.
def _focus_sentences(bundle: dict, report_id: str) -> tuple:
    """A sentence that carries evidence and one that does not, both are drawable states."""
    sentences = sorted(bundle["sentences"].get(report_id, {}))
    if not sentences:
        return (None,)
    return (sentences[0], sentences[-1])


def _focus_terms(bundle: dict, cfg: dict, report_id: str) -> tuple:
    """A term in play on this report, and one that is not, the empty filter must still explain."""
    from . import votes as votes_mod

    gold = set(bundle["gold"].get(report_id, ()))
    predicted = votes_mod.predicted_sets(bundle, cfg).get(report_id, set())
    in_play = sorted(gold | predicted)
    return ((in_play[0] if in_play else "HP:0000001"), "HP:0000001")


def _input_values(registry, run_id: str, report_id: str | None, tab: str = "scorecard") -> dict:
    from .app import VIEWS

    values = {
        ("run-a", "value"): run_id,
        ("store-config", "data"): None,          # forces the resolve()-from-nothing path
        ("store-theme", "data"): "light",
        ("store-focus", "data"): report_id,
        ("tabs", "value"): tab,
        ("cfg-rule", "value"): "vote_k",
        ("cfg-k", "value"): 2,
        ("cfg-models", "value"): None,
        ("cfg-mincount", "value"): 1,
        ("cfg-reset", "n_clicks"): 1,
        ("src-load", "n_clicks"): 1,
        ("src-output", "value"): registry.output_base,
        ("src-pb", "value"): registry.pb_base,
        ("src-frame", "value"): registry.frame_base,
        # "all reports", not a cell, because the render matrix sweeps the subsets itself and
        # this table drives the *callback* smoke, where the point is that every wire is connected.
        ("cfg-subset", "value"): frame_mod.ALL,
        ("pa-report", "value"): report_id,
        ("pa-report", "options"): ([{"label": report_id, "value": report_id}]
                                   if report_id else []),
        ("pa-raw", "value"): [],
        ("pa-sentences", "value"): [],
        ("pa-prev", "n_clicks"): 1,
        ("pa-next", "n_clicks"): 1,
        ("pa-term-clear", "n_clicks"): 1,
        ("pa-sent-clear", "n_clicks"): 1,
        ("pa-sent-focus", "data"): None,
        ("pa-term-focus", "data"): None,
        ("pb-sort", "value"): "delta",
        ("pb-report-table", "active_cell"): {"row": 0, "column": 0},
        ("pb-report-table", "data"): [{"report_id": report_id}] if report_id else [],
        ("pb-report-table", "page_current"): 1,
        ("pb-report-table", "sort_by"): [{"column_id": "delta_f1", "direction": "desc"}],
        ("pb-term-table", "page_current"): 1,
        ("pb-term-table", "sort_by"): [{"column_id": "pb_hits", "direction": "desc"}],
        ("te-outcome", "value"): ["FP", "FN"],
        ("te-models", "value"): [],
        ("te-note", "value"): [],
        ("te-table", "active_cell"): {"row": 0, "column": 0},
        ("te-table", "data"): [{"report_id": report_id}] if report_id else [],
        ("te-table", "page_current"): 1,
        ("te-table", "sort_by"): [{"column_id": "n_votes", "direction": "desc"}],
        ("theme-toggle", "value"): "light",
        ("present-btn", "n_clicks"): 1,
        ("an-models", "value"): [],
        ("an-filter", "value"): "todo",
        ("an-current", "data"): None,
        ("an-saved", "data"): 0,
        ("an-hpo", "value"): [],
        # A real query, not "": the search short-circuits below MIN_QUERY, so an empty string would
        # dispatch the callback without ever reaching the index it exists to exercise.
        ("an-hpo", "search_value"): "seizure",
        ("an-note", "value"): "",
        ("an-save", "n_clicks"): 1,
        ("an-skip", "n_clicks"): 1,
    }
    # Never rendered before, so should_render() lets every gated callback through on its own tab.
    values.update({(f"sig-{vid}", "data"): None for vid, _, _ in VIEWS})
    return values


def _parse_output_spec(spec: str) -> list[dict]:
    """``'..a.children...b.data..'`` → ``[{'id': 'a', ...}, {'id': 'b', ...}]``."""
    parts = spec[2:-2].split("...") if spec.startswith("..") else [spec]
    out = []
    for part in parts:
        component_id, _, prop = part.rpartition(".")
        out.append({"id": component_id, "property": prop})
    return out


def callback_smoke(app, registry, run_id: str, report_id: str | None) -> tuple[list[str], int]:
    """Drive every server-side callback through Dash's own dispatch endpoint, on every tab.

    ``render_*`` covers the view logic. This covers the *wiring*, a callback bound to an id that
    no longer exists, an output arity that does not match what the function returns, a value shape
    Dash cannot serialise. None of those show up until a browser asks, and by then the page has a
    hole in it, not a traceback.

    Every tab, because the view callbacks are gated on which one is open: dispatched with the wrong
    tab they raise ``PreventUpdate`` and Dash answers 204, which proves nothing. So each callback
    is dispatched once per tab and must come back 200 on at least one of them, a callback that
    never runs anywhere is reported, which is what stops the gating from quietly hiding a view from
    its own acceptance check.
    """
    from .app import VIEWS

    client = app.server.test_client()
    problems: list[str] = []
    dispatched = 0
    ran: dict[str, bool] = {}
    skipped: set[str] = set()

    for tab, _, _ in VIEWS:
        values = _input_values(registry, run_id, report_id, tab)
        for spec, entry in app.callback_map.items():
            # Clientside callbacks carry no server function (Dash stores them without a "callback"
            # key), and pattern-matching outputs need a real component instance to match against.
            # Both need a browser. Neither can be dispatched here.
            if entry.get("clientside_function") or "callback" not in entry or "{" in spec:
                continue
            specs = entry["inputs"] + list(entry.get("state") or [])
            # A pattern-matching *input* needs a real component instance to match against,
            # like a pattern-matching output, Dash resolves ``ALL`` from the rendered layout, which
            # this harness does not have. Those callbacks are one line each by design and
            # their bodies (``patient.toggle_focus``) are driven directly by the render matrix, so
            # skipping them here loses nothing. Reporting them as unchecked would be noise that
            # trains the reader to ignore the message that counts.
            if any("{" in s["id"] for s in specs):
                skipped.add(spec)
                continue
            unknown = [(s["id"], s["property"]) for s in specs
                       if (s["id"], s["property"]) not in values]
            if unknown:
                if spec not in ran:
                    problems.append(f"{spec}: selftest has no value for {unknown}, callback not "
                                    "checked")
                    ran[spec] = True    # reported once, not once per tab
                continue

            outputs = _parse_output_spec(spec)
            body = {
                "output": spec,
                "outputs": outputs if len(outputs) > 1 else outputs[0],
                "inputs": [{**s, "value": values[(s["id"], s["property"])]}
                           for s in entry["inputs"]],
                "state": [{**s, "value": values[(s["id"], s["property"])]}
                          for s in (entry.get("state") or [])],
                "changedPropIds": [f"{entry['inputs'][0]['id']}.{entry['inputs'][0]['property']}"],
            }
            try:
                response = client.post("/_dash-update-component", json=body)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{spec} [tab={tab}]: dispatch raised {type(exc).__name__}: {exc}")
                continue
            dispatched += 1
            # 204 is Dash's answer to PreventUpdate, which is the correct answer for a view that
            # is not the open tab. Anything else is a real failure.
            if response.status_code == 200:
                ran[spec] = True
            elif response.status_code != 204:
                problems.append(f"{spec} [tab={tab}]: dispatch returned {response.status_code}, "
                                f"{response.data[:400]!r}")

    never_ran = sorted(spec for spec, entry in app.callback_map.items()
                       if not entry.get("clientside_function") and "callback" in entry
                       and "{" not in spec and spec not in skipped and not ran.get(spec))
    for spec in never_ran:
        problems.append(f"{spec}: PreventUpdate on every tab, the callback body was never "
                        "executed, so nothing about it was checked")
    return problems, dispatched


def _decisive_rows(bundle: dict, cfg: dict):
    from . import votes as votes_mod

    return votes_mod.decisive_rows(bundle, cfg, bundle["gold"])


def _autopsy(bundle: dict, cfg: dict):
    from . import autopsy as autopsy_mod
    from . import votes as votes_mod

    return autopsy_mod.build(bundle, cfg, bundle["gold"],
                             votes_mod.predicted_sets(bundle, cfg))


def http_smoke(app) -> list[str]:
    """Boot the Flask test client and pull the three routes Dash serves its wiring from."""
    problems: list[str] = []
    client = app.server.test_client()

    for route in ("/", "/_dash-layout", "/_dash-dependencies"):
        try:
            response = client.get(route)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"GET {route} raised {type(exc).__name__}: {exc}")
            continue
        if response.status_code != 200:
            problems.append(f"GET {route} returned {response.status_code}")
        elif not response.data:
            problems.append(f"GET {route} returned an empty body")

    # Duplicate ids are legal Python and a broken page: Dash resolves them non-deterministically,
    # so a callback can end up writing to whichever copy it happens to bind.
    try:
        layout_json = json.loads(client.get("/_dash-layout").data)
        seen: set[str] = set()
        duplicates: set[str] = set()

        def walk(node):
            if isinstance(node, dict):
                props = node.get("props", {})
                node_id = props.get("id")
                if isinstance(node_id, str):
                    (duplicates if node_id in seen else seen).add(node_id)
                # Every prop value is walked once, "children" included. Descending into
                # children *and* then into every prop would visit each node twice and report the
                # whole layout as duplicated.
                for key, value in props.items():
                    if key != "id" and isinstance(value, (list, dict)):
                        walk(value)
            elif isinstance(node, list):
                for child in node:
                    walk(child)

        walk(layout_json)
        if duplicates:
            problems.append(f"duplicate component id(s) in the layout: {sorted(duplicates)}")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"could not inspect the layout for duplicate ids: {exc!r}")

    return problems


def unit_checks() -> list[str]:
    """The two new modules, on data built here. Returns the failures, empty when all pass.

    Neither can be exercised by the layers above where this usually runs. The sampling frame is an
    HCY artifact and HCY is patient data, so on GSC+ there is no ``frame.json`` to read. And the
    annotation store writes, which a selftest must not do inside somebody's real run directory.
    Both are pure functions over small inputs, so they are checked directly instead.
    """
    import json
    import tempfile

    from . import annotations as ann

    failures: list[str] = []

    def check(name, condition):
        if not condition:
            failures.append(name)

    # ── the frame ────────────────────────────────────────────────────────────
    base = tempfile.mkdtemp()
    frame_dir = os.path.join(base, frame_mod.FRAME_DIRNAME)
    os.makedirs(frame_dir, exist_ok=True)
    with open(os.path.join(frame_dir, frame_mod.FRAME_FILE), "w", encoding="utf-8") as handle:
        json.dump({"generated": "2026-09-09", "params": {"seed": 1},
                   "picks": {"pb_best": ["r1", "r2"], "family": ["r9"]},
                   "selected": ["r1", "r2", "r9"]}, handle)
    with open(os.path.join(frame_dir, frame_mod.POOLS_FILE), "w", encoding="utf-8",
              newline="") as handle:
        handle.write("patient_id,selected,cell,n_gold,n_pred,tp,fp,fn,precision,recall,f1,"
                     "family,lab_value,implicit,negated\n")
        handle.write("r1,1,pb_best,8,7,7,0,1,1.0,0.875,0.933,0,0,0,0\n")
        handle.write("r9,1,family,0,3,,3,,,,,2,0,0,1\n")

    check("frame is found beside the output base", frame_mod.find_frame(base) == frame_dir)
    check("an override that matches nothing does not fall back",
          frame_mod.find_frame(base, override=os.path.join(base, "nowhere")) is None)
    frame = frame_mod.load(frame_dir)
    check("frame loads", frame is not None)
    ids = ["r1", "r2", "r3", "r9"]
    check("all is unfiltered", frame_mod.select(frame, frame_mod.ALL, ids) == ids)
    check("the sample is the draw", frame_mod.select(frame, frame_mod.SAMPLE, ids)
          == ["r1", "r2", "r9"])
    check("a cell is its own picks", frame_mod.select(frame, "pb_best", ids) == ["r1", "r2"])
    check("selection keeps the caller's order",
          frame_mod.select(frame, frame_mod.SAMPLE, ["r9", "r1"]) == ["r9", "r1"])
    check("an unknown cell selects nothing", frame_mod.select(frame, "nope", ids) == [])
    check("a frame for another cohort does not apply",
          not frame_mod.applies_to(frame, ["g1", "g2"]))
    check("cells_present skips cells with no report here",
          frame_mod.cells_present(frame, ["r9"]) == ["family"])
    check("cell_of inverts the draw", frame_mod.cell_of(frame, "r9") == "family")
    check("a report outside the draw has no cell", frame_mod.cell_of(frame, "r3") == "")
    # A blank precision in pools.csv means undefined, not zero, a report with no ground truth would
    # otherwise sort to the bottom of a ranking it does not belong in.
    check("a blank score reads as None", frame["scored"]["r9"]["f1"] is None)
    check("no frame means no filtering", frame_mod.select(None, "pb_best", ids) == ids)

    # ── the annotation store ─────────────────────────────────────────────────
    store_dir = tempfile.mkdtemp()
    log = ann.AnnotationLog(store_dir, author="selftest")
    key = ann.key_of("run1", "llama", "r1", 3)
    fields = {"key": key, "run_id": "run1", "cohort": "hcy", "prompt_key": "",
              "model": "llama", "report_id": "r1", "sentence_number": 3}

    log.append("annotate", hpo_codes=["HP:0001252", "HP:0000750"], **fields)
    check("an annotation is recorded",
          log.fold()["replies"][key]["hpo_codes"] == ["HP:0001252", "HP:0000750"])
    log.append("annotate", hpo_codes=["HP:0001252"], **fields)
    check("annotate replaces rather than merges",
          log.fold()["replies"][key]["hpo_codes"] == ["HP:0001252"])
    log.append("skip", **fields)
    folded = log.fold()
    check("skip is its own status", folded["replies"][key]["status"] == "skipped")
    check("a skipped reply counts as read", key in ann.visited(folded))
    check("an unvisited reply does not",
          ann.key_of("run1", "llama", "r1", 9) not in ann.visited(folded))

    with open(log.path, "a", encoding="utf-8") as handle:
        handle.write('{"action": "annot')
    check("a torn final line is skipped, not fatal",
          len(ann.AnnotationLog(store_dir, author="other").events) == 3)

    bundle = {"run_id": "run1", "tree": None, "pb_index": {"by_sentence": {
        ("llama", "r1", 3): [{"hpo_id": "HP:0001252", "negated": False},
                             {"hpo_id": "HP:0012759", "negated": False},
                             {"hpo_id": "HP:0002376", "negated": True}]}}}
    log.append("annotate", hpo_codes=["HP:0001252", "HP:0000750"], **fields)
    result = ann.compare(bundle, log.fold())
    row = result["rows"][0]
    check("agreement is the intersection", row["n_agreed"] == 1)
    check("what the reader saw and PhenoBERT did not", row["pb_missed"] == ["HP:0000750"])
    check("what PhenoBERT added", row["pb_spurious"] == ["HP:0012759"])
    check("a negated detection is neither", row["pb_negated"] == ["HP:0002376"]
          and row["n_phenobert"] == 2)
    totals = result["totals"]
    check("PhenoBERT precision is 1 of 2", abs(totals["precision"] - 0.5) < 1e-9)
    check("PhenoBERT recall is 1 of 2", abs(totals["recall"] - 0.5) < 1e-9)
    check("another run's annotations are not scored here",
          ann.compare({**bundle, "run_id": "other"}, log.fold())["n_replies"] == 0)
    check("a subset filter reaches the comparison",
          ann.compare(bundle, log.fold(), report_ids=["r2"])["n_replies"] == 0)
    check("an empty reading gives None, not 0.0",
          ann._score(ann._empty_tally())["precision"] is None)

    log.commit(ann.term_rows(log.fold()), ann.agreement_rows(result))
    check("both derived files are written",
          os.path.isfile(log.terms_path) and os.path.isfile(log.agreement_path))
    return failures


def run_selftest(registry, build_app) -> bool:
    """Gates → render matrix → HTTP. Logs every failure. Returns True only if all three pass."""
    from . import votes as votes_mod

    logging.getLogger().setLevel(logging.INFO)
    logger.info("selftest | %s", _versions())

    # Worth stating up front on the cluster: whether the gates are fixed to the driver's own
    # functions, or running against this app's equivalents because the training stack would not
    # import. Both are valid runs. They carry different guarantees.
    pinned, why_not = votes_mod.driver_status()
    logger.info("selftest | driver pinning: %s",
                "core.slm_ensemble_experiment imported — gates are pinned to its functions"
                if pinned else f"NOT fixed ({why_not}), using this app's equivalents")

    run_ids = registry.run_ids()
    if not run_ids:
        logger.error("selftest | no Free Listing generation cohorts under %s, nothing to verify",
                     registry.output_base)
        return False
    logger.info("selftest | %d cohort(s): %s", len(run_ids), ", ".join(run_ids))

    ok = True

    # 0, the two modules the layers below cannot reach on this data.
    unit_failures = unit_checks()
    for failure in unit_failures:
        logger.error("selftest | unit | %s", failure)
    logger.info("selftest | frame + annotation checks: %d failure(s)", len(unit_failures))
    ok = ok and not unit_failures

    # 1, the gates, per cohort.
    for run_id in run_ids:
        bundle = registry.get_bundle(run_id)
        for gate in bundle["gates"]:
            level = logging.ERROR if gate["status"] == "fail" else logging.INFO
            logger.log(level, "selftest | %s | %-4s %s, %s",
                       run_id, gate["status"].upper(), gate["title"], gate["detail"])
        if bundle["gate_summary"]["n_fail"]:
            ok = False
        if not bundle["models"]:
            # A data state, not an app defect: a cohort still extracting has nothing to vote with.
            # The app must open on it and say so, which the render matrix below checks, so this
            # is a warning, and only a gate failure or a render failure fails the selftest.
            logger.warning("selftest | %s | no voting models yet, the aggregate stage has not "
                           "run, so every view will show its empty state", run_id)

    # 2, every view, every configuration.
    failures, attempts = render_matrix(registry, run_ids)
    for failure in failures:
        logger.error("selftest | render | %s", failure)
    logger.info("selftest | render matrix: %d renders, %d failure(s)", attempts, len(failures))
    ok = ok and not failures and attempts > 0

    # 3, the HTTP surface and every server-side callback.
    try:
        app = build_app(registry)
        problems = http_smoke(app)
        first = registry.get_bundle(run_ids[0])
        report_id = first["report_ids"][0] if first["report_ids"] else None
        cb_problems, dispatched = callback_smoke(app, registry, run_ids[0], report_id)
        problems += cb_problems
        logger.info("selftest | dispatched %d server-side callback(s)", dispatched)
    except Exception as exc:  # noqa: BLE001
        logger.exception("selftest | the app could not be constructed")
        problems = [f"build_app raised {type(exc).__name__}: {exc}"]
    for problem in problems:
        logger.error("selftest | http | %s", problem)
    logger.info("selftest | http smoke: %d problem(s)", len(problems))
    ok = ok and not problems

    logger.info("selftest | %s", "PASS" if ok else "FAIL")
    return ok
