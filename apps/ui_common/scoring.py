"""Target-level HCY scoring with an injected HPOTree.

``HCYDataset.evaluate`` constructs a fresh ``HPOTree()`` on every call, a 15 MB JSON
reparse. That is fine once per experiment run, but the τ explorer calls this scoring path
once per grid point, so it must take the app's shared tree instead.

:func:`evaluate_targets` is a behaviour-preserving transcription of
``HCYDataset.evaluate`` (``src/hpo_extraction/evaluation/datasets/hcy.py:165``) with the tree passed in and
the console printing removed. :func:`assert_matches_dataset` proves the equivalence against
the real thing. The app asserts it once at startup.
"""

from __future__ import annotations

from functools import lru_cache

from hpo_extraction.evaluation.set_metrics import evaluate_micro_macro

_ZERO = {
    "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0,
    "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0,
}


def get_predictions(response_dict: dict, target_symptoms: list[str]) -> dict[str, list[str]]:
    """Target symptoms with at least one Yes verdict, per patient.

    Transcribes ``HCYDataset.get_predictions``, but tolerates a target missing from a
    patient's response dict (a flat baseline run need not have judged every target) rather
    than raising KeyError and taking the whole UI down.
    """
    predictions: dict[str, list[str]] = {}
    for key in response_dict:
        hits: list[str] = []
        for symptom in target_symptoms:
            for entry in response_dict[key].get(symptom, {}).values():
                if "Yes" in entry.get("response", ""):
                    hits.append(symptom)
        predictions[key] = hits
    return predictions


def build_target_dict(target_symptoms: tuple[str, ...], tree) -> dict[str, list[str]]:
    """``{target: [target] + all its descendants}``, the GT roll-up map.

    Descendant expansion is what lets a fine-grained GT code (e.g. a specific seizure type)
    count towards its coarse target symptom. Built once and cached: it is pure ontology.
    """
    return {
        target: [target] + list(tree.data[target]["Child"].keys())
        for target in target_symptoms
        if target in tree.data
    }


@lru_cache(maxsize=4)
def _cached_target_dict(target_symptoms: tuple[str, ...], tree) -> dict:
    """Cached per (target list, tree instance), HPOTree hashes by identity, and the app
    keeps one."""
    return build_target_dict(target_symptoms, tree)


def evaluate_targets(
    response_dict: dict,
    target_symptoms: list[str],
    gt_dict: dict[str, list[str]],
    tree,
    filter_unannotated: bool = True,
) -> dict:
    """Micro/macro P/R/F1 on the collapsed target set, the number experiments report.

    Identical to ``HCYDataset.evaluate(..., filter_unannotated=...)`` but with the tree
    injected. Returns zeros instead of raising when the corpus has no predictions at all
    (reachable from the τ explorer at an extreme τ_accept, where ``evaluate_micro_macro``
    would divide by a zero prediction count).
    """
    if filter_unannotated:
        gt_dict = {k: v for k, v in gt_dict.items() if v}
    if not gt_dict:
        return dict(_ZERO)

    predictions = get_predictions(response_dict, target_symptoms)
    target_dict = _cached_target_dict(tuple(target_symptoms), tree)

    gt_sets: list[set] = []
    pred_sets: list[set] = []
    for key in gt_dict:
        current_predictions = set(predictions.get(key, []))
        current_gt = set(gt_dict[key])
        gt_sets.append({
            target for target, valid in target_dict.items() if set(valid) & current_gt
        })
        pred_sets.append({
            target for target, valid in target_dict.items() if set(valid) & current_predictions
        })

    try:
        metrics, _ = evaluate_micro_macro(gt_sets, pred_sets)
    except ZeroDivisionError:
        return dict(_ZERO)
    return metrics


def assert_matches_dataset(dataset, response_dict, target_symptoms, tree) -> None:
    """Prove the transcription against the real ``HCYDataset.evaluate``. Raises on drift."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        reference = dataset.evaluate(
            response_dict, target_symptoms, output_dir=None, filter_unannotated=True,
        )
    ours = evaluate_targets(
        response_dict, target_symptoms, dataset.load_ground_truth(), tree, filter_unannotated=True,
    )
    for key, expected in reference.items():
        actual = ours.get(key)
        if actual is None or abs(actual - expected) > 1e-9:
            raise AssertionError(
                f"scoring.evaluate_targets diverged from HCYDataset.evaluate on {key}: "
                f"{actual} != {expected}"
            )
