"""Evaluation metrics: micro/macro F1 for HPO sets, binary metrics for UTI."""


def calc_metric(true_hpos: set, method_hpos: set) -> tuple[float, float, float]:
    """
    Compute precision, recall, F1 for two HPO code sets.

    Returns (precision, recall, f1). Both-empty case returns (1, 1, 1).
    """
    if not true_hpos and not method_hpos:
        return 1.0, 1.0, 1.0
    if not true_hpos or not method_hpos:
        return 0.0, 0.0, 0.0
    pr = len(true_hpos & method_hpos) / len(method_hpos)
    re = len(true_hpos & method_hpos) / len(true_hpos)
    f1 = (2 * pr * re / (pr + re)) if pr + re > 0 else 0.0
    return pr, re, f1


def evaluate_micro_macro(
    ground_truth_sets: list[set],
    predicted_sets: list[set],
) -> tuple[dict, dict]:
    """
    Compute micro and macro precision, recall, F1 over a list of samples.

    Returns:
        metrics: {micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1}
        individual_results: {precision: [...], recall: [...], F1: [...]}
    """
    tp_count = 0
    pred_count = 0
    actual_count = 0
    sum_recall = 0.0
    sum_precision = 0.0
    individual_results: dict[str, list] = {"precision": [], "recall": [], "F1": []}

    for true_hpos, pred_hpos in zip(ground_truth_sets, predicted_sets):
        pr, re, f1 = calc_metric(true_hpos, pred_hpos)
        tp_count += len(true_hpos & pred_hpos)
        pred_count += len(pred_hpos)
        actual_count += len(true_hpos)
        sum_recall += re
        sum_precision += pr
        individual_results["precision"].append(pr)
        individual_results["recall"].append(re)
        individual_results["F1"].append(f1)

    micro_precision = tp_count / pred_count
    micro_recall = tp_count / actual_count
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    macro_precision = sum_precision / len(ground_truth_sets)
    macro_recall = sum_recall / len(ground_truth_sets)
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

    metrics = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }
    return metrics, individual_results


def evaluate_UTI(symptomatic_patients, patient_data: list[tuple]) -> tuple[dict, dict]:
    """
    Binary UTI classification evaluation.

    Args:
        symptomatic_patients: Iterable of patient IDs predicted as UTI/CAUTI.
        patient_data: List of (patient_id, diagnosis_label) tuples.
                      Positive labels: "UTI", "CAUTI". Negative label: "AB".

    Returns:
        wrong: {false neg: [...], false pos: [...]}
        results: {Accuracy, Precision, Recall, F1}
    """
    pred_pos = set(symptomatic_patients)
    pred_neg = {entry[0] for entry in patient_data if entry[0] not in pred_pos}
    true_pos = {entry[0] for entry in patient_data if entry[1] in {"UTI", "CAUTI"}}
    true_neg = {entry[0] for entry in patient_data if entry[1] == "AB"}

    acc = (len(pred_pos & true_pos) + len(pred_neg & true_neg)) / len(patient_data)
    precision = len(pred_pos & true_pos) / len(pred_pos)
    recall = len(true_pos & pred_pos) / len(true_pos)
    f1 = 2 * precision * recall / (precision + recall)

    results = {"Accuracy": acc, "Precision": precision, "Recall": recall, "F1": f1}
    wrong = {
        "false neg": [p for p in true_pos if p not in pred_pos],
        "false pos": [p for p in true_neg if p not in pred_neg],
    }
    return wrong, results
