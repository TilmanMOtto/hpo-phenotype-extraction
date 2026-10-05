"""Evaluation dataset loader for the HCY (hypercholesterolemia) dataset."""

import ast
import csv
import os

import pandas as pd

from hpo_extraction.evaluation.datasets.base import EvalDataset
from hpo_extraction.evaluation.set_metrics import evaluate_micro_macro
from hpo_extraction.ontology.hpo_tree import HPOTree


def load_ground_truth_files_HCY(dir_path: str) -> tuple[dict, list, list]:
    """
    Load ground truth CSV files (semicolon or comma delimited).

    Returns:
        gt_dict: {file_key → {hpo_codes: [...], manual_verification: [...]}}
        gt_csv_files: sorted list of CSV filenames
        failed_files: list of filenames that failed to load
    """
    gt_dict: dict = {}
    failed_files: list[str] = []
    gt_csv_files = sorted(f for f in os.listdir(dir_path) if f.endswith(".csv"))

    for file in gt_csv_files:
        file_path = os.path.join(dir_path, file)
        for delimiter in (";", ","):
            try:
                with open(file_path, "r") as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    gt_dict[file[:6]] = {"hpo_codes": [], "manual_verification": []}
                    for row in reader:
                        gt_dict[file[:6]]["hpo_codes"].append(row["hpo_codes"])
                        gt_dict[file[:6]]["manual_verification"].append(row["manual_verification"])
                break
            except Exception:
                continue
        else:
            failed_files.append(file)
            print(f"Failed to read: {file_path}")

    return gt_dict, gt_csv_files, failed_files


def normalize_hpo_codes(gt_dict: dict) -> dict:
    """Normalize HPO code strings: handle list-like strings and fix missing colons."""
    for key, data in gt_dict.items():
        normalized: list[str] = []
        for code in data["hpo_codes"]:
            code = code.replace("\u2018", "'").replace("\u2019", "'")
            try:
                evaluated = ast.literal_eval(code)
                normalized.append(evaluated[0] if isinstance(evaluated, list) and evaluated else code)
            except (ValueError, SyntaxError):
                normalized.append(code)
        # Fix missing colons
        normalized = [
            code if ":" in code or not code else code[:2] + ":" + code[2:]
            for code in normalized
        ]
        gt_dict[key]["hpo_codes"] = normalized
    return gt_dict


def filter_gt_dict_entries(
    gt_dict: dict,
    isin_values: list,
    deletion: bool = False,
) -> tuple[dict, list, list]:
    """
    Filter HPO entries by manual_verification value and optionally delete empty entries.

    Returns:
        gt_dict: Updated dict.
        files_empty: Keys with no entries after filtering.
        files_missingID: Keys that had empty HPO code strings.
    """
    files_empty: list[str] = []
    files_missingID: list[str] = []

    for key, data in list(gt_dict.items()):
        filtered_codes: list[str] = []
        filtered_mv: list[str] = []
        for hpo_code, mv in zip(data["hpo_codes"], data["manual_verification"]):
            if mv in isin_values:
                if hpo_code.strip():
                    filtered_codes.append(hpo_code)
                    filtered_mv.append(mv)
                else:
                    files_missingID.append(key)

        data["hpo_codes"] = filtered_codes
        data["manual_verification"] = filtered_mv
        entry_deleted = False

        if not data["hpo_codes"]:
            files_empty.append(key)
            if deletion:
                del gt_dict[key]
                entry_deleted = True

        if not entry_deleted:
            gt_dict[key] = pd.DataFrame.from_dict(data)

    return gt_dict, files_empty, sorted(set(files_missingID))


def target_code_sets(target_symptoms: list[str], hpo_tree: HPOTree) -> dict[str, set[str]]:
    """``{target: {target} ∪ every descendant}``, the codes that count as each target.

    ``hpo.json``'s ``Child`` is the full descendant set (``Son`` holds the direct children), so a
    annotated term anywhere below a target counts for it. That is the rule PhenoRAG's published HCY
    number was computed under. Changing it would move that number.
    """
    return {
        target: {target, *hpo_tree.data[target]["Child"].keys()}
        for target in target_symptoms
    }


def project_onto_targets(
    gt_dict: dict,
    predictions: dict,
    target_dict: dict,
    report_ids: list[str] | None = None,
) -> tuple[list[str], list[set], list[set]]:
    """Project ground truth and predictions onto the target list: ``(report_ids, gold_sets, pred_sets)``.

    A target is in a report's projected set when that report carries any of the target's codes
    (``target_dict``). ``report_ids`` defaults to the ground truth's own keys, in order, the cohort
    :meth:`HCYDataset.evaluate` scores. A report with no predictions projects to the empty set.
    """
    ids = list(gt_dict) if report_ids is None else list(report_ids)
    gold_sets: list[set] = []
    pred_sets: list[set] = []
    for key in ids:
        current_predictions = set(predictions.get(key, []))
        current_gt = set(gt_dict[key])
        gold_sets.append({t for t, codes in target_dict.items() if set(codes) & current_gt})
        pred_sets.append({t for t, codes in target_dict.items() if set(codes) & current_predictions})
    return ids, gold_sets, pred_sets


class HCYDataset(EvalDataset):
    """
    Evaluation loader for the HCY (hypercholesterolemia) dataset.

    Args:
        gt_path: Path to the pickle file containing the ground truth dict.
        target_symptoms_path: Path to the target symptoms CSV (column: target_codes).
    """

    def __init__(self, gt_path: str, target_symptoms_path: str):
        self.gt_path = gt_path
        self.target_symptoms_path = target_symptoms_path

    def load_ground_truth(self) -> dict[str, list[str]]:
        """
        Load ground truth from a pickle (.pkl) or CSV file.

        CSV format: first column = patient ID, second column = HPO codes as a
        semicolon- or comma-separated string (or a Python list literal).
        """
        import ast
        import pickle

        if self.gt_path.endswith(".csv"):
            df = pd.read_csv(self.gt_path)
            gt: dict[str, list[str]] = {}
            id_col = df.columns[0]
            hpo_col = df.columns[1]
            for _, row in df.iterrows():
                key = str(row[id_col]).strip()
                cell = row[hpo_col]
                # A patient with no annotated terms is a real row, and an empty cell is how it is
                # written. Falling through to `str(NaN)` would coin the annotated term "nan" and charge
                # every method a false negative for not predicting it, which is the shape
                # of the curation export, one row per patient including the un-curated ones.
                if pd.isna(cell):
                    gt[key] = []
                    continue
                raw = str(cell).strip()
                if not raw:
                    gt[key] = []
                    continue
                # Handle Python list literal, semicolon-sep, or comma-sep values
                try:
                    codes = ast.literal_eval(raw)
                    if isinstance(codes, str):
                        codes = [codes]
                except (ValueError, SyntaxError):
                    sep = ";" if ";" in raw else ","
                    codes = [c.strip() for c in raw.split(sep) if c.strip()]
                gt[key] = codes
            return gt

        with open(self.gt_path, "rb") as f:
            return pickle.load(f)

    def get_predictions(self, response_dict: dict, target_symptoms: list[str]) -> dict[str, list[str]]:
        """``{report_id: [HPO identifiers]}`` of the target terms a verifier answered *Yes* for.

        Args:
            response_dict: ``{report: {term: {segment: {"response": text, ...}}}}``.
            target_symptoms: the terms to read.
        """
        predictions: dict[str, list[str]] = {}
        for key in response_dict:
            predictions[key] = []
            for symptom in target_symptoms:
                for target in response_dict[key][symptom]:
                    if "Yes" in response_dict[key][symptom][target]["response"]:
                        predictions[key].append(symptom)
        return predictions

    def evaluate(
        self,
        response_dict: dict,
        target_symptoms: list[str],
        output_dir: str | None = None,
        fine_tuned: bool = False,
        filter_unannotated: bool = False,
    ) -> dict:
        """Score predictions against the HCY ground truth and write the metric files.

        Args:
            response_dict: the verifier responses of a PhenoRAG-style run.
            target_symptoms: terms that are scored.
            output_dir: where the metric files are written, or None.
            fine_tuned: names the output files after a fine-tuned model.
            filter_unannotated: score only reports that have at least one annotated term.

        Returns:
            The metrics as a dict (precision, recall and F1, each between 0 and 1).
        """
        gt_dict = self.load_ground_truth()
        if filter_unannotated:
            gt_dict = {k: v for k, v in gt_dict.items() if v}
        predictions = self.get_predictions(response_dict, target_symptoms)
        target_dict = target_code_sets(target_symptoms, HPOTree())
        _, sort_gt_list, sort_response_list = project_onto_targets(gt_dict, predictions, target_dict)

        metrics, _ = evaluate_micro_macro(sort_gt_list, sort_response_list)

        print("\nEvaluation in Micro Way")
        print("Micro Precision: %.4f" % metrics["micro_precision"])
        print("Micro Recall:    %.4f" % metrics["micro_recall"])
        print("Micro F1:        %.4f" % metrics["micro_f1"])
        print("\nEvaluation in Macro Way")
        print("Macro Precision: %.4f" % metrics["macro_precision"])
        print("Macro Recall:    %.4f" % metrics["macro_recall"])
        print("Macro F1:        %.4f" % metrics["macro_f1"])

        if output_dir:
            response_info = "fineTuned_LLM" if fine_tuned else "base_LLM"
            os.makedirs(output_dir, exist_ok=True)
            pd.DataFrame([metrics]).to_csv(
                os.path.join(output_dir, response_info + ".csv"), index=False
            )

        return metrics


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="Evaluate HCY dataset")
    parser.add_argument("--responses_dir", type=str, required=True)
    parser.add_argument("--fine_tuned", type=str, default="False")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--target_symptoms_path", type=str, required=True)
    parser.add_argument("--ground_truth_path", type=str, required=True)
    args = parser.parse_args()

    fine_tuned = args.fine_tuned.lower() == "true"
    target_symptoms = pd.read_csv(args.target_symptoms_path)["target_codes"].tolist()
    file_name = "fineTuned_LLM.pkl" if fine_tuned else "base_LLM.pkl"

    with open(os.path.join(args.responses_dir, file_name), "rb") as f:
        response_dict = pickle.load(f)

    dataset = HCYDataset(args.ground_truth_path, args.target_symptoms_path)
    dataset.evaluate(response_dict, target_symptoms, args.output_dir, fine_tuned)
