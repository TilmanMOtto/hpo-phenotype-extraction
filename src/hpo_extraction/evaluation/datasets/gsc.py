"""Evaluation dataset loader for the GSC+ corpus (``resources/data/GSC_2024``).

GSC+ is the benchmark RAG-HPO and AutoPCR report on. The corpus is two parallel directories:

    GSC_2024/Text/<doc_id>          raw report text, one file per document (no extension)
    GSC_2024/Annotations/<doc_id>   one annotation per line: ``start:end \t HP:####### \t mention``

Unlike HCY the annotations are *mention-level* and the HPO codes are the **annotated** terms only
(no ancestor closure), which is the ground truth earlier evaluates against. This module
provides the input loader and the ground-truth loader. The flat/hierarchy/calibration metrics
themselves are computed in Step 2, so :meth:`GSCDataset.evaluate` is intentionally a thin
document-level micro/macro P/R/F1 for parity with :class:`HCYDataset` and nothing more.

``load_txt`` cannot read this corpus: the files have no ``.txt`` extension and the directory holds
Windows ``*:Zone.Identifier`` sidecar junk that must be skipped.

The tail of this module handles the *other* GSC ground truth standard: RAG-HPO evaluated on 114 of these
documents against its own re-annotation, vendored in ``resources/data/GSC_RAGHPO``. See
:func:`load_raghpo_ground_truth` and that directory's ``PROVENANCE.md``.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

from hpo_extraction.evaluation.datasets.base import EvalDataset
from hpo_extraction.evaluation.set_metrics import evaluate_micro_macro

_HPO_RE = re.compile(r"HP:\d{7}")


def _gsc_doc_ids(directory: Path) -> list[str]:
    """Document ids present in *both* Text/ and Annotations/, minus WSL/macOS junk.

    Same filter as ``an earlier exploratory run/gt_stats._gsc_ids``: drop AppleDouble ``._``
    sidecars and any ``*:Zone.Identifier`` files. The real corpus is 228 documents. Without this
    filter each doc is double-counted by its junk twin.

    The sidecar test matches the **suffix**, not the colon. Read over a Windows share (a
    ``wsl.localhost`` UNC path) the separator comes back as ``U+F03A``, the Private Use Area
    stand-in for a character NTFS will not surface, so a ``":" not in name`` test lets all 228
    sidecars through and the corpus silently reads as 456 documents. That is invisible on the
    cluster and wrong everywhere else.
    """
    def real(d: Path) -> set[str]:
        return {
            p.name for p in d.iterdir()
            if p.is_file() and not p.name.startswith("._")
            and not p.name.endswith("Zone.Identifier")
        }

    return sorted(real(directory / "Text") & real(directory / "Annotations"))


def load_gsc_reports(directory: str | Path) -> dict[str, str]:
    """``{doc_id: report_text}`` for every GSC+ document."""
    directory = Path(directory)
    return {
        doc_id: (directory / "Text" / doc_id).read_text(encoding="utf-8", errors="replace")
        for doc_id in _gsc_doc_ids(directory)
    }


def load_gsc_ground_truth(directory: str | Path) -> dict[str, list[str]]:
    """``{doc_id: [HP:#######, ...]}``, the annotated terms per document, deduplicated.

    Annotation lines whose second field is not a well-formed HPO code are skipped (they surface
    as unresolvable in the an earlier exploratory run hygiene table). Terms are returned as the annotated codes with
    **no ancestor closure**, that is the ground-truth set earlier scores against.
    """
    directory = Path(directory)
    gt: dict[str, list[str]] = {}
    for doc_id in _gsc_doc_ids(directory):
        seen: dict[str, None] = {}  # dict preserves first-seen order
        for line in (directory / "Annotations" / doc_id).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            code = parts[1].strip() if len(parts) > 1 else ""
            if _HPO_RE.fullmatch(code):
                seen.setdefault(code, None)
        gt[doc_id] = list(seen)
    return gt


# ── The RAG-HPO subset (resources/data/GSC_RAGHPO) ───────────────────────────
#
# The RAG-HPO paper evaluates on 114 of these 228 documents, against its own manual re-annotation.
# Both are vendored. See that directory's PROVENANCE.md for the derivation and the reproduction of
# The paper's Table 1. These loaders exist so a subset can be scored from full-corpus artifacts.

def load_gsc2024_eval_ids(gsc_dir: str | Path) -> list[str]:
    """The 206 GSC-2024 document ids Tao et al. evaluate AutoPCR on, from ``eval_206_ids.txt``.

    The corpus is split 22 development / 206 evaluation following PhenoTagger's split, and the
    published AutoPCR figures are over the 206 only. Regenerate the file with
    ``experiments/03_setup/extract_gsc_eval_split.py``. The ground truth stays the corpus's own annotation in
    ``Annotations/``, so this narrows the document set and nothing else.
    """
    path = Path(gsc_dir) / "eval_206_ids.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist -- run scripts/extract_gsc2024_eval_split.py to vendor "
            f"AutoPCR's evaluation split")
    return [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_raghpo_ids(subset_dir: str | Path) -> list[str]:
    """The 114 GSC+ document ids RAG-HPO evaluated on, from ``document_ids.txt``."""
    path = Path(subset_dir) / "document_ids.txt"
    return [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_raghpo_ground_truth(subset_dir: str | Path) -> dict[str, list[str]]:
    """``{doc_id: [HP:#######, ...]}``, RAG-HPO's *own* annotation of their 114 documents.

    Same shape as :func:`load_gsc_ground_truth`, and likewise with no ancestor closure, so the two
    are interchangeable as a ground truth source. This one is a different ground truth standard, not a subset of
    the other: about 11% of its terms are not annotated in the corpus at all.

    Every id in ``document_ids.txt`` gets a key even if the annotation file has no rows for it, so
    a document can never vanish from the cohort by being unannotated.
    """
    subset_dir = Path(subset_dir)
    gt: dict[str, dict[str, None]] = {doc_id: {} for doc_id in load_raghpo_ids(subset_dir)}
    with open(subset_dir / "annotations.csv", "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            code = (row.get("hpo_id") or "").strip()
            if _HPO_RE.fullmatch(code):
                gt.setdefault((row.get("doc_id") or "").strip(), {}).setdefault(code, None)
    return {doc_id: list(codes) for doc_id, codes in gt.items()}


def restrict_to_ids(ground_truth: dict, doc_ids: Iterable[str]) -> dict:
    """*ground_truth* narrowed to *doc_ids*, raising if one of them is absent.

    A missing id must not quietly shrink the cohort: the result would still look like a complete
    replication while describing fewer documents than it claims.
    """
    doc_ids = list(doc_ids)
    missing = [d for d in doc_ids if d not in ground_truth]
    if missing:
        raise KeyError(
            f"{len(missing)} of {len(doc_ids)} requested document(s) are absent from the "
            f"ground truth (e.g. {missing[:5]})"
        )
    return {doc_id: ground_truth[doc_id] for doc_id in doc_ids}


class GSCDataset(EvalDataset):
    """GSC+ ground-truth loader with a document-level micro/macro evaluate for parity.

    Instantiate with the corpus directory. ``load_ground_truth`` returns ``{doc_id: [hpo]}``
    (annotated, no closure). ``get_predictions`` / ``evaluate`` mirror the flat treatment the HCY
    loader gives so a run can log a main F1. All hierarchy-aware and calibration metrics live
    in Step 2 and consume the per-run JSONL artifacts, not this method.
    """

    def __init__(self, gsc_dir: str | Path):
        self.gsc_dir = Path(gsc_dir)

    def load_ground_truth(self) -> dict[str, list[str]]:
        """``{document_id: [HPO identifiers]}`` of the GSC+ corpus annotation."""
        return load_gsc_ground_truth(self.gsc_dir)

    def get_predictions(self, response_dict: dict, target_symptoms: list[str]) -> dict:
        """``{doc_id: set(hpo)}`` from a ``{doc_id: {hpo: [{response: Yes/No}]}}`` response dict."""
        preds: dict[str, set] = {}
        for doc_id, hpo_map in response_dict.items():
            positive = {
                hpo for hpo, resp in hpo_map.items()
                if resp and "Yes" in str(resp[0].get("response", ""))
            }
            preds[doc_id] = positive & set(target_symptoms) if target_symptoms else positive
        return preds

    def evaluate(
        self, response_dict: dict, target_symptoms: list[str], output_dir: str | None = None
    ) -> dict:
        """Score predictions against the GSC+ ground truth.

        Args:
            response_dict: the verifier responses of a PhenoRAG-style run.
            target_symptoms: terms that are scored. Empty means every term.
            output_dir: where the metric files are written, or None.

        Returns:
            The metrics as a dict (precision, recall and F1, each between 0 and 1).
        """
        gt = self.load_ground_truth()
        preds = self.get_predictions(response_dict, target_symptoms)
        label_space = set(target_symptoms) if target_symptoms else None
        gold_sets, pred_sets = [], []
        for doc_id in sorted(set(gt) | set(preds)):
            gold = set(gt.get(doc_id, []))
            pred = set(preds.get(doc_id, []))
            if label_space is not None:  # score only within the evaluated label space
                gold, pred = gold & label_space, pred & label_space
            gold_sets.append(gold)
            pred_sets.append(pred)
        metrics, _ = evaluate_micro_macro(gold_sets, pred_sets)
        return metrics
