"""
Data loading utilities for the HCY Annotation UI.

Loads segmented reports, prior_annotation ground truth annotations, previous annotation
sessions, and HPO ontology options. Provides GT→segment matching via exact and
fuzzy string matching.
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pandas as pd

# Make src/ importable when running as a script

from hpo_extraction.ontology.hpo_tree import HPOTree


def load_segmented_reports(path: str) -> dict[str, list[str]]:
    """Load segmented_reports.csv → {patient_id: [sentence0, sentence1, …]}."""
    df = pd.read_csv(path)
    result: dict[str, list[str]] = {}
    for pid, group in df.groupby("patient_id"):
        result[str(pid)] = (
            group.sort_values("sentence_idx")["sentence"].dropna().astype(str).tolist()
        )
    return result


def load_prior_annotation_gt(
    path: str,
) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """
    Load hcy_holistic_ground_truth.csv.

    Returns
    -------
    (gt_anns, report_texts)
        gt_anns      : {patient_id: [ann_dict, …]}  (empty list if no annotations)
        report_texts : {patient_id: report_text}
    """
    df = pd.read_csv(path)
    gt_anns: dict[str, list[dict]] = {}
    report_texts: dict[str, str] = {}

    for pid, group in df.groupby("patient_id"):
        pid = str(pid)
        # Report text: repeated on every row, take first non-null
        rt = group["report_text"].dropna()
        report_texts[pid] = str(rt.iloc[0]) if not rt.empty else ""

        anns: list[dict] = []
        for _, row in group.iterrows():
            hpo_code = str(row.get("hpo_code", "")).strip()
            if not hpo_code or hpo_code.lower() in ("nan", ""):
                continue
            anns.append({
                "hpo_code": hpo_code,
                "hpo_name": str(row.get("hpo_name", "") or ""),
                "trigger_word": str(row.get("trigger_word", "") or ""),
                "sentence_context": str(row.get("sentence_context", "") or ""),
                "char_offset": str(row.get("char_offset", "") or ""),
            })
        gt_anns[pid] = anns

    return gt_anns, report_texts


def match_gt_to_segments(
    gt_anns: list[dict],
    segments: list[str],
) -> tuple[list[dict], list[dict]]:
    """
    Match GT annotations to Stanza-segmented sentences.

    Matching strategy (in order):
      1. Exact string match
      2. Case-insensitive exact match
      3. Substring containment (sentence_context ⊆ segment or vice versa)
      4. Fuzzy match via difflib (cutoff=0.6)

    Returns
    -------
    (matched, unmatched)
        matched   : annotation dicts with added key ``segment_idx``
        unmatched : annotation dicts that could not be matched
    """
    matched: list[dict] = []
    unmatched: list[dict] = []

    segments_lower = [s.lower() for s in segments]

    for ann in gt_anns:
        ctx = ann.get("sentence_context", "").strip()
        if not ctx or ctx.lower() == "nan":
            unmatched.append(ann)
            continue

        # 1. Exact
        if ctx in segments:
            matched.append({**ann, "segment_idx": segments.index(ctx)})
            continue

        # 2. Case-insensitive exact
        ctx_lower = ctx.lower()
        if ctx_lower in segments_lower:
            matched.append({**ann, "segment_idx": segments_lower.index(ctx_lower)})
            continue

        # 3. Substring containment
        found_idx = -1
        for idx, seg in enumerate(segments):
            if ctx_lower in seg.lower() or seg.lower() in ctx_lower:
                found_idx = idx
                break
        if found_idx >= 0:
            matched.append({**ann, "segment_idx": found_idx})
            continue

        # 4. Fuzzy
        close = difflib.get_close_matches(ctx, segments, n=1, cutoff=0.6)
        if close:
            matched.append({**ann, "segment_idx": segments.index(close[0])})
        else:
            unmatched.append(ann)

    return matched, unmatched


def load_previous_annotations(
    hcy_dir: str,
) -> tuple[dict[str, list[dict]], list[str], str] | None:
    """
    Load the most recent annotation state from *hcy_dir*.

    Considers both the timestamped ``segment_annotations_*.csv`` exports and the
    continuously-written ``autosave_annotations.csv``, and picks whichever was
    modified last, the autosave is usually the newest state, and ignoring it
    silently discards everything since the last manual export.

    Returns
    -------
    (annotations, confirmed_patients, source_filename), or None if *hcy_dir* holds
    no previous annotation file at all.

    Raises
    ------
    Any parse error is allowed to propagate. It must never be swallowed: a failed
    resume leaves the caller rebuilding the store from ground truth alone, which
    silently reverts manual annotation work. Fail loudly instead.
    """
    p = Path(hcy_dir)
    if not p.is_dir():
        return None

    candidates = list(p.glob("segment_annotations_*.csv"))
    autosave = p / "autosave_annotations.csv"
    if autosave.is_file():
        candidates.append(autosave)
    if not candidates:
        return None

    # Newest wins. Fall back to filename order (timestamped, so chronological).
    most_recent = max(candidates, key=lambda f: (f.stat().st_mtime, f.name))

    df = pd.read_csv(most_recent)

    result: dict[str, list[dict]] = {}
    confirmed: list[str] = []
    for pid, group in df.groupby("patient_id"):
        pid = str(pid)
        anns: list[dict] = []
        for _, row in group.iterrows():
            hpo_code = str(row.get("hpo_code", "")).strip()
            if not hpo_code or hpo_code.lower() in ("nan", ""):
                continue
            anns.append({
                "segment_idx": int(row.get("segment_idx", 0)),
                "hpo_code": hpo_code,
                "hpo_name": str(row.get("hpo_name", "") or ""),
                "trigger_word": str(row.get("trigger_word", "") or ""),
            })
        if anns:
            result[pid] = anns
        # ``confirmed`` is a per-patient flag repeated on each of its rows.
        if "confirmed" in group.columns and \
                group["confirmed"].astype(str).str.lower().eq("true").any():
            confirmed.append(pid)

    if not result:
        return None
    return result, confirmed, most_recent.name


def load_hpo_options(hpo_json_path: str) -> list[dict]:
    """
    Build sorted dcc.Dropdown options from the HPO ontology.

    Returns a list of ``{"label": "<Name> (<Code>)", "value": "<Code>"}`` dicts.
    """
    tree = HPOTree(hpo_json_path=hpo_json_path)
    options: list[dict] = []
    for code in tree.hpo_list:
        try:
            name = str(tree.getNameByHPO(code)).title()
        except Exception:
            name = code
        options.append({"label": f"{name} ({code})", "value": code})
    return sorted(options, key=lambda x: x["label"].lower())
