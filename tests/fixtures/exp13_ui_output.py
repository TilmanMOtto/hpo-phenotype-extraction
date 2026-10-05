"""Fixture extensions the deep-dive UI needs and ``build_exp13_tree`` does not write.

``tests/fixtures/exp13_output.build_exp13_tree`` writes as much of an earlier run as
``result_tables`` reads. The UI reads two more things and cares about a third situation the metrics
harness refuses outright:

* ``*_node_metadata.jsonl``, the labels, subtree sizes and organ systems the tables display.
* the cohort's **report texts**, which the deep-dive puts on screen and which the sentence
  recovery in ``apps/treephenorag_ui/evidence.py`` verifies its segmentation against.
* the Free Listing generation run's ``llm_extractions_{model}.jsonl``, the one earlier artifact that persists sentence
  *text* against the sentence numbers the tree runs index by, the route that lets the deep-dive
  show sentences on a machine where Stanza will not load.
* shard-suffixed artifacts, both complete and incomplete, since the UI reads them (flagged)
  rather than refusing them.
* a truncated final line, which is the normal shape of a wall-clock kill and must raise, not be skipped.

Everything here writes the driver's exact schema. The file:line of the dict literal each one
mirrors is named in its docstring.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from fixtures.toy_ontology import A, B, C, D, E, F, G, H, M, NAMES, ToyTree

#: Depths in the toy graph, matching ``fixtures.exp13_output._depth``.
_DEPTHS = {A: 1, G: 1, B: 2, E: 2, H: 2, C: 3, D: 3, F: 3, M: 3}


def write_node_metadata(path: Path, hpo_ids=None) -> None:
    """``core/node_metadata.py:48-59``, the static per-node sidecar."""
    tree = ToyTree()
    hpo_ids = hpo_ids or [A, B, C, D, E, F, G, H, M]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for hpo_id in hpo_ids:
            entry = tree.data[hpo_id]
            children = sorted(entry["Son"].keys())
            f.write(json.dumps({
                "hpo_id": hpo_id,
                "hpo_label": NAMES.get(hpo_id, hpo_id),
                "subtree_size": len(entry["Child"]) + 1,
                "n_direct_children": len(children),
                "is_leaf": not children,
                "depth_bfs": _DEPTHS.get(hpo_id),
                "depth_long": _DEPTHS.get(hpo_id),
                "layer1_organ": NAMES.get(A if hpo_id in (A, B, E, C, D, F) else G),
                "in_phenotypic_abnormality": True,
            }) + "\n")


def write_report_texts(directory, texts: dict[str, str]) -> Path:
    """One ``<report_id>.txt`` per report, the shape ``loaders.load_report_texts`` keys by stem."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for report_id, text in texts.items():
        (directory / f"{report_id}.txt").write_text(text, encoding="utf-8")
    return directory


def write_segments_csv(path, sentences: dict[str, list[str]]) -> Path:
    """``segmented_reports.csv``, ``experiments/03_setup/segment_reports.py:68-73``, column for column.

    The persisted Stanza segmentation. This is the file the deep-dive's preferred recovery route
    reads, and the coordinate system the curated ground truth's ``segment_idx`` indexes, so a fixture that
    wrote it in any other shape would be testing a file that does not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "sentence_idx", "sentence"])
        for report_id in sorted(sentences):
            for index, sentence in enumerate(sentences[report_id]):
                writer.writerow([report_id, index, sentence])
    return path


def write_extractions(run_dir, sentences: dict[str, list[str]], model: str = "toy",
                      *, drop: tuple[str, int] | None = None) -> Path:
    """The Free Listing generation run's ``llm_extractions_{model}.jsonl``, ``slm_ensemble_experiment.py:265-271``.

    Only the four fields the segmentation recovery reads are meaningful here; ``llm_output`` is
    filled so the record shape is the driver's, not a subset of it. *drop* omits one
    ``(report_id, sentence_number)``, which is how the "this file has a hole in its numbering"
    case is produced, the one that must be rejected, not silently renumbered.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"llm_extractions_{model}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for report_id, sents in sorted(sentences.items()):
            for number, text in enumerate(sents):
                if drop == (report_id, number):
                    continue
                f.write(json.dumps({
                    "model": model, "patient_id": report_id, "sentence_number": number,
                    "sentence_text": text, "llm_output": "no phenotype",
                }) + "\n")
    return path


def shard(path: Path, n_shards: int = 3, *, drop: int | None = None) -> list[Path]:
    """Split one merged artifact into ``{stem}_s{i}of{N}_{kind}.jsonl`` files, round-robin.

    Mirrors what the SLURM array produces before ``merge_shards.py`` runs: the merged file is
    *removed*, because the real thing does not exist until the merge. ``drop`` omits one shard
    index, which is how the "the array is still running" state is reproduced.
    """
    name = path.name
    stem, _, kind = name.rpartition("_")
    kind = kind.removesuffix(".jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    written: list[Path] = []
    for index in range(n_shards):
        if index == drop:
            continue
        target = path.with_name(f"{stem}_s{index}of{n_shards}_{kind}.jsonl")
        target.write_text("".join(f"{line}\n" for line in lines[index::n_shards]),
                          encoding="utf-8")
        written.append(target)
    path.unlink()
    return written


def truncate_last_line(path: Path) -> None:
    """Chop the final line in half, the shape a wall-clock kill leaves behind."""
    text = path.read_text(encoding="utf-8")
    body, _, last = text.rstrip("\n").rpartition("\n")
    path.write_text(f"{body}\n{last[:len(last) // 2]}", encoding="utf-8")


def write_gold_csv(path, gold: dict[str, list[str]]) -> Path:
    """The two-column CSV ``HCYDataset.load_ground_truth`` reads.

    Accepts a ``str`` as well as a ``Path``, tests rewrite the file through
    ``registry.hcy_gt_path``, which is a plain string.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["patient_id,hpo_codes"]
    rows += [f"{rid},{';'.join(sorted(codes))}" for rid, codes in sorted(gold.items())]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


#: ``hcy_ground_truth/dataset.py``'s ``ANNOTATION_FIELDS``, minus the four columns this
#: app never reads (``edited``, ``original_hpo_code``, ``key``, and the per-qualifier ``q_*``
#: booleans, which are a groupby convenience for whoever wants to rebuild the ground truth). A fixture
#: writing every column would fix this app to the exact shape of a file it reads
#: leniently, ``curated._annotation`` takes what it recognises and ignores the rest.
CURATED_ANNOTATION_FIELDS = [
    "patient_id", "hpo_code", "hpo_name", "in_gold", "exclude_reason", "source", "status",
    "segment_idx", "trigger_word", "segment_text", "anchored", "anchor_how", "qualifiers", "note",
]

CURATED_REPORT_FIELDS = [
    "patient_id", "in_cohort", "cohort_reason", "n_segments", "n_annotations", "n_gold_terms",
    "n_dropped", "n_unanchored", "n_prior_annotation", "n_daphne", "n_suggestions", "n_adjudicated",
    "is_confirmed", "difficulty", "report_labels", "n_comments",
]


def write_curated_dataset(root, date: str, annotations: list[dict],
                          reports: list[dict] | None = None,
                          manifest: dict | None = None,
                          gold: dict[str, list[str]] | None = None) -> Path:
    """A ``curated_ground_truth_<date>/`` directory as ``hcy_ground_truth`` writes one.

    *annotations* is the whole input: the two-column ground truth is **derived** from its ``in_gold`` rows
    unless *ground truth* overrides it, and the manifest counts are derived from the result. That is the
    consistent case, and it has to be the default, the checks in
    ``app/exp13_ui/selfcheck._check_curated`` exist to catch the inconsistent one, so a
    test that wants it has to ask for it by passing a mismatched *ground truth* or *manifest*.
    """
    import csv as _csv

    root = Path(root) / f"curated_ground_truth_{date}"
    root.mkdir(parents=True, exist_ok=True)

    rows = [{field: row.get(field, "") for field in CURATED_ANNOTATION_FIELDS}
            for row in annotations]
    with open(root / "hcy_curated_annotations.csv", "w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=CURATED_ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if gold is None:
        derived: dict[str, set] = {}
        for row in rows:
            if str(row.get("in_gold")) in ("1", "True", "true"):
                derived.setdefault(str(row["patient_id"]), set()).add(str(row["hpo_code"]))
        gold = {pid: sorted(codes) for pid, codes in derived.items()}
    write_gold_csv(root / "hcy_ground_truth_curated.csv", gold)

    report_rows = reports if reports is not None else [
        {"patient_id": pid, "in_cohort": 1, "cohort_reason": "prior_annotation",
         "n_gold_terms": len(codes)} for pid, codes in sorted(gold.items())]
    with open(root / "hcy_curated_reports.csv", "w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=CURATED_REPORT_FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in CURATED_REPORT_FIELDS}
                          for row in report_rows])

    payload = {
        "dataset": root.name,
        "date": date,
        "generated_at": f"{date}T12:00:00+02:00",
        "built_by": "experiments/03_setup/ground_truth",
        "inputs": {"curation_log": "events.jsonl", "curation_log_sha256_16": "0123456789abcdef"},
        "policy": {"criteria": ["prior_annotation", "daphne", "suggestion"], "require_evidence": True,
                   "exclude_labels": ["unsure", "family"]},
        "counts": {"n_reports_in_cohort": len(gold),
                   "n_gold_pairs": sum(len(v) for v in gold.values())},
    }
    if manifest:
        payload.update(manifest)
    (root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return root
