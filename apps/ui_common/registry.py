"""Run discovery, the shared HPOTree, and the two-level cache.

The cache is what makes the app usable over an SSH tunnel:

* **Level 1, the RunReport, on disk.** A small JSON blob keyed by ``(run_id, artifact mtime,
  schema version)``. Restarting the app costs nothing. A rerun of the experiment invalidates
  itself automatically.
* **Level 2, the bundle, in process.** The node table, the confidence table, the segment index:
  everything a drill-down needs and nothing the browser may see. An LRU of a handful of runs,
  because loading one costs a few seconds and comparing two must not cost ten.

The ``HPOTree`` is built once. Its constructor parses a 15.5 MB JSON and normalises
every phrase in the ontology. Doing that per callback would make the app unusable, and doing it
per prompt (as ``hpo_extraction.treephenorag.verifier_prompt.embed_symptom_in_prompt_v1`` does) would make it hopeless.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from hpo_extraction.evaluation.datasets.hcy import HCYDataset
from hpo_extraction.ontology.hpo_tree import HPOTree

from . import calib, loaders, nodes as nodes_mod, report as report_mod

logger = logging.getLogger(__name__)

# Bump when the RunReport schema changes, so stale disk-cached reports are ignored.
# v2: graded scoring gained micro_{exact,soft,closure}_{precision,recall,f1}.
SCHEMA_VERSION = 2

_BUNDLE_LRU_SIZE = 6

# The ontology is independent of which run or ground truth is being looked at, and parsing it costs
# seconds. It is held here rather than on the Registry so that repointing the UI at a different
# output folder (which rebuilds the Registry) does not re-parse 15 MB of JSON.
_TREE: HPOTree | None = None


def get_tree() -> HPOTree:
    """The shared ``HPOTree``, loaded once per process."""
    global _TREE
    if _TREE is None:
        logger.info("Building HPOTree (parsing hpo.json) …")
        tree = HPOTree()
        tree.buildHPOTree()  # populates depth_dict + depth_long, which every tree metric needs
        _TREE = tree
        logger.info("HPOTree ready: %d nodes, max depth %d", len(tree.data), getattr(tree, "depth", -1))
    return _TREE


def _jsonable(obj: Any) -> Any:
    """Coerce numpy/pandas scalars so the report survives a JSON round-trip."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if obj is pd.NaT or (isinstance(obj, float) and pd.isna(obj)):
        return None
    return obj


class Registry:
    """Everything the views ask questions of. One instance, created at app startup."""

    def __init__(
        self,
        output_base: str,
        ground_truth_path: str,
        target_symptoms_path: str,
        input_dir: str | None = None,
        cache_dir: str | None = None,
    ):
        self.output_base = output_base
        self.ground_truth_path = ground_truth_path
        self.target_symptoms_path = target_symptoms_path
        self.input_dir = input_dir
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/phenorag_tree_ui")
        os.makedirs(self.cache_dir, exist_ok=True)

        self._gt: dict[str, list[str]] | None = None
        self._targets: list[str] | None = None
        self._texts: dict[str, str] | None = None
        self._bundles: OrderedDict[str, dict] = OrderedDict()

    # ── shared, expensive singletons ─────────────────────────────────────────

    @property
    def tree(self) -> HPOTree:
        """The shared ``HPOTree``."""
        return get_tree()

    @property
    def gt_dict(self) -> dict[str, list[str]]:
        """``{report: list of HPO identifiers}`` of the loaded ground truth."""
        if self._gt is None:
            self._gt = HCYDataset(self.ground_truth_path, self.target_symptoms_path).load_ground_truth()
        return self._gt

    @property
    def target_symptoms(self) -> list[str]:
        """The HPO identifiers the cohort was annotated for."""
        if self._targets is None:
            self._targets = pd.read_csv(self.target_symptoms_path)["target_codes"].tolist()
        return self._targets

    @property
    def patient_texts(self) -> dict[str, str]:
        """``{report: text}`` of the cohort."""
        if self._texts is None:
            self._texts = loaders.load_patient_texts(self.input_dir)
        return self._texts

    def dataset(self) -> HCYDataset:
        """The ``HCYDataset`` behind the registry."""
        return HCYDataset(self.ground_truth_path, self.target_symptoms_path)

    def validate(self) -> list[str]:
        """Human-readable problems with the configured paths, or [] if all is well.

        Called when the user points the UI at a new set of paths, so a typo comes back as a
        sentence in the sidebar, not a stack trace in a callback.
        """
        problems: list[str] = []
        if not os.path.isdir(self.output_base):
            problems.append(f"Output folder not found: {self.output_base}")
        elif not self.list_runs():
            problems.append(f"No runs found under {self.output_base}")

        if not os.path.isfile(self.ground_truth_path):
            problems.append(f"Ground truth not found: {self.ground_truth_path}")
        else:
            try:
                if not self.gt_dict:
                    problems.append("Ground truth loaded but is empty.")
            except Exception as exc:  # a malformed CSV/pickle must not take the app down
                problems.append(f"Could not read ground truth: {exc}")

        if not os.path.isfile(self.target_symptoms_path):
            problems.append(f"Target symptoms CSV not found: {self.target_symptoms_path}")
        else:
            try:
                self.target_symptoms
            except Exception as exc:
                problems.append(f"Could not read target symptoms: {exc}")

        if self.input_dir and not os.path.isdir(self.input_dir):
            problems.append(
                f"Report folder not found: {self.input_dir}, the patient view will show no text.")
        return problems

    # ── runs ─────────────────────────────────────────────────────────────────

    def list_runs(self) -> list[dict]:
        """The discovered runs, one dict per run."""
        return loaders.find_runs(self.output_base)

    def _run_ref(self, run_id: str) -> dict | None:
        run_dir = os.path.join(self.output_base, run_id)
        if not os.path.isdir(run_dir):
            return None
        variant = loaders.detect_variant(run_dir)
        if variant is None:
            return None
        return {"run_id": run_id, "run_dir": run_dir, "variant": variant}

    def _cache_path(self, run_id: str, mtime: float) -> str:
        key = hashlib.sha1(
            f"{run_id}|{mtime}|{SCHEMA_VERSION}|{self.ground_truth_path}".encode()
        ).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{run_id}.{key}.json")

    def get_report(self, run_id: str) -> dict | None:
        """The compact RunReport. Disk-cached. The only thing that goes to the browser."""
        ref = self._run_ref(run_id)
        if ref is None:
            return None

        cache_path = self._cache_path(run_id, loaders.artifacts_mtime(ref["run_dir"]))
        if os.path.isfile(cache_path):
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                logger.warning("Discarding unreadable cache entry %s", cache_path)

        bundle = self.get_bundle(run_id)
        if bundle is None:
            return None
        report = bundle["report"]
        try:
            with open(cache_path, "w") as f:
                json.dump(_jsonable(report), f)
        except OSError:
            logger.warning("Could not write report cache to %s", cache_path)
        return report

    def get_bundle(self, run_id: str) -> dict | None:
        """Raw artifacts + node table + traversal graph + confidence table. Server-side only."""
        if run_id in self._bundles:
            self._bundles.move_to_end(run_id)
            return self._bundles[run_id]

        ref = self._run_ref(run_id)
        if ref is None:
            return None

        logger.info("Loading run %s …", run_id)
        raw = loaders.load_raw(ref["run_dir"], ref["variant"])
        nodes = nodes_mod.build_node_table(raw, self.gt_dict, self.tree)
        graph = calib.build_graph(nodes, self.tree) if not nodes.empty else {}
        node_conf = calib.node_conf_table(raw, nodes) if not nodes.empty else {}
        report = report_mod.build_report(
            run_id, raw, nodes, self.gt_dict, self.target_symptoms, self.tree,
        )

        bundle = {
            "run_id": run_id,
            "raw": raw,
            "nodes": nodes,
            "graph": graph,
            "node_conf": node_conf,
            "report": report,
            "mtime": loaders.artifacts_mtime(ref["run_dir"]),
        }
        self._bundles[run_id] = bundle
        self._bundles.move_to_end(run_id)
        while len(self._bundles) > _BUNDLE_LRU_SIZE:
            evicted, _ = self._bundles.popitem(last=False)
            logger.debug("Evicted bundle %s from the LRU", evicted)
        return bundle
