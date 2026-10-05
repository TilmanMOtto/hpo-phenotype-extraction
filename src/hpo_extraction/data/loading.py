"""Data loading utilities: text files and UTI reports."""

import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

_BULLET = re.compile(r"^[*\-•·]\s*")


def load_txt(context_dir: str) -> dict[str, str]:
    """
    Load all .txt files from a directory.

    Keys are filenames (no extension) with _ character characters replaced by colons,
    mapping to file contents (latin1 encoding).
    """
    context_dict = {}
    files = [f for f in os.listdir(context_dir) if f.endswith(".txt")]
    for file in files:
        with open(os.path.join(context_dir, file), "r", encoding="latin1") as f:
            content = f.read()
        context_dict[file.split(".")[0].replace("_", ":")] = content
    return context_dict


def prepare_context(context_dict: dict[str, str]) -> dict[str, list[str]]:
    """
    Convert raw context file contents to lists of synthetic-sentencey sentences.

    Splits on lines and strips any leading bullet character (* - • ·),
    discarding empty lines. Supports mixed or single-symbol bullet formats.
    """
    for key in context_dict:
        sentences = []
        for line in context_dict[key].splitlines():
            stripped = _BULLET.sub("", line.strip())
            if stripped:
                sentences.append(stripped)
        context_dict[key] = sentences
    return context_dict


class LazyContextDict:
    """
    Lazy, disk-cached replacement for the load_txt → prepare_context → encode_dict
    pipeline.

    Construction only scans the directory (cheap os.listdir). Each HPO's embedding
    is loaded on first access, with a .npy cache written to context_dir/.cache/.
    Cache invalidation: re-encodes if the .npy is older than the source .txt.
    """

    def __init__(self, context_dir: str, model: SentenceTransformer) -> None:
        self._context_dir = context_dir
        self._model = model
        self._cache_dir = os.path.join(context_dir, ".cache")
        self._memory: dict[str, np.ndarray] = {}
        self._dim: int | None = None
        self._dim_resolved = False
        # "HP:0001234" → "HP_0001234"
        self._key_to_stem: dict[str, str] = {
            fname[:-4].replace("_", ":"): fname[:-4]
            for fname in os.listdir(context_dir)
            if fname.endswith(".txt")
        }

    def _model_dim(self) -> int | None:
        """The model's embedding dimension, or None if it can't be determined (test stubs).

        Used to invalidate a cached ``.npy`` whose width doesn't match the current model. Only a
        genuine positive integer is trusted, a mock model returns a non-int from
        ``get_sentence_embedding_dimension``, and coercing that would fabricate a wrong dimension
        and invalidate perfectly good caches.
        """
        if not self._dim_resolved:
            self._dim_resolved = True
            getter = getattr(self._model, "get_sentence_embedding_dimension", None)
            val = getter() if callable(getter) else None
            self._dim = int(val) if isinstance(val, (int, np.integer)) and val > 0 else None
        return self._dim

    def __contains__(self, key: object) -> bool:
        return key in self._key_to_stem or key in self._memory

    def keys(self):
        """All term identifiers with synthetic sentences, on disk or already encoded."""
        return set(self._key_to_stem) | set(self._memory)

    def __len__(self) -> int:
        return len(set(self._key_to_stem) | set(self._memory))

    def __iter__(self):
        return iter(self.keys())

    def is_file_backed(self, key: str) -> bool:
        """True if this key maps to a source .txt file (not an injected entry)."""
        return key in self._key_to_stem

    def set_encoded(self, key: str, embedding: np.ndarray) -> None:
        """Inject a pre-encoded embedding (e.g. for ontology-derived entries)."""
        self._memory[key] = embedding

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self._memory:
            return self._memory[key]
        if key not in self._key_to_stem:
            raise KeyError(key)

        stem = self._key_to_stem[key]
        txt_path = os.path.join(self._context_dir, stem + ".txt")
        npy_path = os.path.join(self._cache_dir, stem + ".npy")

        # Disk cache hit: .npy exists and is newer than source .txt
        if os.path.isfile(npy_path) and os.path.getmtime(npy_path) >= os.path.getmtime(txt_path):
            embedding = self._safe_load(npy_path)
            # ``_safe_load`` returns None for a corrupt/partial/pickled cache, the failure mode
            # when several processes ``np.save`` the same path concurrently (a full earlier array
            # sharing one context_dir). Fall through to re-encode in that case.
            # Cache invalidation is otherwise by mtime only, which cannot detect a .npy written by
            # a *different* embedding model. A wrong-dimension cache would blow up later when
            # stacked (UnionScorer.prepare vstack), so treat a dimension mismatch as a miss too.
            if embedding is not None:
                expected = self._model_dim()
                if expected is None or (embedding.size and embedding.shape[-1] == expected):
                    self._memory[key] = embedding
                    return embedding

        # Cache miss (or corrupt / stale / wrong-dimension cache): read, parse, encode, persist
        with open(txt_path, "r", encoding="latin1") as f:
            raw = f.read()
        sentences = []
        for line in raw.splitlines():
            stripped = _BULLET.sub("", line.strip())
            if stripped:
                sentences.append(stripped)
        embedding = self._model.encode(sentences)
        self._atomic_save(npy_path, embedding)
        self._memory[key] = embedding
        return embedding

    @staticmethod
    def _safe_load(npy_path: str):
        """``np.load`` that returns None instead of raising on a corrupt/partial/pickled file."""
        try:
            return np.load(npy_path)
        except Exception:
            return None

    def _atomic_save(self, npy_path: str, embedding: np.ndarray) -> None:
        """Write the cache atomically so concurrent readers never see a half-written file.

        A full earlier array runs 12 jobs against one shared ``context_dir``. Plain ``np.save``
        lets two processes interleave writes to the same path and produce a truncated ``.npy``
        that later fails to load. Writing to a private temp file and ``os.replace``-ing it in is
        atomic on a POSIX filesystem, so a reader sees either the old file or the complete new one.
        (Prewarming the cache once, see experiments/04_treephenorag/score_store/build_sentence_cache.py, avoids the races
        entirely. This is the belt-and-braces guard.)
        """
        import tempfile

        os.makedirs(self._cache_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._cache_dir, suffix=".npy")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.save(fh, embedding)
            os.replace(tmp, npy_path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise


def load_cohort(cfg, logger) -> tuple[dict[str, str], dict[str, list[str]]]:
    """``(reports, gold)`` for the cohort named by ``cfg.dataset``, ``"hcy"`` or ``"gsc"``.

    Ground truth is the *annotated* term set per report, with no ancestor closure: that is what the earlier comparison scores against. Every method-comparison driver
    (:mod:`hpo_extraction.baselines.rag_hpo_experiment`, :mod:`hpo_extraction.baselines.phenobert_baseline`) loads its cohort here, so
    "which reports and which ground truth" cannot drift between the methods being compared.

    Dataset modules are imported lazily: this module is imported by the retrieval pipeline, which
    has no business pulling in the evaluation package.
    """
    if cfg.dataset == "hcy":
        from hpo_extraction.evaluation.datasets.hcy import HCYDataset

        reports = load_txt(cfg.input_dir)
        gt = HCYDataset(cfg.ground_truth_path, cfg.target_symptoms_path).load_ground_truth()
    elif cfg.dataset == "gsc":
        from hpo_extraction.evaluation.datasets.gsc import GSCDataset, load_gsc_reports

        reports = load_gsc_reports(cfg.gsc_dir)
        gt = GSCDataset(cfg.gsc_dir).load_ground_truth()
    else:
        raise ValueError(f"unknown dataset {cfg.dataset!r}")
    logger.info("Dataset %s | %d reports | %d with GT", cfg.dataset, len(reports),
                sum(1 for v in gt.values() if v))
    return reports, {k: list(v) for k, v in gt.items()}


def getReports(report_dir: str = "path/to/UTI/files") -> tuple[dict, dict]:
    """
    Load translated UTI report files from patient subfolders.

    Args:
        report_dir: Directory containing one subfolder per patient,
                    each with files ending in 'translation.txt'.

    Returns:
        report_dict: {patient_folder → {filename → text}}
        summary_dict: {patient_folder → concatenated_text}
    """
    patient_folders = os.listdir(report_dir)
    report_dict: dict = {}
    summary_dict: dict = {}
    for folder in patient_folders:
        files = [
            f for f in os.listdir(os.path.join(report_dir, folder))
            if f.endswith("translation.txt")
        ]
        report_dict[folder] = {}
        patient_summary = ""
        for file in files:
            with open(os.path.join(report_dir, folder, file), "r") as content:
                file_content = content.read().replace("\n", " ")
                report_dict[folder][file] = file_content
                patient_summary += file_content + "\n"
        summary_dict[folder] = patient_summary
    return report_dict, summary_dict
