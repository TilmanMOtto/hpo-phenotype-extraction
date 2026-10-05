"""Locations inside this repository.

Every module that needs a file shipped with the repository (the ontology, the vendored baselines,
the synthetic sentences) takes its location from here, so moving a folder means editing one line.
Paths outside the repository (data, models, results) are never defined in code: they come from
``configs/cluster_leomed.yaml`` or ``configs/local_example.yaml``.
"""
from __future__ import annotations

from pathlib import Path

#: Root of the repository checkout (the folder that holds ``src/``, ``configs/`` and ``resources/``).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Static inputs shipped with the repository.
RESOURCES_DIR = REPO_ROOT / "resources"

#: Vendored third-party code (AutoPCR, RAG-HPO) and the place where PhenoBERT is installed.
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
AUTOPCR_DIR = THIRD_PARTY_DIR / "AutoPCR"
RAGHPO_DIR = THIRD_PARTY_DIR / "RAG-HPO"
PHENOBERT_DIR = THIRD_PARTY_DIR / "PhenoBERT"

#: Configuration files.
CONFIGS_DIR = REPO_ROOT / "configs"


# ── Locations outside the repository ─────────────────────────────────────────
#
# Data, models and results are named in one YAML file: configs/cluster_leomed.yaml on the cluster,
# or a copy of configs/local_example.yaml elsewhere, selected with the environment variable
# HPO_PATHS. Experiment configs refer to its keys as ${paths:key}, for example
# ${paths:hcy.curated_ground_truth}.

#: Environment variable that names the path file.
PATHS_ENV = "HPO_PATHS"
DEFAULT_PATHS_FILE = CONFIGS_DIR / "cluster_leomed.yaml"

_loaded: dict = {}


def path_config(path: str | Path | None = None):
    """The path file as an OmegaConf object (``HPO_PATHS``, else ``configs/cluster_leomed.yaml``)."""
    import os

    from omegaconf import OmegaConf

    path = Path(path or os.environ.get(PATHS_ENV) or DEFAULT_PATHS_FILE)
    key = str(path.resolve())
    if key not in _loaded:
        _loaded[key] = OmegaConf.load(path)
    return _loaded[key]


def lookup(key: str) -> str:
    """The value of one dotted key of the path file, for example ``"hcy.input_dir"``."""
    from omegaconf import OmegaConf

    value = OmegaConf.select(path_config(), key, throw_on_missing=True)
    if value is None:
        raise KeyError(f"{key!r} is not set in the path file ({PATHS_ENV} or "
                       f"{DEFAULT_PATHS_FILE})")
    return value


def results_dir() -> Path:
    """Folder holding the result directories: ``results_dir`` of the path file, if one is selected
    with ``HPO_PATHS``, otherwise ``output/`` inside the repository."""
    import os

    if os.environ.get(PATHS_ENV):
        return Path(lookup("results_dir"))
    return REPO_ROOT / "output"


def thesis_dir() -> Path:
    """The thesis LaTeX source the figure scripts write into: ``thesis_dir`` of the path file if
    ``HPO_PATHS`` is set, otherwise ``thesis/`` inside the repository (ignored by git)."""
    import os

    if os.environ.get(PATHS_ENV):
        return Path(lookup("thesis_dir"))
    return REPO_ROOT / "thesis"


def _register_resolver() -> None:
    try:
        from omegaconf import OmegaConf
    except ImportError:
        # AutoPCR's parser environment imports the package for its NLTK data and has no OmegaConf.
        return

    if not OmegaConf.has_resolver("paths"):
        OmegaConf.register_new_resolver("paths", lookup)


_register_resolver()

