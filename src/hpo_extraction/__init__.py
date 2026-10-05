"""HPO phenotype extraction from clinical text: TreePhenoRAG and PhenoJury."""

__version__ = "1.0.0"

# Registers the ${paths:...} resolver the experiment configs use (see hpo_extraction/paths.py).
from hpo_extraction import paths as _paths  # noqa: E402,F401
