"""TreePhenoRAG: term-by-term verification while walking the HPO from the top.

The application interface is :class:`TreePhenoRAG` (``pipeline.py``) and the command line
``hpo-treephenorag`` (``cli.py``). The other modules hold the building blocks the thesis
experiments use: the traversal, the pooling operators, the score store and its offline evaluation,
and the selection of thresholds by nested cross-validation.
"""


def __getattr__(name):
    # Imported on first use, so that importing a building block does not load the models' stack.
    if name in ("TreePhenoRAG", "TreePhenoRAGSettings", "POOLINGS"):
        from hpo_extraction.treephenorag import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
