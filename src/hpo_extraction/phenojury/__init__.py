"""PhenoJury: a jury of small language models names findings, a normaliser codes them, a vote keeps
the terms enough jurors agree on.

The application interface is :class:`PhenoJury` (``pipeline.py``) and the command line
``hpo-phenojury`` (``cli.py``). The other modules hold the building blocks the thesis experiments
use: prompts, generation, normalisers, PhenoBERT, and the vote.
"""


def __getattr__(name):
    # Imported on first use, so that importing a building block does not load the models' stack.
    if name in ("PhenoJury", "PhenoJurySettings", "dictionary_normaliser", "phenobert_normaliser"):
        from hpo_extraction.phenojury import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
