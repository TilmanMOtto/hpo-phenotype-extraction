"""Deep-dive analysis UI for ``phenojury_generation_free_listing``, the 8-model SLM ensemble.

The best-scoring method in the whole earlier comparison, and the least legible one: eight small
LLMs read every sentence of a report, PhenoBERT grounds their free text to HPO ids, and a k-of-N
vote turns that into a term set. The shipped artifacts say what came out (two CSVs and nine rule
directories). They say nothing about *what each model wrote*, *which models carried a given
prediction*, or *where a missed annotated term died* in the generate → ground → vote chain. This app
answers those three questions.

Not to be confused with:

``app/ensemble_ui``
    the earlier ensemble UI. Same 8 models, incompatible contract: earlier projects onto the
    691-symptom target matrix with children-aware matching, aggregates union/majority/plurality,
    is HCY-only, and puts each run in its own top-level directory.
``app/exp13_ui``
    the earlier runs *tree-traversal* deep dive (an earlier exploratory run/01/02), pruning decisions, τ sweeps, node
    scores. Nothing in common with this package but the experiment group number.

Run it with ``python apps/phenojury_ui/app.py``. See ``README.md``.
"""
