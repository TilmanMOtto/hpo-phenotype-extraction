"""Deep-dive analysis UI for the earlier tree-traversal experiments.

``app/tree_ui`` reads the earlier runs/07/10 artifact contract. earlier shares none of it, nested
``<exp>/<cohort>/tau_*/`` directories, shard suffixes, ``report_id`` instead of ``patient_id``,
two independently-thresholded node scores, open-set predictions with a ``{"summary": true}``
terminator line, and no persisted sentence text. This package reads that contract instead, and
asks the question the aggregate tables of ``result_tables`` cannot: *why* recall is
where it is, which pruning decision severed which annotated term, at what depth, at what score.

Run it with ``python apps/treephenorag_ui/app.py``. See ``README.md``.
"""
