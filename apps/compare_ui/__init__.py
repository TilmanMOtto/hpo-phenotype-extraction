"""Five main methods against one report, side by side, with each one's own reasoning.

``comparison`` puts PhenoBERT, AutoPCR, RAG-HPO, PhenoJury and TreePhenoRAG in one
table, and its finest grain is a row per (method x cohort x qualifier). That is the right shape for
a results chapter and the wrong shape for the question this app exists to answer: standing on one
report, *why* did PhenoJury find ``HP:0001250`` and TreePhenoRAG not?

The evidence is all on disk already, in five different shapes and two different orders of
magnitude -- PhenoBERT's detections are 240 KB, TreePhenoRAG's verifier calls are 4.3 GB. So this
package is two halves that never run in the same process:

**The builder** (:mod:`build`, :mod:`methods`) runs where the artifacts are, reads each method's
own records for the twenty reports of the deep-dive frame, and distils them into one compact JSON
per report. It is where the ontology gets loaded, where the tree cache gets re-run, and where a
report's 39 MB of verifier calls become the few dozen that bear on a term somebody will click.

**The app** (:mod:`app`, :mod:`views`) reads those bundles and nothing else. It holds no ontology,
opens no experiment directory and never touches a file larger than a few hundred kilobytes, which
is what makes it usable through an ``ssh -L`` tunnel.

What this is NOT
----------------
It is **not** a metrics surface. The twenty reports are a purposive sample drawn by
``apps/compare_ui/select_hcy_documents.py`` -- the cells are defined on the very outcome a reader would want
to measure -- so no rate computed over them means anything, and the scorecard says so on every
panel. The quotable numbers come from ``comparison``'s own tables and are shown unchanged.

It is **not** a curation tool. Nothing here writes. The ground truth is read from the curated dataset and
drawn. Disagreeing with it is a job for ``app/hcy_curation_ui``.

It is **not** a replacement for ``app/exp13_ui``, which goes deeper on TreePhenoRAG alone. This one
goes wider: one column per method, one question -- who got this term, and how.

Running it
----------
On the cluster (HCY is patient data and does not leave it)::

    sbatch slurm/compare_ui_bundles.sbatch           # 10 to 20 min, CPU only
    python apps/compare_ui/app.py --port 8057
    # locally:  ssh -L 8057:localhost:8057 <cluster>  ->  http://localhost:8057

Self-verifying, without any data::

    python apps/compare_ui/build.py --selftest
    python apps/compare_ui/app.py --selftest
"""

__all__ = ["roster", "bundles", "build", "methods"]
