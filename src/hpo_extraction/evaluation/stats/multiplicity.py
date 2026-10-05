r"""Holm-Bonferroni adjustment, the correction for testing a whole results table at once.

A results chapter does not run one test, it runs one per (system, baseline) pair. At the 0.05 level
with ten comparisons, the chance of at least one spurious "significant" result is around 40 % if
nothing is corrected. Holm controls the family-wise error rate without assuming the tests are
independent, which counts here because every comparison shares the same ground truth and often the same
baseline, so they are strongly dependent.

Holm over Bonferroni because it is uniformly more powerful and costs nothing: same guarantee,
strictly smaller adjusted p-values.

**The family is one cohort.** A comparison on HCY and the same comparison on GSC are two questions
about two datasets, and pooling them would penalise a system for having been measured twice. This
module takes whatever dict it is handed and does not decide the family. The caller does, and should
say so in the table caption.
"""

from __future__ import annotations

from typing import Mapping


def holm(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    """``{key: {p_raw, p_holm, reject, rank}}`` over one family of hypotheses.

    Sort ascending. The *i*-th smallest of *m* is multiplied by ``m - i``. The running maximum is
    then enforced so the adjusted values are monotone in the raw ones (without it a later, larger
    raw p-value could receive a smaller adjusted one). Everything is capped at 1.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, float | bool]] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        adjusted = min(1.0, (m - i) * float(p))
        running = max(running, adjusted)
        out[key] = {
            "p_raw": float(p),
            "p_holm": running,
            "reject": bool(running <= alpha),
            "rank": i + 1,
        }
    return out
