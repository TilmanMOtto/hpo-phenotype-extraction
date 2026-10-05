"""Run PhenoBERT's ``annotate.py`` with a recursion limit that survives a long line.

Executed by the *PhenoBERT* interpreter (stanza 1.4.1), not ours, so it may use the standard
library only, and must not import anything from ``src/``.

Why this exists
---------------
stanza's dependency parser resolves the maximum spanning tree with Chu-Liu/Edmonds, whose cycle
detection (``stanza/models/common/chuliu_edmonds.py:strong_connect``) is a *recursive* Tarjan SCC:
one Python frame per token in the cycle. On a normal sentence that is a few dozen frames. On a
single unbroken line of a few thousand tokens it passes CPython's default 1000-frame limit and the
whole annotation dies with ``RecursionError``, taking the array task with it.

That is what killed the ``medgemma`` cells of ``an earlier exploratory run`` under the p2/p3/p5/
p6 prompts: those prompts made the model emit one very long unbroken line, and
``experiments/findings/exp13_phenobert_input_format.md`` documents that layout as the worst input
PhenoBERT can be handed anyway.

Raising the limit is *only* a reliability fix. It cannot change a parse that already
succeeded, so cells that ran before stay bit-for-bit comparable with cells backfilled after, which
is what lets the earlier ranking be recomputed over all eight models instead of seven.
"""
import os
import runpy
import sys
import threading

#: 20 000 frames of Tarjan at roughly 100 bytes of C stack each stays far inside the usual 8 MB
#: main-thread stack, so this trades a RecursionError for neither a segfault nor a silent hang.
DEFAULT_RECURSION_LIMIT = 20_000

#: stanza's parser may run inside worker threads (``annotate.py -t``), and a thread's stack is
#: sized at creation, so this has to be set before PhenoBERT spawns anything.
DEFAULT_THREAD_STACK_BYTES = 64 * 1024 * 1024


def main() -> None:
    """Run the script named in ``sys.argv[1]`` as ``__main__``, with a raised recursion limit and thread stack size."""
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <script.py> [args…]")

    sys.setrecursionlimit(int(os.environ.get("PHENOBERT_RECURSION_LIMIT", DEFAULT_RECURSION_LIMIT)))
    try:
        threading.stack_size(
            int(os.environ.get("PHENOBERT_THREAD_STACK_BYTES", DEFAULT_THREAD_STACK_BYTES))
        )
    except (ValueError, RuntimeError):
        # Some platforms refuse a non-default thread stack size. The main-thread limit above is
        # The essential half. Losing this one is not worth failing the run over.
        pass

    script = sys.argv[1]
    # Direct execution (``python annotate.py``) puts the script's directory at sys.path[0];
    # ``runpy.run_path`` does *not*. Without this, annotate.py's bare ``from util import …``
    # raises ModuleNotFoundError and the launcher breaks PhenoBERT outright.
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    # Hand the target the argv it would have had if it were invoked directly.
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
