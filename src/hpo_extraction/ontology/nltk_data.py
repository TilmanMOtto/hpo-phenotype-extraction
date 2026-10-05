"""Where NLTK looks for its corpora, so a checkout is self-sufficient for the ones we vendor.

NLTK searches ``~/nltk_data`` first and, failing that, a handful of directories under ``sys.prefix``
and ``/usr/share``. Every one of those is *per user*: the corpora arrive via
``download_resources.py``, which writes into the home directory of whoever ran it.

That is fine for one person and breaks the moment a second one shares the work. The curation UI is
the case in point, it is run by clinical curators who have access to the group's work directory and
no access to anybody's home, and the failure is the worst shape available: ``stopwords.words()`` is
evaluated in a **class body** in :mod:`hpo_extraction.ontology.hpo_items`, so the whole app dies with a
``LookupError`` at import, before any path in its sidebar has had a chance to matter.

So the repo carries the corpora it can afford to carry, and this module puts them on the search
path. ``resources/nltk_data`` goes **first**: a checkout should behave the same way on every machine
it is on, and a stale copy in somebody's home silently winning is the kind of difference
that is only ever discovered from a result that will not reproduce.

Only ``stopwords`` is vendored, 180 KB, and the only corpus the curation UI touches (verified by
running its selftest against a search path containing nothing else). ``wordnet``, ``punkt`` and
``averaged_perceptron_tagger`` are tens of megabytes, are needed by the segmentation and clinical-NLP
paths rather than by the UI, and stay a ``download_resources.py`` job. Both mechanisms coexist:
this prepends, it never replaces, so a machine that has run the downloader keeps everything it has.

``NLTK_DATA`` is still honoured by NLTK itself and still works. It is no longer *required*.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

from hpo_extraction.paths import REPO_ROOT as _REPO_ROOT

#: The vendored corpora, tracked in git so they travel with the checkout.
BUNDLED = _REPO_ROOT / "resources" / "nltk_data"

#: What :data:`BUNDLED` actually contains, so a caller can say what is and is not covered.
BUNDLED_CORPORA = ("stopwords",)


def ensure_nltk_path() -> str | None:
    """Put the repo's vendored corpora at the front of ``nltk.data.path``. Idempotent.

    Returns the directory added, or ``None`` when there is nothing to add, a checkout without
    ``resources/nltk_data`` is not an error here, it just falls back to whatever the machine has.
    """
    if not BUNDLED.is_dir():
        return None
    import nltk

    bundled = str(BUNDLED)
    if bundled in nltk.data.path:
        return bundled
    nltk.data.path.insert(0, bundled)
    logger.debug("nltk.data.path now leads with %s", bundled)
    return bundled


def missing_corpora() -> list[str]:
    """Which of :data:`BUNDLED_CORPORA` cannot be resolved, for a startup check worth making.

    Kept separate from :func:`ensure_nltk_path` because "extend the search path" and "tell me what
    is still missing" are different questions, and only the second one is worth reporting to a user.
    """
    ensure_nltk_path()
    import nltk

    missing = []
    for name in BUNDLED_CORPORA:
        try:
            nltk.data.find(f"corpora/{name}")
        except LookupError:
            missing.append(name)
    return missing


def describe() -> str:
    """One line for a log: where the corpora are coming from."""
    added = ensure_nltk_path()
    if added is None:
        return f"nltk: no bundled corpora at {BUNDLED} — falling back to NLTK_DATA / ~/nltk_data"
    return f"nltk: {', '.join(BUNDLED_CORPORA)} from {added} (NLTK_DATA still honoured)"


# Importing this module is enough. ``hpo_extraction.ontology.hpo_items`` reads ``stopwords.words()`` in a class
# body, so the path has to be right before that module is *executed*, not before it is called.
ensure_nltk_path()


if os.environ.get("PHENORAG_NLTK_DEBUG"):  # pragma: no cover - a hook for diagnosing a bad machine
    logging.basicConfig(level=logging.DEBUG)
    logger.debug("%s", describe())
