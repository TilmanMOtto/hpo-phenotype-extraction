"""The PhenoBERT baseline's matches, placed onto this app's numbered sentences.

The PhenoBERT baseline grounds HPO terms in the *report*: it records, per match, a character span into the
document it was handed. The earlier runs'deep dive reads the same reports as a numbered list of sentences,
because ``sent_index`` is what a call record carries and the sentence number is the only join
between "what the SLM answered" and "what it was looking at". Showing the baseline's extraction
here means moving one onto the other.

Two things make that less trivial than it sounds, and both are why this module exists rather than
the view calling :mod:`hpo_extraction.curation.phenobert_output` directly:

**The offsets do not index the text this app loads.** They index
``phenobert_input/<safe_name>.txt``, the copy the PhenoBERT baseline staged for itself. ``registry.report_texts``
reads the cohort's own input directory. The two are *usually* the same bytes and are not guaranteed
to be: a restaged input beside a stale detections file differs, and highlighting with the wrong
one underlines the wrong words with complete confidence. So every span is honoured against
``pb["texts"]`` and nothing else, and :func:`annotate` renders the slice it verified, not the
sentence string the segmenter produced.

**Alignment is all-or-nothing.** ``pbstandalone.align_sentences`` places every sentence or gives
up, and this module keeps that contract. A partial map would drop the baseline's matches for the
sentences it could not place, and a reader cannot tell "PhenoBERT found nothing here" from "this
sentence was never located", the first is a result, the second is a bug in the display.

Everything here returns a reason string instead of raising. The annotation layer is additive: a
cohort with no PhenoBERT baseline run is an ordinary state, and the deep dive must open without it.
"""

from __future__ import annotations

import logging

from hpo_extraction.curation import phenobert_output as pbstandalone

log = logging.getLogger(__name__)

#: The method key ``discovery.METHODS`` gives the PhenoBERT baseline. The cell it names already knows the run
#: directory, so the walk-up search in ``pbstandalone.find_run`` is only the fallback.
METHOD_KEY = "phenobert"


def run_dir(registry, cohort: str) -> str | None:
    """The PhenoBERT baseline run directory for *cohort*, or ``None``.

    This app discovers the PhenoBERT baseline as a method in its own right (``discovery.METHODS``), so the run
    directory is normally already known and no searching is needed. The walk-up fallback covers the
    layout where the baseline sits beside the tree runs, not under the same output base.
    """
    cell = registry.cell(f"{METHOD_KEY}/{cohort}")
    if cell is not None and pbstandalone.is_run_dir(cell.run_dir):
        return cell.run_dir

    for other in registry.cells.values():
        if other.cohort == cohort:
            found = pbstandalone.find_run(other.run_dir, cohort)
            if found:
                return found
    return None


def searched(registry, cohort: str) -> list[str]:
    """Where :func:`run_dir` looked. Shown on screen when it found nothing.

    Naming the paths is the difference between "no baseline" and "no baseline *here*", and only the
    second is something a reader can act on.
    """
    paths: list[str] = []
    cell = registry.cell(f"{METHOD_KEY}/{cohort}")
    if cell is not None:
        paths.append(cell.run_dir)
    for other in registry.cells.values():
        if other.cohort == cohort:
            paths.extend(pbstandalone.searched_paths(other.run_dir, cohort))
            break
    # Order-preserving de-duplication: the walk-up produces the same parent twice on nested layouts.
    return list(dict.fromkeys(paths))


def load(path: str) -> dict | None:
    """``pbstandalone.load``, but ``None`` instead of an exception.

    A malformed or half-written the PhenoBERT baseline run must cost the deep dive its annotation layer and
    nothing else.
    """
    if not path:
        return None
    try:
        return pbstandalone.load(path)
    except Exception as exc:  # a broken baseline must not take the page down
        log.warning("PhenoBERT annotations unavailable from %s: %s", path, exc)
        return None


def annotate(pb: dict | None, report_id: str, sentences) -> tuple[dict[int, tuple], str | None]:
    """Place the baseline's matches onto *sentences*.

    Args:
        pb: a :func:`pbstandalone.load` bundle, or ``None``.
        report_id: the report to annotate.
        sentences: this app's segmentation, in ``sent_index`` order, the same list
            ``evidence.segments`` returns, whose position *is* the sentence number.

    Returns:
        ``({sent_index: (text, [(start, end, row)])}, reason)``. The mapping holds only the
        sentences that were placed *and* carry at least one match; ``text`` is the slice of the
        staged report those offsets were verified against, and the offsets are relative to it.
        ``reason`` is ``None`` on success and a short explanation otherwise, always one or the
        other, never both.
    """
    if pb is None:
        return {}, "no PhenoBERT baseline run was found for this cohort"

    text = (pb.get("texts") or {}).get(str(report_id))
    if text is None:
        return {}, ("this report has no staged text in `phenobert_input/`, the run kept its "
                    "detections but not its inputs")

    if not pbstandalone.report_ok(pb, str(report_id)):
        return {}, ("the baseline's character offsets do not land on their own matched phrases in "
                    "this report's staged text (gate G6 failed), so the annotations would "
                    "underline the wrong words")

    if not sentences:
        return {}, "this report has no recovered sentence segmentation to place the matches on"

    spans = pbstandalone.align_sentences(text, dict(enumerate(sentences)))
    if not spans:
        return {}, ("this app's sentence segmentation could not be located in the staged report "
                    "text, so no match can be attributed to a sentence")

    rows = (pb.get("by_report") or {}).get(str(report_id), [])
    placed: dict[int, tuple] = {}
    for index, (start, end) in spans.items():
        here = [(r["start"] - start, r["end"] - start, r)
                for r in rows if start <= r["start"] and r["end"] <= end]
        if here:
            placed[index] = (text[start:end], here)
    return placed, None


def counts(placed: dict[int, tuple]) -> dict[str, int]:
    """How many matches were placed, split the way the legend splits them."""
    out = {"positive": 0, "negated": 0, "unresolved": 0}
    for _text, spans in placed.values():
        for _start, _end, row in spans:
            out[kind(row)] += 1
    return out


def kind(row: dict) -> str:
    """What happened to a match, and therefore how it is drawn.

    ``positive``    in the ontology and not negated, this is what became a prediction.
    ``negated``     PhenoBERT found the phenotype and ruled it out, so it never counted.
    ``unresolved``  the code is not in this HPO release, so it was dropped.

    Identical to ``apps/phenojury_ui/views/patient.py:_kind``: the same run must not be described
    differently by the two apps that read it.
    """
    if row.get("negated"):
        return "negated"
    if not row.get("resolved", True):
        return "unresolved"
    return "positive"
