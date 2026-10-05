"""The two cohorts this app can read, and every place they differ.

``compare_ui`` was built for HCY and hard-coded it in nine places: the artifact subdirectory each
driver writes into, the ground truth and how an annotated term is placed on a sentence, how the report text is
decoded, which row of ``comparison``'s ``t1_overall.csv`` is the quotable one, and where the
protocol methods' prediction sets live. Adding GSC+ by threading a string through those nine
places would work until the tenth, so they are named once here instead.

The two are genuinely different objects and the differences are not cosmetic:

``hcy``
    118 German clinical reports, **patient data**. Read latin1 (``hpo_extraction.data.loading.load_txt``). Ground truth is a ``curated_ground_truth_<date>`` dataset whose annotations carry a segment index, a trigger
    word and curation qualifiers, so a drawn underline is something a person recorded on that
    sentence. Nothing here leaves the cluster.

``gsc``
    The 114 published journal abstracts RAG-HPO evaluated on, under **RAG-HPO's own
    re-annotation** (``resources/data/GSC_RAGHPO``. See its ``PROVENANCE.md``). Read UTF-8.
    Published text, so it may be read by a model and may be pulled off the cluster -- which is
    what makes a GSC+ comparison shareable in a way the HCY one is not.

    Its ground truth carries **no offsets and no trigger words**: RAG-HPO's annotation is
    ``(doc_id, hpo_id, hpo_description)`` and nothing else. Placement therefore runs a ladder,
    and the rungs are distinguished on screen rather than merged -- see :mod:`apps.compare_ui.gold`.

The cohort a bundle was built for is written into the bundle and into the index, so the app never
infers it from a directory name.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cohort:
    """One reading cohort: where its text is, what its ground truth is, and what may be said about it."""

    key: str
    label: str
    #: The sidebar's one-line description of what the reader is looking at.
    blurb: str
    #: The subdirectory every ``fixed`` driver writes into, under ``{output_base}/{exp_id}/``.
    #: ``gsc_raghpo_ann`` is a *derived* cohort -- no driver was ever run against it -- so its
    #: artifacts are the full-GSC+ run's, restricted at scoring time. This is that distinction.
    artifact_dir: str
    #: The ``cohort`` value ``comparison/tables/t1_overall.csv`` uses. The only
    #: rates this app may quote are that file's, so getting this wrong prints another cohort's
    #: numbers under this cohort's heading.
    scored_cohort: str
    #: Which ground truth source builds this cohort's terms -- see :mod:`apps.compare_ui.gold`.
    gold_kind: str
    #: How the report text is decoded. Named because reading HCY as UTF-8 silently changes every
    #: report containing an umlaut, which is all of them.
    encoding: str
    #: True when the text may not leave the machine it is on.
    patient_data: bool
    #: What the cohort's documents are called in prose, singular and plural.
    unit: str
    units: str
    #: The gutter note this cohort's ground truth requires, or ``""``.
    gold_note: str = ""


HCY = Cohort(
    key="hcy",
    label="HCY",
    blurb="five systems, one HCY report",
    artifact_dir="hcy",
    scored_cohort="hcy",
    gold_kind="curated",
    encoding="latin1",
    patient_data=True,
    unit="report",
    units="reports",
)

GSC = Cohort(
    key="gsc",
    label="GSC+ (RAG-HPO subset)",
    blurb="five systems, one GSC+ abstract",
    artifact_dir="gsc",
    scored_cohort="gsc_raghpo_ann",
    gold_kind="raghpo",
    encoding="utf-8",
    patient_data=False,
    unit="abstract",
    units="abstracts",
    gold_note=(
        "The ground truth is RAG-HPO's own re-annotation of these 114 documents (resources/data/"
        "GSC_RAGHPO), which records a term and a description and no position. A term is drawn on "
        "a sentence only when the GSC+ corpus itself annotates the same code on this document, "
        "or when RAG-HPO's description occurs verbatim in it; everything else is listed under the "
        "grid rather than underlined somewhere plausible."),
)

ALL: tuple[Cohort, ...] = (HCY, GSC)
BY_KEY: dict[str, Cohort] = {c.key: c for c in ALL}

#: What ``--cohort`` accepts, in the order the dataset picker lists them.
KEYS: tuple[str, ...] = tuple(c.key for c in ALL)

DEFAULT = HCY


def get(key: str) -> Cohort:
    """The cohort named *key*. Raises, not defaulting -- see :func:`resolve`."""
    try:
        return BY_KEY[str(key)]
    except KeyError:
        raise ValueError("unknown cohort {!r}; expected one of {}".format(key, ", ".join(KEYS)))


def resolve(key, default: Cohort = DEFAULT) -> Cohort:
    """The cohort named *key*, or *default* when it names nothing.

    For the **read** side only. A bundle written by an older builder carries no cohort key, and an
    app that refused to open it would turn a schema gap into an outage. A builder handed a cohort
    it does not know must fail instead, which is why :func:`get` raises.
    """
    return BY_KEY.get(str(key or ""), default)
