"""Difficulty labels, *why* an annotation is hard, so recall can later be sliced by it.

A model that misses a phenotype misses it for a reason, and "recall = 0.71" hides every one of
them. The ground-truth set as it stands cannot answer whether the misses are concentrated in terms the
report never names, in terms belonging to a *relative* rather than the patient, or in codes that
were arguably wrong to begin with. That is the question these labels exist to make answerable: they
are recorded during curation, exported long-form, and joined against predictions afterwards.

Two levels, because two different things are hard:

``annotation``  what kind of annotation this is, negated, a relative's, a lab value, implicit,
                doubtful. The label travels with the (patient, code) pair, so a per-term recall
                breakdown is a group-by
``patient``     how hard the whole report is, and why, is this a document whose phenotypes belong
                to relatives, a document that describes, not names

and one **grade** per level (:func:`difficulty_for`), single-valued, because "how hard" and "why
hard" are different questions and conflating them makes the first one unanswerable. The grade is
what a stratified recall table splits on. The labels are what explains the split.

**The two levels have disjoint vocabularies, and that is a change from how this started.** They
used to share ids, *Family confusion* meaning the same thing at either level, on the theory that
"how often is this the problem, at either level" should be one group-by. In practice the annotation
level was nineteen checkboxes and a grade on every row, and a vocabulary that fine is one nobody
applies. An unapplied label aggregates to nothing at all, which is worse than a coarse one.

So the annotation level is now the seven qualifiers in the ``qualifier`` group, ticked on the
suggestion form itself and repaired in the same inline editor as the trigger word, they travel
*with* the annotation, not being a second job done afterwards. The report level keeps the
original vocabulary and its grade, because "what kind of report is this" is asked once and a
handful of coarse buckets covers it.

The fine-grained labels that were annotation-only, inflection, granularity, several-codes-fit, are **retired, not deleted**: they keep their entry in :data:`INDEX`, so a log that recorded
one still renders its name and its definition, but their ``scopes`` tuple is empty, so nothing
offers them and :func:`clean` accepts them nowhere. A log outlives the vocabulary that wrote it.

A label's ``scopes`` tuple is what says where it may be applied, and :func:`clean` enforces it
against anything arriving from a browser.

Nothing here is inferred. Every label is something a human decided while looking at the report, which
is what makes it worth joining against a model's output, a label the pipeline could have
derived would just be measuring the pipeline against itself.

The vocabulary is closed. Free-text tags do not aggregate: "fam hx", "family", and
"relative" are three columns in a group-by and one concept in the annotator's head. Anything that
does not fit goes in the note field, and if it recurs it earns a label here.
"""

from __future__ import annotations

#: ``(id, display, when to use it, scopes)`` grouped by the question the group answers.
#: A group appears at a level when any of its labels do.
GROUPS: list[dict] = [
    {
        "id": "qualifier",
        "title": "What kind of annotation is this?",
        "help": "The seven things that turn out to be worth knowing about a term while curating "
                "it. Deliberately coarse and deliberately few: a vocabulary a curator has to "
                "think about is one they stop applying, and an unapplied label aggregates to "
                "nothing. Tick as many as are true.",
        "labels": [
            ("unsure_report",
             "Unsure (from report)",
             "The report itself does not settle it. Reading the sentence again would not help — "
             "the text is genuinely ambiguous about whether this phenotype is present.",
             ("annotation",)),
            ("unsure_annotation",
             "Unsure (from annotation)",
             "The report is clear enough; it is the annotation that is doubtful — the wrong code, "
             "the wrong words, or a claim the sentence does not support.",
             ("annotation",)),
            ("family",
             "Family",
             "The finding belongs to a relative, not to the patient. Family history, a sibling, a "
             "parent. A model that reads the sentence and extracts the term is wrong for a reason "
             "worth counting on its own.",
             ("annotation",)),
            ("resolved",
             "Resolved",
             "Present in the past and not current at the time of the report — or explicitly "
             "described as having gone away.",
             ("annotation",)),
            ("negated",
             "Negated",
             "The report states the phenotype is absent. Kept as a label rather than settled as a "
             "verdict, because a gold file that carries a negated term anyway is a labelling "
             "error worth measuring rather than just deleting.",
             ("annotation",)),
            ("lab_value",
             "Lab value",
             "The evidence is a measurement rather than a phrase — a haemoglobin figure, a "
             "percentile, a level with a unit. Nothing lexical points at the term.",
             ("annotation",)),
            ("implicit",
             "Implicit",
             "Described but never named. \u201ccould not hold his head up\u201d for hypotonia. No lexical "
             "route to the term exists; only a reader who understands the sentence finds it.",
             ("annotation",)),
        ],
    },
    {
        "id": "attribution",
        "title": "Whose phenotype is it?",
        "help": "The failure mode where a term is real, present in the text, and not the "
                "patient's. Extraction systems are bad at this and the gold set cannot currently "
                "say how often it happens.",
        "labels": [
            ("family_member",
             "Family confusion",
             "The text describes a relative — family history, a sibling, a parent — not the "
             "patient. A model that reads the sentence and extracts the term is wrong for a "
             "reason worth counting separately.",
             ("patient",)),
            ("hypothetical",
             "Suspected / differential",
             "Considered, queried or being ruled out, not established. “to exclude epilepsy”, "
             "“cannot rule out anemia”.",
             ("patient",)),
            ("historical",
             "Historical / resolved",
             "Present in the past and not current at the time of the report.",
             ("patient",)),
            ("negated_in_text",
             "Explicitly negated",
             "The report states the phenotype is absent. Kept as a label rather than a verdict "
             "because a gold file that carries it anyway is a labelling error worth measuring, "
             "not just deleting.",
             ("patient",)),
        ],
    },
    {
        "id": "expression",
        "title": "How is it written?",
        "help": "The retrieval difficulty axis. A term the report names outright and a term it "
                "only implies are not the same task, and the current gold set treats them "
                "identically.",
        "labels": [
            ("direct_term",
             "Direct word",
             "The report uses the ontology's own term, or one of its synonyms, verbatim. "
             "“recurrent seizures” for Seizure. The easy case, and the baseline everything else "
             "is measured against.",
             ("patient",)),
            ("inflection",
             "Inflected form",
             "The same word in another form — “hypotonic” for Hypotonia, “atrophic” for Atrophy. "
             "Trivial for a human, a miss for exact string matching. Annotation-only: it is a "
             "property of one wording, and a whole report is never uniformly inflected.",
             ()),
            ("abbreviation",
             "Abbreviation",
             "GDD, FTT, NDD, ASD. Short, ambiguous, and usually absent from the ontology's "
             "synonym list.",
             ("patient",)),
            ("paraphrase",
             "Indirectly described",
             "The phenotype is described but never named — “could not hold his head up” for "
             "hypotonia. No lexical route to the term exists; only a model that understands the "
             "sentence can find it.",
             ("patient",)),
            ("inferred",
             "Needs clinical inference",
             "Derived from a finding or a measurement rather than stated — a haemoglobin value "
             "implying anemia, a percentile implying failure to thrive.",
             ("patient",)),
            ("multi_sentence",
             "Evidence spans sentences",
             "No single segment carries the annotation; it takes two or more read together. "
             "Sentence-level pipelines cannot see this by construction.",
             ("patient",)),
            ("non_english",
             "Translation artefact",
             "The wording is a translation or was left untranslated, and that is what makes it "
             "hard rather than the phenotype itself. Annotation-only: it describes one phrase, "
             "not the document, which is translated in its entirety or not at all.",
             ()),
        ],
    },
    {
        "id": "ontology",
        "title": "Is the code right?",
        "help": "Difficulty that belongs to the labelling rather than the text. These are the "
                "rows where a model marked 'wrong' may in fact have been reasonable. "
                "Annotation-only — every one of them is a statement about a particular code.",
        "labels": [
            ("granularity",
             "Granularity mismatch",
             "The text supports a more or a less specific term than the code records. The "
             "commonest source of an unfair false positive: the model said the parent, the gold "
             "says the child.",
             ()),
            ("ambiguous",
             "Several codes fit",
             "The wording supports more than one HPO term equally well, and the choice between "
             "them was arbitrary.",
             ()),
            ("outdated_code",
             "Obsolete / unresolvable code",
             "Not a phenotypic-abnormality term in the current hpo.json — nothing can score "
             "against it either way.",
             ()),
            ("disputed",
             "Genuinely unclear",
             "The two annotators disagree and reading the report does not settle it. Distinct "
             "from a plain verdict: this says the disagreement is the *text's* fault.",
             ()),
        ],
    },
]

#: The single-valued grade, per level. Kept apart from :data:`GROUPS` because "how hard" and "why
#: hard" are different questions: the grade is what a stratified recall table splits on, and it
#: only works as a split if one value can be true.
#:
#: ``unfair`` is annotation-only. "This term should not count against a model" is a claim about one
#: term with one piece of evidence. A whole report is rarely unfair, and grading it so would quietly
#: excuse every annotation in it.
DIFFICULTY = [
    ("easy", "Easy", "Stated outright; any reasonable system should find it.",
     ("patient",)),
    ("medium", "Medium", "Findable, but not by string matching alone.",
     ("patient",)),
    ("hard", "Hard", "Needs inference, context, or clinical knowledge.",
     ("patient",)),
    ("unfair", "Unfair",
     "Should not count against a model — the evidence is not in the text, or the code is wrong. "
     "Recorded rather than deleted, so the size of this bucket is itself a result.",
     ()),
]

DIFFICULTY_IDS = [value for value, _, _, _ in DIFFICULTY]

SCOPES = ("annotation", "patient")


def difficulty_for(scope: str) -> list[tuple[str, str, str]]:
    """``[(id, display, help)]``, the grades offered at *scope*."""
    return [(value, display, help_text)
            for value, display, help_text, scopes in DIFFICULTY if scope in scopes]


def labels_for(group: dict, scope: str) -> list[tuple[str, str, str]]:
    """``[(id, display, help)]``, the labels of *group* offered at *scope*."""
    return [(value, display, help_text)
            for value, display, help_text, scopes in group["labels"] if scope in scopes]


def groups_for(scope: str) -> list[dict]:
    """The label groups with at least one label applicable at *scope*."""
    return [group for group in GROUPS if labels_for(group, scope)]


def _index() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for group in GROUPS:
        for value, display_name, help_text, scopes in group["labels"]:
            out[value] = {"value": value, "display": display_name, "help": help_text,
                          "group": group["id"], "group_title": group["title"],
                          "scopes": scopes}
    for value, display_name, help_text, scopes in DIFFICULTY:
        out[value] = {"value": value, "display": display_name, "help": help_text,
                      "group": "difficulty", "group_title": "Difficulty", "scopes": scopes}
    return out


#: ``{label id: metadata}`` for every label and every difficulty grade.
INDEX = _index()

#: Every valid label id at each scope, for validating what arrives from a browser.
VALID = {scope: {v for v, m in INDEX.items() if scope in m["scopes"]} for scope in SCOPES}


def display(value: str) -> str:
    """The human name for a label id. The id itself for one this version does not know.

    A log written by a newer version must still open here, and an unknown label is data to show,
    not a reason to fail.
    """
    entry = INDEX.get(value)
    return entry["display"] if entry else value


def help_for(value: str) -> str:
    """Help text of one difficulty label, or a note that the label is unknown to this version."""
    entry = INDEX.get(value)
    return entry["help"] if entry else "Unknown label — recorded by another version of this app."


def grade_ok(value: str, scope: str) -> bool:
    """Whether *value* is a grade offered at *scope*. ``unfair`` at report level is not."""
    return value in {v for v, _, _ in difficulty_for(scope)}


def clean(values, scope: str) -> list[str]:
    """The valid labels in *values* for *scope*, de-duplicated, in vocabulary order.

    Vocabulary order, not click order, so two curators who tick the same boxes produce the
    same row and a diff of the exported file shows real changes only. Grades are dropped: they live
    in their own single-valued field, and one leaking into the set would double-count.
    """
    allowed = VALID[scope]
    chosen = {str(v) for v in (values or []) if str(v) in allowed}
    return [v for v in INDEX if v in chosen and v not in DIFFICULTY_IDS]


def label_ids(scope: str) -> list[str]:
    """Every label id offered at *scope*, in vocabulary order, the report file's column order."""
    return [value for group in GROUPS for value, _, _ in labels_for(group, scope)]


#: Annotation-source labels that the stored curation log on the cluster still uses, mapped to the
#: labels the code uses now. ``holistic`` and ``marc2`` were renamed for the published
#: repository. The log is append-only and is never rewritten, so old events are translated when
#: they are read (:func:`translate_stored_labels`), and a log that mixes both spellings folds to
#: The same state as one written entirely with the new labels.
STORED_SOURCE_ALIASES: dict[str, str] = {
    "holistic": "prior_annotation",
    "marc2": "prior_annotation_2",
}


def translate_stored_labels(value):
    """Return ``value`` with old source labels replaced by the current ones.

    Args:
        value: one parsed log event (a dict), or any string, list or dict inside one.

    Returns:
        The same structure. A string equal to an old label, or a key that starts with
        ``"<old label>|"`` (for example ``holistic|SYN004|HP:0001250``), gets the new label. Every
        other value is returned unchanged.
    """
    if isinstance(value, str):
        if value in STORED_SOURCE_ALIASES:
            return STORED_SOURCE_ALIASES[value]
        head, sep, rest = value.partition("|")
        if sep and head in STORED_SOURCE_ALIASES:
            return f"{STORED_SOURCE_ALIASES[head]}|{rest}"
        return value
    if isinstance(value, dict):
        return {k: translate_stored_labels(v) for k, v in value.items()}
    if isinstance(value, list):
        return [translate_stored_labels(v) for v in value]
    return value
