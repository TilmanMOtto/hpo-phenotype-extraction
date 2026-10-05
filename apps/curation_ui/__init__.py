"""Patient-by-patient curation of the HCY ground-truth set.

The ground truth every earlier number is scored against has three sources, and they do not agree:

``hcy_holistic_ground_truth.csv``  one row per annotation, the code, the **trigger word**, the
                                   sentence it sits in, a character offset, and the verbatim report
``annotations_confirmed.csv``      one row per annotation, already located to a **segment index**,
                                   with a provenance and a confirmed flag
``hcy_ground_truth_marc2.csv``     the two-column ``patient_id, hpo_codes`` shape, **code-only**,
                                   so it says nothing at all about where a term came from

The first two replaced ``hcy_ground_truth_raw.csv``, which was code-only like ``prior_annotation_2``. That is
the difference this app is built around: an annotation that names its own evidence can be *drawn on
the sentence it came from*, while a bare code can only be guessed at (:mod:`locate`). PhenoBERT
(``baseline_phenobert``) meanwhile produced per-detection output with exact character offsets that has never
been walked against any of them.

They are also not equals. ``annotations_confirmed.csv`` is a confirmed pass over the prior_annotation set,
so **it** is what Approve mode rules on. Where it dropped a term the prior_annotation file still carries,
the prior_annotation row stands in, because that drop is itself a judgement somebody has to confirm.
``prior_annotation_2`` and PhenoBERT are **reference**, drawn, listed and named in every row's agreement chip,
never asked about, because nothing of either can enter the curated ground truth. That ordering is
``sources.ADJUDICATION_ORDER``.

This app is that walk. One screen per patient: the segmented report with every annotation and
detection marked in place, an **Edit** mode for proposing a phenotype (segment + trigger word + HPO
code) and an **Approve** mode for adjudicating those proposals *and* passing a keep/remove verdict
on every annotation that already exists. Anything on the screen can also be **repaired**, its
segment, its trigger word, its term, because a file written against a different segmentation is
right about the phenotype and wrong about the place (:mod:`views.editor`).

Not to be confused with ``app/annotation_ui``, the ancestor of this idea, kept as a legacy tool.
That app ships all ~19 000 HPO options and the whole annotation set into browser stores and
rewrites its autosave CSV on a 10-second interval. This one keeps every heavy object server-side
(:mod:`registry`), searches the ontology in a callback capped at 50 hits (:mod:`search`), renders
one patient at a time, and persists by appending a single line per action (:mod:`store`).

Run it with ``python apps/curation_ui/app.py``. See ``README.md``.
"""
