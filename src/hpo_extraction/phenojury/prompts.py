"""Extraction prompts for the earlier prompt sweep.

The Free Listing generation run ensemble asks each SLM to "list any clinical signs or symptoms" in a sentence and
hands the free text to PhenoBERT for grounding. The 2026-08-11 deep dive found the models mostly
**echo the sentence back**, so PhenoBERT sees the string it would have seen from the raw report and
the ensemble cannot structurally beat PhenoBERT standalone.

``experiments/findings/exp13_phenobert_input_format.md`` sized the prize: on real HCY annotations,
PhenoBERT recovers the intended HPO id from the **canonical term name 89.2 %** of the time against
**58.5 %** from the report's own wording of the same annotation. Closing that 31-point gap is what
every non-baseline prompt here is trying to do.

Each spec is one **atomic change** from its predecessor (one change at a time), so a
difference in the screen's ranking is attributable to a single edit:

    p0_baseline ── p1_list ── p2_canonical ─┬─ p3_two_column
                                            ├─ p4_implicit
                                            ├─ p5_fewshot
                                            └─ p6_attribution

Two rules bind every template below, both from the same finding, and both violated by prompts that
read perfectly well in isolation:

**No numbered lists, no trailing whitespace.** ``process_text2phrases`` turns a newline after a word
character into a bare period with no space. ``terminate_lines`` (phenobert_runner) repairs the
common case, but ``Term\\n2.`` still corrupts the last word of every item (21.9 % recovery) and a
line ending in a space defeats the substitution outright (31.3 %). So the templates ask for plain
lines and say so.

**Avoid PhenoBERT's negation vocabulary.** ``getNegativeWords()`` = {no, not, none, negative, non,
never, few, lower, fewer, less, barely, normal}, matched as bare tokens *anywhere* in a candidate
phrase, and the detection is then discarded as ``Neg``. A prompt that tells the model to write
"no dysmorphic features" silently deletes its own output, which is why p6_attribution asks for
omission rather than denial, and why no template uses the word "not".

── The q-family (an earlier exploratory run) ───────────────────────────────────────────────────────────────────────

``q1_ontology_lines`` … ``q7_recall`` are a second, independent prompt-engineering pass, taken
**verbatim** from ``context/prompt_engineering.txt``. They branch from that file rather than from
p2, so they are a family and not a continuation of the chain above::

    q1_ontology_lines ─┬─ q2_sentence_last   (instruction after the sentence, not before)
                       ├─ q3_rewrite         (one prose sentence instead of a list)
                       ├─ q4_span_json       (span + term, as JSON)
                       ├─ q5_synonyms        (three label strings per finding)
                       ├─ q6_two_level       (specific term + a broader one)
                       └─ q7_recall          (propose uncertain readings too)

**They break both rules above, and are registered unedited anyway.** Rewriting somebody's prompt to
satisfy this module's priors would measure the rewrite, not the prompt, so the violations are
recorded here and *measured* in the ranking table instead of being silently repaired:

* *No template says "do not number the lines."* They ask for "one finding per line, plain text",
  which does not forbid ``1.``. Any model that numbers anyway is floored at the 21.9 % layout
  documented above. ``format_compliance`` is the column that detects it, and it is the first thing
  to read for a q-condition, a condition with low compliance is reporting a layout result, not a wording one.
* *Four of them ask for output containing negation vocabulary.* ``NONE`` (q1, q2, q5, q6, q7) and
  ``NO FINDINGS`` (q3) are sentinels, so they are filtered by ``ensemble_eval.wrote_something``
  before grounding and cost nothing. What is not free is q3, whose single-sentence contract puts
  every finding in one chunk: one trigger token anywhere in that sentence discards the lot.
  ``negation_trap_rate`` is the column for that.

Because their layouts differ, so does what "the term" means on a line, hence :data:`LINE_SHAPES`
and the ``line_shape`` field, which the diagnostics dispatch on.
"""
from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent

#: PhenoBERT's ``getNegativeWords()``. Any candidate phrase containing one of these as a bare token
#: is discarded as negated, so a prompt must never *ask* for output containing them. Exported for
#: The screen's negation-trap diagnostic, which counts how often a prompt produces them anyway.
NEGATION_TRIGGER_WORDS: frozenset[str] = frozenset({
    "no", "not", "none", "negative", "non", "never",
    "few", "lower", "fewer", "less", "barely", "normal",
})

#: What a model is told to emit when a sentence carries no phenotype. Kept identical across every
#: variant, it is the one string the screen filters out before grounding, and a variant-specific
#: spelling would make the "found nothing" case incomparable between prompts. "no phenotype" is
#: The earlier runs'own wording. It contains a negation trigger, but never reaches PhenoBERT.
NO_PHENOTYPE = "no phenotype"

#: The line shared by every restating variant. Split out because it is the part that must not drift
#: between p1..p6, if one variant silently permitted numbering, its ranking would measure the
#: layout bug rather than the wording.
_FORMAT_RULES = (
    "Write one finding per line. Do not number the lines, do not add any prefix or heading, "
    "and write nothing except the findings."
)


#: How one output line of this prompt is meant to be read. The text diagnostics
#: (:mod:`hpo_extraction.phenojury.prompt_diagnostics`) dispatch on it, so a prompt that asks for an unusual layout is
#: still judged on the part of the line that is supposed to carry the ontology term.
#:
#: Before this existed the two non-plain layouts were hard-coded by key inside
#: ``format_compliance`` / ``label_exactness``. That worked for one prompt family and silently
#: mis-scores the next: a prompt asked for a JSON object or a single rewritten sentence scores
#: 0.0 label exactness not because it wrote bad terms but because the metric was reading the
#: wrong substring, and 0.0 in a ranking table is indistinguishable from a real failure.
SHAPE_FREE = "free"                 # no layout was requested, line-shape compliance is vacuous
SHAPE_PLAIN = "plain"               # one bare term per line
SHAPE_TWO_COLUMN = "two_column"     # ``<span> => <Term>``, the term is on the right
SHAPE_PIPE_FIRST = "pipe_first"     # ``<Term> | <alt> | <alt>``, the term is the first field
SHAPE_SENTENCE = "sentence"         # one prose sentence about the patient, no per-line contract
SHAPE_JSON = "json"                 # a JSON object. Terms live in its ``term`` fields

LINE_SHAPES: frozenset[str] = frozenset({
    SHAPE_FREE, SHAPE_PLAIN, SHAPE_TWO_COLUMN, SHAPE_PIPE_FIRST, SHAPE_SENTENCE, SHAPE_JSON,
})


@dataclass(frozen=True)
class PromptSpec:
    """One extraction prompt, as the driver sends it.

    ``key`` is essential in the same way ``MODEL_KEYS`` is in the ensemble driver: it selects the
    spec, names the output directory of every (prompt, model) cell, and is the identifier that ends
    up in MLflow and in the findings table. Renaming one orphans its cached extractions.
    """

    key: str
    system: str
    user_template: str
    #: One line on what this variant changes versus its predecessor, and why, mirrored into
    #: The run log and MLflow so a result is never separated from its rationale.
    change: str
    #: One of :data:`LINE_SHAPES`. Declared beside the template it describes, because the two must
    #: agree: a template that asks for ``a | b | c`` and a shape that says ``plain`` would have the
    #: diagnostics score the whole line against the ontology and report a real prompt as a failure.
    line_shape: str = SHAPE_PLAIN

    def __post_init__(self) -> None:
        if self.line_shape not in LINE_SHAPES:
            raise ValueError(
                f"{self.key}: unknown line_shape {self.line_shape!r} "
                f"(expected one of {sorted(LINE_SHAPES)})"
            )

    def render(self, sentence: str) -> str:
        """The user message for one sentence."""
        return self.user_template.format(sent=sentence)


# ── p0, the control ─────────────────────────────────────────────────────────
# Verbatim from core/slm_ensemble_experiment.py (itself verbatim from an earlier exploratory run). Every number the
# sweep produces is a delta against this, so it must not be "improved", not even its odd spacing.
P0_BASELINE = PromptSpec(
    key="p0_baseline",
    system="You are a medical expert who recognizes signs and symptoms in medical case reports.",
    user_template=(
        "List any clinical signs or symptoms in this sentence: '{sent}' "
        "Short responses only. If there are no symptoms mentioned, respond 'no phenotype'."
    ),
    change="control — exp06/exp13_06 prompt verbatim",
    # p0 asks for "short responses" and nothing else, so there is no layout to comply with. The
    # diagnostics read this as "compliance is uninformative here", not as "compliance is perfect".
    line_shape=SHAPE_FREE,
)


# ── p1, output contract ─────────────────────────────────────────────────────
# Only the shape of the answer changes. The task and the vocabulary are still p0's. This separates
# "the model listed things cleanly" from "the model restated them", which the meeting could not:
# MedGemma's clean lists were confounded with whatever wording MedGemma happened to choose.
P1_LIST = PromptSpec(
    key="p1_list",
    system="You are a medical expert who recognizes signs and symptoms in medical case reports.",
    user_template=(
        "List any clinical signs or symptoms in this sentence: '{sent}'\n"
        f"{_FORMAT_RULES} Keep each line short. "
        f"If the sentence mentions no symptoms, write '{NO_PHENOTYPE}'."
    ),
    change="p0 + explicit one-per-line output contract (no numbering, no prefixes)",
)


# ── p2, the wording gap ─────────────────────────────────────────────────────
# The main variant: same task, same layout, but the model is asked for the *term* rather than
# The sentence's phrasing. This is the one that targets the 89.2 % vs 58.5 % gap directly, and the
# one whose failure would mean the restatement hypothesis is wrong rather than under-specified.
P2_CANONICAL = PromptSpec(
    key="p2_canonical",
    system=(
        "You are a clinical geneticist who names patient phenotypes using standard "
        "phenotype ontology terminology."
    ),
    user_template=(
        "Name each abnormal phenotypic feature of the patient described in this sentence: "
        "'{sent}'\n"
        "Write each one as its standard clinical name, the way it would appear as a term in a "
        "phenotype ontology, rather than repeating the sentence's own wording. "
        f"{_FORMAT_RULES} "
        f"If the sentence describes no abnormal feature, write '{NO_PHENOTYPE}'."
    ),
    change="p1 + canonical ontology wording instead of the sentence's phrasing",
)


# ── p3, span and term ───────────────────────────────────────────────────────
# Restating can silently drop the explicit-mention wins: when the report already says
# "microcephaly", echoing is the correct answer, and a bare "do not copy" instruction destroys it.
# Emitting both columns is strictly additive, PhenoBERT sees the canonical term *and* the original
# span, and the left column is a free hallucination check, since it must occur in the sentence.
#
# Known risk, and the reason p3 is a variant rather than the default: the copied span can drag a
# negation trigger onto the line ("was never able to sit"), and PhenoBERT discards any candidate
# phrase containing one. The `=>` separator keeps the two halves in different chunks, so the term
# should survive, but that is an assumption, and the screen's negation-trap diagnostic is what
# tests it. If p3 grounds far below p2 with a high trap rate, this is why.
P3_TWO_COLUMN = PromptSpec(
    key="p3_two_column",
    system=(
        "You are a clinical geneticist who names patient phenotypes using standard "
        "phenotype ontology terminology."
    ),
    user_template=(
        "Sentence: '{sent}'\n"
        "For each abnormal phenotypic feature of the patient, write one line of the form\n"
        "<words from the sentence> => <standard clinical name of that feature>\n"
        "The left side must be copied from the sentence. The right side is the name the feature "
        "would carry as a term in a phenotype ontology, for example "
        "'small head => Microcephaly' or 'floppy limbs => Hypotonia'. "
        f"{_FORMAT_RULES} "
        f"If the sentence describes no abnormal feature, write '{NO_PHENOTYPE}'."
    ),
    change="p2 + keep the source span alongside the canonical term (two columns)",
    line_shape=SHAPE_TWO_COLUMN,
)


# ── p4, the edge over PhenoBERT ─────────────────────────────────────────────
# PhenoBERT cannot reach a phenotype that is described but never named, the meeting's Denver
# screening-test case, the PhenoRAG paper's "was never able to sit". That gap is the ensemble's
# entire reason to exist, and no other variant asks for it.
P4_IMPLICIT = PromptSpec(
    key="p4_implicit",
    system=(
        "You are a clinical geneticist who names patient phenotypes using standard "
        "phenotype ontology terminology."
    ),
    user_template=(
        "Name each abnormal phenotypic feature of the patient described in this sentence: "
        "'{sent}'\n"
        "Include features the sentence describes without naming them: a developmental milestone "
        "the patient has failed to reach, an abnormal measurement or test result, an observed "
        "behaviour or limitation. Name the abnormality each one implies. "
        "Write each one as its standard clinical name, the way it would appear as a term in a "
        "phenotype ontology, rather than repeating the sentence's own wording. "
        f"{_FORMAT_RULES} "
        f"If the sentence describes no abnormal feature, write '{NO_PHENOTYPE}'."
    ),
    change="p2 + name phenotypes the sentence describes but never states",
)


# ── p5, style transfer ──────────────────────────────────────────────────────
# Instruction-following is a per-model variable. A demonstration reaches models that an
# instruction does not. The synthetic sentences are SYNTHETIC, inventing them rather than lifting sentences
# from HCY or GSC is what keeps the screen from leaking its own test set. One synthetic-sentence per failure
# mode the deep dive named: echo, compound merge, implicit phenotype, and the empty case.
P5_FEWSHOT = PromptSpec(
    key="p5_fewshot",
    system=(
        "You are a clinical geneticist who names patient phenotypes using standard "
        "phenotype ontology terminology."
    ),
    user_template=(
        "Name each abnormal phenotypic feature of the patient described in a sentence, writing "
        "each one as its standard clinical name, the way it would appear as a term in a phenotype "
        "ontology, rather than repeating the sentence's own wording. "
        f"{_FORMAT_RULES}\n"
        "\n"
        "Sentence: 'The boy had a small head and floppy limbs.'\n"
        "Microcephaly\n"
        "Hypotonia\n"
        "\n"
        "Sentence: 'She was still unable to sit unsupported at 14 months of age.'\n"
        "Delayed gross motor development\n"
        "\n"
        "Sentence: 'Plasma homocysteine was markedly raised and the liver was enlarged.'\n"
        "Hyperhomocysteinemia\n"
        "Hepatomegaly\n"
        "\n"
        "Sentence: 'The patient was referred by his general practitioner in March.'\n"
        f"{NO_PHENOTYPE}\n"
        "\n"
        "Sentence: '{sent}'\n"
    ),
    change="p2 + four synthetic worked examples (style transfer, not instruction)",
)


# ── p6, attribution policy ──────────────────────────────────────────────────
# Precision is the weak axis of every method in the earlier runs. The single biggest source of a spurious
# positive is a phenotype the sentence mentions but does not assert of this patient.
#
# Note the phrasing: the model is told to *leave out*, never to *write* an absent finding. Asking
# it to mark negation would produce lines carrying PhenoBERT's negation triggers, which are
# discarded as Neg, the instruction would defeat itself.
P6_ATTRIBUTION = PromptSpec(
    key="p6_attribution",
    system=(
        "You are a clinical geneticist who names patient phenotypes using standard "
        "phenotype ontology terminology."
    ),
    user_template=(
        "Name each abnormal phenotypic feature that this sentence states the patient has: "
        "'{sent}'\n"
        "Leave out anything the sentence rules out or describes as absent, anything belonging to a "
        "relative rather than the patient, anything raised only as a possibility or a plan, any "
        "finding described as within the normal range, and the names of tests, procedures, "
        "medications and diagnoses. "
        "Write each remaining feature as its standard clinical name, the way it would appear as a "
        "term in a phenotype ontology, rather than repeating the sentence's own wording. "
        f"{_FORMAT_RULES} "
        f"If nothing remains, write '{NO_PHENOTYPE}'."
    ),
    change="p2 + omit negated, family, hypothetical, normal and procedure findings",
)


# ── q1, the round-two control ───────────────────────────────────────────────
# Everything from here on is the second prompt-engineering pass. It was written by hand in
# ``context/prompt_engineering.txt`` after the an earlier exploratory run screen, and is registered here VERBATIM:
# system and user blocks byte-identical to that file, asserted by
# ``tests/unit/test_exp14_prompt_screen.py::TestRoundTwoIsVerbatim``. They are NOT
# edited to fit this module's own two house rules, and they break both, see the note on the
# q-family in the module docstring, and read ``format_compliance`` in the ranking table before
# reading any q-condition's F1.
#
# q1 is the family's own control: p2's canonical-wording instruction expanded into a full output
# contract, HPO-label capitalisation, a term cap, an explicit assertion policy, and five worked
# examples, carried in the SYSTEM message rather than the user turn.
Q1_ONTOLOGY_LINES = PromptSpec(
    key="q1_ontology_lines",
    system=dedent("""
        You extract clinical phenotype findings from clinical text and name them using
        Human Phenotype Ontology (HPO) label conventions.

        TASK
        Given one sentence, list the abnormal phenotypic features that the sentence asserts are present in the patient.

        OUTPUT FORMAT
        - One finding per line, plain text.
        - Each line is an HPO-style term name, capitalized as an HPO label
          (Microcephaly, Delayed gross motor development).
        - Write the ontology term name, not the sentence's wording. Drop severity,
          timing, and laterality qualifiers.
        - Write term names only. Never write HPO identifiers.
        - List each finding once. At most 8 lines.
        - If the sentence asserts no abnormal phenotypic feature present in the
          patient, write exactly: NONE

        WHICH FINDINGS COUNT AS PRESENT
        - Include findings the sentence states or implies are present in this patient,
          including findings implied by an abnormal measurement or an unmet milestone.
        - Write NONE when the sentence's only findings are negated, absent, ruled out,
          hypothetical, conditional, planned, in a family member rather than the
          patient, or when the sentence describes a treatment, procedure, or
          administrative event.
        - If a sentence contains both present and non-present findings, list only the
          present ones.
        - Split combined findings into components: hepatosplenomegaly becomes
          Hepatomegaly and Splenomegaly.
        - Choose the most specific term the sentence supports. Do not add specificity
          the sentence does not state.

        EXAMPLES
        <sentence>The boy had a small head and floppy limbs.</sentence>
        Microcephaly
        Hypotonia
        </example>

        <sentence>She was still unable to sit unsupported at 14 months of age.</sentence>
        Delayed gross motor development
        </example>

        <sentence>Plasma homocysteine was markedly raised and the liver was enlarged.</sentence>
        Hyperhomocysteinemia
        Hepatomegaly
        </example>

        <sentence>There were no seizures and no history of developmental regression.</sentence>
        NONE
        </example>

        <sentence>The patient was referred by his general practitioner in March.</sentence>
        NONE
        </example>
        """).strip(),
    user_template=dedent("""
        List the phenotypic features present in this patient, one per line, or NONE.
        <sentence>{sent}</sentence>
        """).strip(),
    change="prompt-engineering round 2 control — a full HPO-label output contract with five worked examples, replacing p2's single paragraph",
    line_shape=SHAPE_PLAIN,
)

# ── q2, instruction position ────────────────────────────────────────────────
# One atomic difference from q1: the user turn's two lines are swapped, so the model reads the
# sentence first and the instruction last. Same system prompt, same words, same ideas, only
# recency moves. It is the cheapest possible measurement of how much of a prompt's effect is its
# wording and how much is its position, and the pair is worth its eight extra cells
# because nothing else about them differs.
Q2_SENTENCE_LAST = PromptSpec(
    key="q2_sentence_last",
    system=dedent("""
        You extract clinical phenotype findings from clinical text and name them using
        Human Phenotype Ontology (HPO) label conventions.

        TASK
        Given one sentence, list the abnormal phenotypic features that the sentence asserts are present in the patient.

        OUTPUT FORMAT
        - One finding per line, plain text.
        - Each line is an HPO-style term name, capitalized as an HPO label
          (Microcephaly, Delayed gross motor development).
        - Write the ontology term name, not the sentence's wording. Drop severity,
          timing, and laterality qualifiers.
        - Write term names only. Never write HPO identifiers.
        - List each finding once. At most 8 lines.
        - If the sentence asserts no abnormal phenotypic feature present in the
          patient, write exactly: NONE

        WHICH FINDINGS COUNT AS PRESENT
        - Include findings the sentence states or implies are present in this patient,
          including findings implied by an abnormal measurement or an unmet milestone.
        - Write NONE when the sentence's only findings are negated, absent, ruled out,
          hypothetical, conditional, planned, in a family member rather than the
          patient, or when the sentence describes a treatment, procedure, or
          administrative event.
        - If a sentence contains both present and non-present findings, list only the
          present ones.
        - Split combined findings into components: hepatosplenomegaly becomes
          Hepatomegaly and Splenomegaly.
        - Choose the most specific term the sentence supports. Do not add specificity
          the sentence does not state.

        EXAMPLES
        <sentence>The boy had a small head and floppy limbs.</sentence>
        Microcephaly
        Hypotonia
        </example>

        <sentence>She was still unable to sit unsupported at 14 months of age.</sentence>
        Delayed gross motor development
        </example>

        <sentence>Plasma homocysteine was markedly raised and the liver was enlarged.</sentence>
        Hyperhomocysteinemia
        Hepatomegaly
        </example>

        <sentence>There were no seizures and no history of developmental regression.</sentence>
        NONE
        </example>

        <sentence>The patient was referred by his general practitioner in March.</sentence>
        NONE
        </example>
        """).strip(),
    user_template=dedent("""
        <sentence>{sent}</sentence>
        List the phenotypic features present in this patient, one per line, or NONE.
        """).strip(),
    change="q1 with the two user-message lines swapped: the sentence first, the instruction last",
    line_shape=SHAPE_PLAIN,
)

# ── q3, rewrite, not list ───────────────────────────────────────────────────
# A different output contract rather than a different instruction: the model writes one clinical
# sentence about the patient, so PhenoBERT reads prose, the input format it was built for. That
# is the argument for it. The argument against is that a single line concentrates every finding
# into one chunk, so one negation trigger anywhere in it can take the whole line down. That is
# what ``negation_trap_rate`` measures, and it is the first column to read for this condition.
#
# Its ``NO FINDINGS`` sentinel is not the library's ``no phenotype``,: it is the
# file's own wording. Both are caught by ``ensemble_eval.wrote_something`` before grounding.
Q3_REWRITE = PromptSpec(
    key="q3_rewrite",
    system=dedent("""
        You rewrite clinical sentences into standardized clinical terminology.

        TASK
        Rewrite the sentence as a single plain statement about the patient, replacing
        lay or descriptive wording with the standard clinical term for each finding.

        RULES
        - Output one sentence, beginning "The patient has ".
        - Use standard clinical term names as they would appear in a phenotype
          ontology. Do not repeat the sentence's own wording.
        - Keep only findings the sentence asserts are present in this patient. Delete
          negated, absent, ruled-out, hypothetical, planned, and family-member content.
        - Split combined findings into their components.
        - Drop severity, timing, laterality, and measurement values.
        - If nothing is asserted present in the patient, output exactly: NO FINDINGS
        - Output the rewritten sentence only. No explanation, no list, no headings.

        EXAMPLES
        <sentence>The boy had a small head and floppy limbs.</sentence>
        The patient has microcephaly and hypotonia.

        <sentence>She was still unable to sit unsupported at 14 months of age.</sentence>
        The patient has delayed gross motor development.

        <sentence>Plasma homocysteine was markedly raised and the liver was enlarged.</sentence>
        The patient has hyperhomocysteinemia and hepatomegaly.

        <sentence>He had marked hepatosplenomegaly and mild drooping of the left eyelid.</sentence>
        The patient has hepatomegaly, splenomegaly, and ptosis.

        <sentence>There were no seizures and no history of developmental regression.</sentence>
        NO FINDINGS

        <sentence>His mother suffers from epilepsy.</sentence>
        NO FINDINGS

        <sentence>The patient was referred by his general practitioner in March.</sentence>
        NO FINDINGS
        """).strip(),
    user_template=dedent("""
        <sentence>{sent}</sentence>
        Rewrite this as a single statement about the patient, or NO FINDINGS.
        """).strip(),
    change="q1 + the output is one rewritten sentence about the patient instead of a list of terms",
    line_shape=SHAPE_SENTENCE,
)

# ── q4, span-grounded JSON ──────────────────────────────────────────────────
# The earlier runs'output contract (a term is real only if a verbatim span evidences it) applied to the
# ensemble. The span is a free hallucination check, since it has to occur in the sentence.
#
# The risk is the one the Free Listing generation run already knows: PhenoBERT sees JSON punctuation rather than a list.
# That is why ``line_shape=SHAPE_JSON`` exists, the diagnostics parse the ``term`` fields out
# before judging them, so a low score here is a grounding result rather than an artefact of the
# metric reading braces.
Q4_SPAN_JSON = PromptSpec(
    key="q4_span_json",
    system=dedent("""
        You extract phenotype mentions from clinical text and normalize them to
        Human Phenotype Ontology (HPO) label conventions.

        TASK
        For each abnormal phenotypic feature the sentence asserts is present in the
        patient, output the exact substring of the sentence that evidences it, and the
        HPO-style term name for it.

        RULES
        - The "span" field must be copied character-for-character from the sentence.
          Never paraphrase or reconstruct it. If you cannot copy it exactly, omit the
          finding.
        - The "term" field is the ontology term name, capitalized as an HPO label.
        - One span may yield several terms; emit one object per term, repeating the
          span.
        - Include only findings present in this patient. Exclude negated, absent,
          hypothetical, planned, and family-member findings.
        - Output valid JSON only, matching the schema below. No prose, no code fences.

        SCHEMA
        {"findings": [{"span": "<verbatim substring>", "term": "<HPO-style label>"}]}

        EXAMPLES
        <sentence>The boy had a small head and floppy limbs.</sentence>
        {"findings": [{"span": "small head", "term": "Microcephaly"},
                      {"span": "floppy limbs", "term": "Hypotonia"}]}

        <sentence>Plasma homocysteine was markedly raised and the liver was enlarged.</sentence>
        {"findings": [{"span": "homocysteine was markedly raised", "term": "Hyperhomocysteinemia"},
                      {"span": "the liver was enlarged", "term": "Hepatomegaly"}]}

        <sentence>He had marked hepatosplenomegaly.</sentence>
        {"findings": [{"span": "hepatosplenomegaly", "term": "Hepatomegaly"},
                      {"span": "hepatosplenomegaly", "term": "Splenomegaly"}]}

        <sentence>There were no seizures and no history of developmental regression.</sentence>
        {"findings": []}

        <sentence>The patient was referred by his general practitioner in March.</sentence>
        {"findings": []}
        """).strip(),
    user_template=dedent("""
        <sentence>{sent}</sentence>
        Return the JSON.
        """).strip(),
    change="q1 + every term carries the verbatim span that evidences it, emitted as JSON",
    line_shape=SHAPE_JSON,
)

# ── q5, several label strings per finding ───────────────────────────────────
# Aimed squarely at the 89.2 % / 58.5 % gap. Rather than betting on the model picking the one
# canonical name, ask for three and give the lexical matcher three chances. The ontology label is
# The first field, which is what ``line_shape=SHAPE_PIPE_FIRST`` tells the diagnostics to judge.
#
# The bet it makes against itself: three names per line is three times the text, and a wrong
# synonym becomes a false positive at the same rate a right one becomes a recovery.
Q5_SYNONYMS = PromptSpec(
    key="q5_synonyms",
    system=dedent("""
        You name clinical findings using several alternative surface forms, so that a
        downstream lexical matcher has multiple chances to match the correct ontology
        term.

        TASK
        For each abnormal phenotypic feature the sentence asserts is present in the
        patient, output several different names for the same finding.

        RULES
        - One finding per line.
        - On each line, write 3 alternative names separated by " | ".
          1st: the term name as it would appear as a phenotype ontology label.
          2nd: a common clinical synonym or the expanded form of any abbreviation.
          3rd: a plain-language description a non-specialist would use.
        - All three names on a line must denote the same finding. Never put two
          different findings on one line.
        - Do not vary by severity or specificity across the three names. They are
          synonyms, not a hierarchy.
        - Include only findings present in this patient. If none, write exactly: NONE
        - At most 8 lines. Write nothing except the lines.

        EXAMPLES
        <sentence>The boy had a small head and floppy limbs.</sentence>
        Microcephaly | Small head circumference | Head smaller than normal
        Hypotonia | Muscular hypotonia | Floppy, low muscle tone

        <sentence>She was still unable to sit unsupported at 14 months of age.</sentence>
        Delayed gross motor development | Gross motor delay | Late to sit, crawl or walk

        <sentence>Plasma homocysteine was markedly raised and the liver was enlarged.</sentence>
        Hyperhomocysteinemia | Elevated plasma homocysteine | High homocysteine in the blood
        Hepatomegaly | Enlarged liver | Liver bigger than normal

        <sentence>There were no seizures and no history of developmental regression.</sentence>
        NONE
        """).strip(),
    user_template=dedent("""
        <sentence>{sent}</sentence>
        List each finding with its alternative names, one finding per line, or NONE.
        """).strip(),
    change="q1 + three surface forms per finding, so a lexical matcher gets three chances at the term",
    line_shape=SHAPE_PIPE_FIRST,
)

# ── q6, specific and broader ────────────────────────────────────────────────
# A hierarchy hedge: the specific term plus one that subsumes it, so a miss on the specific name
# can still land an ancestor. It is the prompt-side analogue of what an earlier exploratory run does after the fact,
# and both exist because the thesis metrics award hierarchy credit, a broader term is worth
# strictly more than nothing.
Q6_TWO_LEVEL = PromptSpec(
    key="q6_two_level",
    system=dedent("""
        You name clinical findings at two levels of specificity, so that a downstream
        system can fall back to a broader term when the specific one cannot be matched.

        TASK
        For each abnormal phenotypic feature the sentence asserts is present in the
        patient, output the most specific term the sentence supports, followed by a
        broader term that subsumes it.

        RULES
        - One finding per line, in the form: <specific term> | <broader term>
        - The specific term must not add detail the sentence does not state.
        - The broader term must be a more general phenotype ontology term that is
          certainly correct even if the specific term is wrong. Prefer an organ-system
          or anatomical-region level term.
        - The two terms must never be identical. If the sentence supports only a broad
          finding, write that finding as the specific term and go one level broader
          still for the second field.
        - Include only findings present in this patient. If none, write exactly: NONE
        - At most 8 lines. Write nothing except the lines.

        EXAMPLES
        <sentence>The boy had a small head and floppy limbs.</sentence>
        Microcephaly | Abnormality of skull size
        Hypotonia | Abnormality of muscle physiology

        <sentence>MRI showed hypoplasia of the cerebellar vermis.</sentence>
        Cerebellar vermis hypoplasia | Abnormality of the cerebellum

        <sentence>She was still unable to sit unsupported at 14 months of age.</sentence>
        Delayed gross motor development | Neurodevelopmental delay

        <sentence>Plasma homocysteine was markedly raised and the liver was enlarged.</sentence>
        Hyperhomocysteinemia | Abnormal circulating amino acid concentration
        Hepatomegaly | Abnormality of the liver

        <sentence>There were no seizures and no history of developmental regression.</sentence>
        NONE
        """).strip(),
    user_template=dedent("""
        <sentence>{sent}</sentence>
        List each finding as "specific | broader", one per line, or NONE.
        """).strip(),
    change="q1 + a broader term beside the specific one, so grounding can fall back a level",
    line_shape=SHAPE_PIPE_FIRST,
)

# ── q7, recall first ────────────────────────────────────────────────────────
# The only condition that trades precision away on purpose: propose every reading, uncertain ones
# included, on the premise that the k-of-N vote is the filter. That premise is testable here for
# free, because the vote sweep is re-run at every k, if q7 wins at k=4 and loses at k=1 the
# trade worked. If it loses everywhere, the ensemble is not the precision filter it assumes.
Q7_RECALL = PromptSpec(
    key="q7_recall",
    system=dedent("""
        You propose candidate phenotype terms from clinical text. Your goal is recall:
        a later step will remove wrong candidates, so a missed finding is far more
        costly than a spurious one.

        TASK
        Propose every abnormal phenotypic feature that the sentence might be asserting
        in the patient.

        RULES
        - One candidate per line, written as an HPO-style term name.
        - Include uncertain and weakly implied findings. Include a finding even if you
          are only somewhat confident the sentence supports it.
        - For an ambiguous phrase, propose each plausible reading on its own line.
        - Split combined findings into components, and also propose the combined form
          if it is a standard term.
        - Do not include findings that are clearly negated or clearly attributed to a
          relative.
        - At most 12 lines. If the sentence supports nothing at all, write: NONE
        - Write nothing except the lines.

        EXAMPLES
        <sentence>The boy had a small head and floppy limbs.</sentence>
        Microcephaly
        Hypotonia
        Muscle weakness
        Abnormality of the head

        <sentence>She was slow to reach her milestones and spoke her first words at three.</sentence>
        Global developmental delay
        Delayed speech and language development
        Neurodevelopmental delay
        Delayed gross motor development

        <sentence>The patient was referred by his general practitioner in March.</sentence>
        NONE
        """).strip(),
    user_template=dedent("""
        <sentence>{sent}</sentence>
        Propose candidate findings, one per line, or NONE.
        """).strip(),
    change="q1 + a recall-first instruction: propose uncertain and ambiguous readings too",
    line_shape=SHAPE_PLAIN,
)


#: The an earlier exploratory run family: one atomic edit per step, rooted at the earlier runs/phenojury_generation_free_listing control.
P_PROMPTS: tuple[PromptSpec, ...] = (
    P0_BASELINE,
    P1_LIST,
    P2_CANONICAL,
    P3_TWO_COLUMN,
    P4_IMPLICIT,
    P5_FEWSHOT,
    P6_ATTRIBUTION,
)

#: The an earlier exploratory run family, ``context/prompt_engineering.txt`` verbatim, in the file's own order.
#: APPENDED, never interleaved: the SLURM array decodes against declaration order, so inserting a
#: q-prompt among the p-prompts would repoint every already-queued an earlier exploratory run task at a different
#: prompt and silently attribute its cached cells to the wrong condition.
Q_PROMPTS: tuple[PromptSpec, ...] = (
    Q1_ONTOLOGY_LINES,
    Q2_SENTENCE_LAST,
    Q3_REWRITE,
    Q4_SPAN_JSON,
    Q5_SYNONYMS,
    Q6_TWO_LEVEL,
    Q7_RECALL,
)

PROMPTS: tuple[PromptSpec, ...] = P_PROMPTS + Q_PROMPTS

#: Declaration order. The SLURM array index decodes against this list, so appending is safe and
#: reordering is not, it would repoint every queued task at a different prompt.
PROMPT_KEYS: tuple[str, ...] = tuple(p.key for p in PROMPTS)

#: Per-family key tuples, so a config or a cluster script can name one whole family without
#: hard-coding seven strings that then drift from this file.
P_PROMPT_KEYS: tuple[str, ...] = tuple(p.key for p in P_PROMPTS)
Q_PROMPT_KEYS: tuple[str, ...] = tuple(p.key for p in Q_PROMPTS)

PROMPTS_BY_KEY: dict[str, PromptSpec] = {p.key: p for p in PROMPTS}

#: The control every other variant is measured against, and the reference for the promotion gate.
BASELINE_KEY = P0_BASELINE.key


def get_prompt(key: str) -> PromptSpec:
    """The spec for ``key``, or a ``ValueError`` naming the valid ones.

    Fails loudly rather than falling back to the baseline: a typo'd key that silently ran p0 would
    produce eight plausible-looking result files attributed to the wrong prompt.
    """
    try:
        return PROMPTS_BY_KEY[key]
    except KeyError:
        raise ValueError(
            f"unknown prompt_key {key!r} (expected one of {list(PROMPT_KEYS)})"
        ) from None
