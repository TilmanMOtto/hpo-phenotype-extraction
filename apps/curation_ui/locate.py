"""Where in the report does a bare ground truth code come from?

``hcy_ground_truth_marc2.csv`` is code-only: ``SYN004`` has ``HP:0001250``, and nothing says which
sentence earned it. Approve mode cannot ask "is this right?" about a term with no evidence attached,
so this module supplies the evidence, for a code, the sentences in *this* patient's report that
contain one of the ontology's phrases for it.

Since the two rich sources arrived this is the **exception rather than the rule**, and the registry
only asks for candidates for codes nothing could be drawn for. A guess sitting beside an annotation
already underlined on its own words is noise competing with evidence, and a code whose file named
a trigger that occurs nowhere in the report is the case where a candidate still helps.

The match is lexical and word-boundary located, over ``tree.getPhrasesByHPO(code)`` (the label
plus every synonym). Longest phrase first, so *global developmental delay* is reported, not
the *delay* inside it, and a segment is reported once with its best phrase.

Hits come in two tiers, and the tier travels with the hit so the screen can show them differently:

``exact``      the ontology phrase appears verbatim (case-insensitively) in the sentence
``inflected``  it appears in another inflection, the report says *hypotonic*, the ontology says
               *hypotonia*. Only words of six letters or more are relaxed, and only in their last
               two characters plus a short suffix, so *delay* stays exact and cannot reach
               *deliver*. Exact hits always outrank inflected ones for the same code.

Lexical on purpose. The retrieval this project actually uses is a sentence-transformer
(``hpo_extraction.retrieval.similarity.SymptomScoreCalculator``), and loading it here would put torch and a GPU-shaped
model behind a login-node UI for a convenience feature. What this offers instead is explainable:
every hit names the ontology phrase it came from and the report words it matched. A code whose
evidence is genuinely paraphrased finds nothing, and the curator evidence locations it by hand, which is the
honest outcome, not a failure.

Nothing here decides anything. A hit is a button that pre-fills a segment and a trigger word. The
curator still presses it.
"""

from __future__ import annotations

import re

#: Never scan more phrases than this for one code. A handful of high-level HPO terms carry
#: hundreds of synonyms, and a curator does not benefit from the 400th.
MAX_PHRASES = 120

#: How many segments to offer per code. More than a few means the phrase is a stopword-ish one and
#: The list stops being a shortcut.
MAX_HITS = 5

#: Phrases shorter than this are skipped, one- and two-letter synonyms ("ad", "id") match noise.
MIN_PHRASE_LEN = 3

#: A word must be at least this long before its ending is relaxed. Below it, the stem left over is
#: too short to be specific: relaxing *delay* would reach *deliver*.
MIN_INFLECT_LEN = 6

#: How many characters an inflected ending may run to. Covers -s, -ic, -ies, -ally.
MAX_SUFFIX = 4


def phrases_for(tree, code: str) -> list[str]:
    """Every ontology phrase for *code*, longest first, capped. Empty for an unknown code."""
    try:
        phrases = tree.getPhrasesByHPO(code)
    except Exception:  # noqa: BLE001 - an unknown code has no phrases, which is not an error here
        return []
    seen: set[str] = set()
    out: list[str] = []
    for phrase in phrases:
        phrase = (phrase or "").strip().lower()
        if len(phrase) < MIN_PHRASE_LEN or phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
    out.sort(key=lambda p: (-len(p), p))
    return out[:MAX_PHRASES]


def find_triggers(tree, code: str, segments: list[str], max_hits: int = MAX_HITS) -> list[dict]:
    """``[{"segment_idx", "trigger_word", "phrase", "start", "end"}, …]``, best hit per segment.

    ``trigger_word`` is the text **as the segment spells it**, not the ontology's phrase: it is what
    gets written to the curated file, and a curator reviewing the CSV must see the report's words.
    """
    candidates = phrases_for(tree, code)
    if not candidates:
        return []

    exact = [(p, re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE)) for p in candidates]
    inflected = [(p, re.compile(_inflected_pattern(p), re.IGNORECASE)) for p in candidates]

    hits: list[dict] = []
    for tier, patterns in (("exact", exact), ("inflected", inflected)):
        for idx, segment in enumerate(segments):
            if any(h["segment_idx"] == idx for h in hits):
                continue                      # this segment already has its best hit
            for phrase, pattern in patterns:
                match = pattern.search(segment)
                if match is None:
                    continue
                # Phrases are longest-first, so the first hit in a segment is already its best.
                hits.append({
                    "segment_idx": idx,
                    "trigger_word": segment[match.start():match.end()],
                    "phrase": phrase,
                    "tier": tier,
                    "start": match.start(),
                    "end": match.end(),
                })
                break
            if len(hits) >= max_hits:
                return hits
    return hits


def _inflected_pattern(phrase: str) -> str:
    r"""A regex for *phrase* that tolerates a different ending on each long word.

    ``hypotonia`` → ``\bhypotoni[a-z]{0,4}\b``, which reaches *hypotonic*. Short words are left
    exact, and the words are joined by ``[\s\-]+`` so a hyphenated writing still matches.
    """
    parts = []
    for word in phrase.split():
        if len(word) >= MIN_INFLECT_LEN and word.isalpha():
            parts.append(re.escape(word[:-2]) + "[a-z]{0," + str(MAX_SUFFIX) + "}")
        else:
            parts.append(re.escape(word))
    return r"\b" + r"[\s\-]+".join(parts) + r"\b"


def find_all(tree, codes, segments: list[str]) -> dict[str, list[dict]]:
    """:func:`find_triggers` for a set of codes, what one patient's Approve panel needs at once."""
    return {code: find_triggers(tree, code, segments) for code in codes}
