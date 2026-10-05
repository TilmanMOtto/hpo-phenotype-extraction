"""Stage C, deterministic candidate retrieval over the fixed ontology.

Fully deterministic: no model call, no sampling, no learned component. Given a span, assemble the
list of ontology terms it could plausibly denote, with enough metadata for Stage D to make a
specificity decision and Stage F to check coherence.

Four indices, because the four fail on different spans:

* **Exact**, the term's own label. Catches *craniosynostosis*.
* **Synonym**, every synonym type the release carries, including obsolete-era wording such as
  "mental retardation". Catches the register the letter is actually written in.
* **Definition**, the ``def:`` text, embedded. This is the only path by which *axial flexor
  movements involving head and eyes* reaches *Epileptic spasm*, whose label and synonyms share not
  one word with the span. Lexical retrieval cannot do it at any threshold.
* **Abbreviation**, a curated expansion table, because ``C3``, ``5MTHF``, ``tHcy`` and ``MCV``
  appear in no HPO label or synonym and are how the letters are actually written.

Plus a fuzzy pass over the lexical keys, tolerant to the optical-recognition damage Stage 0 flagged
but did not repair (see ``span_detection``'s module docstring): ``homocvsteinaemia``
must still retrieve *homocysteinemia*, and it does so here at query time rather than by editing the
patient's letter.

**Why this does not reuse ``HPOTree.p_phrase2HPO``.** That map is built with ``last write wins``, every label and synonym in the release is hashed to a sorted bag of processed words, and a collision
silently overwrites. It answers "one term that could match", which is the wrong question for a
candidate list: the whole point of Stage C is to hand Stage D the terms that compete, including the
near-synonyms that make a specificity decision necessary. The indices here keep every id per key.
"""

from __future__ import annotations

import csv
import functools
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from hpo_extraction.ontology.hpo_items import WordItem, processStr
from hpo_extraction.ontology.hpo_tree import HPO_class, HPOTree

logger = logging.getLogger(__name__)

from hpo_extraction.paths import REPO_ROOT as _REPO_ROOT
DEFAULT_ABBREV_PATH = _REPO_ROOT / "resources" / "reference" / "abbreviations.csv"

#: Where a candidate came from, in the order a tie is broken.
SOURCES = ("exact", "synonym", "abbreviation", "partial", "fuzzy", "definition")


def read_reference_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a ``resources/reference/`` table, skipping ``#`` comment lines.

    Every reference asset carries a header comment explaining what its columns mean and what a
    blank cell is allowed to signify, that documentation is the difference between a table someone
    can extend correctly and one they cannot. ``csv.DictReader`` has no notion of comments, so it
    would take the first ``#`` line as the header and silently yield rows keyed by ``'# Stage C
    expansion table ...'``. Every lookup then misses and the index loads empty, with no error: the
    exact failure this function exists to prevent, and one that was caught here only by auditing
    the loaded row count against the file.
    """
    lines = [
        line
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return list(csv.DictReader(lines))


@dataclass
class Candidate:
    """One ontology term offered to Stage D, with everything needed to choose between candidates."""

    hpo_id: str
    label: str
    synonyms: list[str] = field(default_factory=list)
    definition: str = ""
    direct_parents: list[str] = field(default_factory=list)
    direct_parent_labels: list[str] = field(default_factory=list)
    direct_children: list[str] = field(default_factory=list)
    ancestors: list[str] = field(default_factory=list)
    depth: int | None = None
    sources: list[str] = field(default_factory=list)
    score: float = 0.0
    obsolete_match: bool = False


# ---------------------------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------------------------


def normalise_phrase(text: str) -> str:
    """The repository's own phrase key: accent-stripped, lowercased, tokenised, **sorted**.

    Sorting makes the key a bag of words, so "nails, hypoplastic" and "hypoplastic nails" collide on
    purpose. Reusing ``processStr``, not writing a new normaliser keeps this index in the same
    space as ``HPOTree.matchPhrase2HPO``, so a term reachable there is reachable here.
    """
    return " ".join(sorted(processStr(text)))


def lemma_key(text: str) -> str:
    """The lemmatised variant of :func:`normalise_phrase`, using the shared lemma cache."""
    return " ".join(
        sorted(WordItem.lemma_dict.get(tok, tok) for tok in processStr(text))
    )


#: Morpheme pairs that invert a term's meaning. A fuzzy match that crosses one of these is
#: rejected outright, at any ratio.
#:
#: Threshold tuning cannot substitute for this and it is worth being precise about why. Measured on
#: this ontology, ``macrocytosis`` scores **0.917** against ``microcytosis``, its opposite, and
#: only **0.867** against the intended repair of ``homocvsteinaemia``. The antonym is *closer* than
#: The true match, so every threshold that admits the repair also admits the inversion, and every
#: threshold that excludes the inversion also excludes the repair. The ordering is inverted, so no
#: cutoff separates them.
#:
#: This is the retrieval-layer half of the specification's direction rule ("choose NONE, not
#: guessing a direction"): a candidate list that contains only the wrong pole gives Stage D nothing
#: to abstain *from*.
ANTONYM_MORPHEMES: tuple[tuple[str, str], ...] = (
    ("macro", "micro"),
    ("hyper", "hypo"),
    ("increas", "decreas"),
    ("elevat", "reduc"),
    ("poly", "oligo"),
    ("brady", "tachy"),
    ("hyperplas", "hypoplas"),
    ("over", "under"),
    ("high", "low"),
    ("large", "small"),
    ("long", "short"),
    ("abov", "below"),
)


def crosses_polarity(a: str, b: str) -> bool:
    """True if one of the two strings carries each side of an antonym pair.

    Requiring *one* on each side is what keeps ``hypertelorism`` matching itself: both
    strings carry ``hyper``, neither carries ``hypo``, so nothing crosses. It fires only when one
    string says one pole and the other says the other.
    """
    for left, right in ANTONYM_MORPHEMES:
        a_l, a_r = left in a, right in a
        b_l, b_r = left in b, right in b
        if (a_l and not a_r and b_r and not b_l) or (a_r and not a_l and b_l and not b_r):
            return True
    return False


def _trigrams(key: str) -> set[str]:
    packed = key.replace(" ", "")
    return {packed[i : i + 3] for i in range(len(packed) - 2)} or {packed}


# ---------------------------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------------------------


class OntologyIndex:
    """Four lexical indices over one fixed ontology, plus a fuzzy fallback.

    Built once per ontology fingerprint. The definition index is *not* built here, it needs an
    encoder and 18k forward passes, and is attached separately by
    :meth:`attach_definition_index` so that every lexical path stays usable on a login node with no
    GPU and no sentence-transformer checkout.
    """

    def __init__(
        self,
        tree: HPOTree,
        *,
        abbreviations_path: str | Path | None = DEFAULT_ABBREV_PATH,
        restrict_to_phenotypic_abnormality: bool = True,
    ) -> None:
        self.tree = tree
        self.scorable = (
            set(tree.phenotypic_abnormality)
            if restrict_to_phenotypic_abnormality
            else set(tree.data)
        )
        self.exact: dict[str, set[str]] = defaultdict(set)
        self.synonym: dict[str, set[str]] = defaultdict(set)
        self.abbreviation: dict[str, set[str]] = defaultdict(set)
        self._trigram: dict[str, set[str]] = defaultdict(set)
        self._definition_vectors = None
        self._definition_ids: list[str] = []
        self._encoder = None
        self.expansions: dict[str, list[str]] = {}
        self.token_expansions: dict[str, list[str]] = {}
        self.n_abbreviation_rows = 0

        self._build_lexical()
        if abbreviations_path is not None:
            self._build_abbreviations(Path(abbreviations_path))

    # -- construction ---------------------------------------------------------------------

    def _index_key(self, key: str, hpo_id: str, table: dict[str, set[str]]) -> None:
        if not key:
            return
        table[key].add(hpo_id)
        for tri in _trigrams(key):
            self._trigram[tri].add(key)

    def _build_lexical(self) -> None:
        for hpo_id in self.scorable:
            node = HPO_class(self.tree.data[hpo_id])
            for name in node.name:
                self._index_key(normalise_phrase(name), hpo_id, self.exact)
            for phrase in list(node.name) + list(node.synonym):
                self._index_key(normalise_phrase(phrase), hpo_id, self.synonym)
        logger.info(
            "ontology index: %d exact keys, %d synonym keys, %d trigrams",
            len(self.exact), len(self.synonym), len(self._trigram),
        )

    def _build_abbreviations(self, path: Path) -> None:
        """Load the curated expansion table. A missing file is a warning, not a crash.

        Stage C degrades to three indices without it, and the run reports that it did, which is
        preferable to a hard failure on a login node, and far preferable to inventing expansions.
        """
        if not path.exists():
            logger.warning(
                "abbreviation table %s not found — Stage C runs without it. "
                "Spans written as C3, 5MTHF, tHcy or MCV will not retrieve.", path,
            )
            return
        for row in read_reference_csv(path):
            surface = (row.get("surface") or "").strip()
            expansion = (row.get("expansion") or "").strip()
            if not surface or not expansion:
                continue
            self.n_abbreviation_rows += 1
            self.expansions.setdefault(normalise_phrase(surface), []).append(expansion)
            # A single-token surface can also be substituted *inside* a longer span:
            # "plasmatic homocysteine" -> "plasma homocysteine".
            tokens = processStr(surface)
            if len(tokens) == 1:
                self.token_expansions.setdefault(tokens[0], []).append(expansion)
            # Where the expansion happens to be an ontology phrase itself, pre-resolve it too.
            for hpo_id in self.synonym.get(normalise_phrase(expansion), set()):
                self.abbreviation[normalise_phrase(surface)].add(hpo_id)
        logger.info(
            "abbreviation table: %d rows, %d surface forms, %d directly resolvable to a term",
            self.n_abbreviation_rows, len(self.expansions), len(self.abbreviation),
        )

    def expand(self, text: str) -> list[str]:
        """Alternative query strings for a span, from the curated table.

        Two mechanisms, because the table holds two kinds of entry. A **whole-span** match rewrites
        the span outright (``big erythrocytes`` -> ``macrocytosis``). A **token** match substitutes
        one word inside it (``plasmatic homocysteine`` -> ``plasma homocysteine``), which is how the
        Gallicisms and the British/American orthography pairs earn their place.

        Essentially, an expansion is a *query*, not an answer. Most expansions here are analyte names, ``cobalamin``, ``methylmalonic acid``, and HPO has no term with those labels. It has
        *Decreased circulating cobalamin concentration* and *Methylmalonic aciduria*. So the
        expanded string is fed back through the whole retrieval stack, not looked up once,
        and it is the definition and fuzzy indices that finish the job.
        """
        out: list[str] = list(self.expansions.get(normalise_phrase(text), []))
        tokens = processStr(text)
        if len(tokens) > 1:
            for i, token in enumerate(tokens):
                for replacement in self.token_expansions.get(token, []):
                    out.append(" ".join(tokens[:i] + [replacement] + tokens[i + 1 :]))
        return out

    def attach_definition_index(self, ids: list[str], vectors, encoder=None) -> None:
        """Attach pre-computed definition embeddings and the encoder that made them.

        ``vectors`` must be row-normalised, so a query dotted against them is a cosine. The encoder
        has to be the *same* one the vectors were built with, a query embedded by a different
        model lands in a different space and the cosines are meaningless while still being numbers
        in [-1, 1]. :func:`build_definition_index` records the encoder name in the cache for this
        reason and :func:`load_definition_index` refuses a mismatch.
        """
        self._definition_ids = list(ids)
        self._definition_vectors = vectors
        if encoder is not None:
            self._encoder = encoder

    # -- lookup ---------------------------------------------------------------------------

    def lexical(self, text: str) -> dict[str, set[str]]:
        """The dictionary tiers alone: ``{hpo_id: {source, ...}}``, no similarity of any kind.

        Public counterpart of :meth:`_lexical_hits`, for callers that want the cheap half of
        :meth:`candidates` without paying for it. The distinction is not cosmetic, these are three
        dict lookups, while ``candidates`` additionally runs ``partial`` and ``fuzzy``, which cost
        roughly **1 s per prose-length query** (``SequenceMatcher`` over trigram-blocked keys). For
        a linker resolving thousands of generated lines that is the difference between a 40-second
        job and a two-hour one (``hpo_extraction.baselines.autopcr_link``).

        Note what it buys over a raw surface lookup: ``normalise_phrase`` strips punctuation and
        ``lemma_key`` lemmatises, so *Basal cell carcinomas* reaches *Basal cell carcinoma*, which
        ``surface_index.surface_key`` (case and whitespace only) cannot.
        """
        return self._lexical_hits(text)

    def _lexical_hits(self, text: str) -> dict[str, set[str]]:
        """``{hpo_id: {source, ...}}`` for the exact, synonym and abbreviation indices."""
        hits: dict[str, set[str]] = defaultdict(set)
        for key in {normalise_phrase(text), lemma_key(text)}:
            if not key:
                continue
            for source, table in (
                ("exact", self.exact),
                ("synonym", self.synonym),
                ("abbreviation", self.abbreviation),
            ):
                for hpo_id in table.get(key, ()):
                    hits[hpo_id].add(source)
        return hits

    def partial(self, text: str, *, max_extra_tokens: int = 2, limit: int = 20) -> dict[str, float]:
        """Terms whose label *contains* every token of the span, as a token subset.

        Motivated by a real miss on GSC+: the corpus annotates ``HP:0100264`` for the mention
        "symphalangism", but this ontology carries no bare *Symphalangism* term, only *Proximal
        symphalangism* and *Cushing's symphalangism*. Exact and synonym lookup both fail, and edit
        similarity does not rescue it either ("symphalangism" against "proximal symphalangism" is
        ~0.72, below any threshold that is safe for the rest of the vocabulary).

        This is not an edge case but a **systematic** property of clinical writing: the letter names
        the finding, the ontology names a qualified version of it. Retrieving the qualified term is
        the right behaviour, deciding whether the text actually supports that extra qualifier is
        Stage D's specificity rule, and it can only make that decision about candidates it is shown.

        ``max_extra_tokens`` is what keeps this from degenerating: without it, a one-token span like
        "nails" would retrieve every term containing the word. The polarity guard still applies, so
        a subset match cannot cross ``macro``/``micro``.
        """
        tokens = set(processStr(text))
        if not tokens:
            return {}
        out: dict[str, float] = {}
        for key, ids in self.synonym.items():
            key_tokens = key.split()
            if len(key_tokens) <= len(tokens) or len(key_tokens) - len(tokens) > max_extra_tokens:
                continue
            if not tokens.issubset(key_tokens):
                continue
            if crosses_polarity(" ".join(sorted(tokens)), key):
                continue
            score = len(tokens) / len(key_tokens)
            for hpo_id in ids:
                out[hpo_id] = max(out.get(hpo_id, 0.0), score)
        return dict(sorted(out.items(), key=lambda kv: -kv[1])[:limit])

    def fuzzy(self, text: str, *, threshold: float = 0.82, limit: int = 20) -> dict[str, float]:
        """Trigram-blocked fuzzy match over the synonym keys.

        Blocking on shared character trigrams before scoring keeps this linear in the *candidate*
        keys, not in the 18 000-term vocabulary; ``homocvsteinaemia`` and *homocysteinemia*
        share plenty of trigrams even though no token of either is equal.
        """
        key = normalise_phrase(text)
        if not key:
            return {}
        blocked: set[str] = set()
        for tri in _trigrams(key):
            blocked |= self._trigram.get(tri, set())
        scored: list[tuple[float, str]] = []
        for cand_key in blocked:
            if crosses_polarity(key, cand_key):
                continue
            ratio = SequenceMatcher(None, key, cand_key).ratio()
            if ratio >= threshold:
                scored.append((ratio, cand_key))
        scored.sort(reverse=True)

        out: dict[str, float] = {}
        for ratio, cand_key in scored[:limit]:
            for hpo_id in self.synonym.get(cand_key, ()):
                out[hpo_id] = max(out.get(hpo_id, 0.0), ratio)
        return out

    def candidates(
        self,
        text: str,
        *,
        sentence: str = "",
        k: int = 15,
        expansions: list[str] | None = None,
        fuzzy_threshold: float = 0.82,
    ) -> list[Candidate]:
        """Assemble up to *k* candidates for one span.

        Order of operations follows the specification: query all indices with the span. Re-query
        with any abbreviation-expanded form. Query the definition index with the span *and* its
        sentence. Then the fuzzy pass. Sources accumulate on the candidate, not replacing
        each other, so "matched the label *and* the definition" is visible to Stage D.
        """
        scored: dict[str, tuple[set[str], float]] = {}

        def add(hpo_id: str, source: str, score: float) -> None:
            sources, best = scored.get(hpo_id, (set(), 0.0))
            sources.add(source)
            scored[hpo_id] = (sources, max(best, score))

        for hpo_id, sources in self._lexical_hits(text).items():
            for source in sources:
                add(hpo_id, source, 1.0)

        expanded = list(expansions or []) + self.expand(text)
        for expansion in expanded:
            for hpo_id, sources in self._lexical_hits(expansion).items():
                for source in sources:
                    add(hpo_id, source, 0.98)
            for hpo_id, ratio in self.fuzzy(expansion, threshold=fuzzy_threshold).items():
                add(hpo_id, "abbreviation", 0.95 * ratio)

        for hpo_id, score in self.partial(text).items():
            add(hpo_id, "partial", 0.90 * score)

        for hpo_id, ratio in self.fuzzy(text, threshold=fuzzy_threshold).items():
            add(hpo_id, "fuzzy", ratio)

        for hpo_id, sim in self._definition_hits(
            " ".join([text] + expanded[:2]), sentence, k=k
        ).items():
            add(hpo_id, "definition", sim)

        ranked = sorted(
            scored.items(),
            key=lambda kv: (-kv[1][1], min(SOURCES.index(s) for s in kv[1][0]), kv[0]),
        )[:k]
        return [self._materialise(hpo_id, sources, score) for hpo_id, (sources, score) in ranked]

    def _definition_hits(
        self, text: str, sentence: str, *, k: int, floor: float = 0.45
    ) -> dict[str, float]:
        """Nearest definitions by cosine, or nothing when no definition index is attached.

        The ``floor`` counts more than the ``k``: this index is dense, so it always returns *some*
        nearest neighbour, and an unfloored dense retriever quietly fills every candidate list with
        its k best guesses no matter how bad they are. Stage D would then never see an empty list,
        and "zero candidates for a finding", an escalation trigger, would stop firing.
        """
        if self._definition_vectors is None or self._encoder is None:
            return {}
        import numpy as np

        query = self._encoder(f"{text}. {sentence}".strip())
        if query is None:
            return {}
        query = np.asarray(query, dtype="float32").ravel()
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return {}
        sims = self._definition_vectors @ (query / norm)
        top = np.argsort(-sims)[:k]
        return {
            self._definition_ids[i]: float(sims[i]) for i in top if float(sims[i]) >= floor
        }

    def _label_of(self, hpo_id: str) -> str:
        """A term's primary label, degrading to the id when this release does not carry it."""
        node = self.tree.data.get(hpo_id)
        if not node:
            return hpo_id
        names = node.get("Name") or []
        return names[0] if names else hpo_id

    def _materialise(self, hpo_id: str, sources: set[str], score: float) -> Candidate:
        node = HPO_class(self.tree.data[hpo_id])
        return Candidate(
            hpo_id=hpo_id,
            label=node.name[0] if node.name else hpo_id,
            synonyms=list(node.synonym),
            definition=node.definition[0] if node.definition else "",
            direct_parents=sorted(node.is_a),
            direct_parent_labels=[self._label_of(pid) for pid in sorted(node.is_a)],
            direct_children=sorted(node.son),
            ancestors=sorted(node.father),
            depth=self.tree.depth_dict.get(hpo_id) if hasattr(self.tree, "depth_dict") else None,
            sources=sorted(sources, key=SOURCES.index),
            score=score,
            obsolete_match=hpo_id not in self.scorable,
        )


@functools.lru_cache(maxsize=2)
def load_index(hpo_json_path: str | None = None) -> OntologyIndex:
    """Build (and cache) the lexical index for one ontology file."""
    tree = HPOTree(hpo_json_path) if hpo_json_path else HPOTree()
    tree.buildHPOTree()
    return OntologyIndex(tree)


# ---------------------------------------------------------------------------------------------
# The definition index
# ---------------------------------------------------------------------------------------------


def definition_corpus(tree: HPOTree, scorable: set[str]) -> tuple[list[str], list[str]]:
    """``(ids, texts)`` for every term carrying a ``def:``.

    The text embedded is ``label. definition``, not the definition alone. The label is the
    single most discriminative string a term has, and dropping it makes near-identical definitions
    ("An abnormally increased ...") collide across unrelated organ systems.
    """
    ids: list[str] = []
    texts: list[str] = []
    for hpo_id in sorted(scorable):
        node = HPO_class(tree.data[hpo_id])
        definition = node.definition[0] if node.definition else ""
        if not definition:
            continue
        label = node.name[0] if node.name else hpo_id
        ids.append(hpo_id)
        texts.append(f"{label}. {definition}")
    return ids, texts


def build_definition_index(
    index: "OntologyIndex",
    encode_batch,
    *,
    encoder_name: str,
    cache_path: str | Path,
    fingerprint: str,
    batch_size: int = 256,
):
    """Embed every definition once and cache it, keyed by ontology fingerprint + encoder name.

    ``encode_batch(list[str]) -> ndarray`` is supplied by the caller so this module never imports a
    model. The cache is invalidated by *either* key changing: a new ontology means new definitions,
    and a new encoder means a different vector space.
    """
    import numpy as np

    cache_path = Path(cache_path)
    ids, texts = definition_corpus(index.tree, index.scorable)
    logger.info("embedding %d definitions with %s", len(ids), encoder_name)

    chunks = []
    for start in range(0, len(texts), batch_size):
        chunks.append(np.asarray(encode_batch(texts[start : start + batch_size]), dtype="float32"))
    vectors = np.vstack(chunks) if chunks else np.zeros((0, 1), dtype="float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, ids=np.array(ids), vectors=vectors,
        fingerprint=fingerprint, encoder=encoder_name,
    )
    logger.info("wrote %s (%d x %d)", cache_path, *vectors.shape)
    return ids, vectors


def load_definition_index(cache_path: str | Path, *, fingerprint: str, encoder_name: str):
    """Load a cached definition index, or ``None`` when it is absent or stale.

    Staleness is a refusal, not a warning that gets ignored: a cache built against another ontology
    would index terms by a position that no longer means the same thing, and one built by another
    encoder would return confident cosines from an unrelated vector space.
    """
    import numpy as np

    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as data:
        cached_fp = str(data["fingerprint"])
        cached_enc = str(data["encoder"])
        if cached_fp != fingerprint:
            logger.warning(
                "definition index %s was built against ontology %s, not %s — ignoring it",
                cache_path, cached_fp, fingerprint,
            )
            return None
        if cached_enc != encoder_name:
            logger.warning(
                "definition index %s was built with %s, not %s — ignoring it",
                cache_path, cached_enc, encoder_name,
            )
            return None
        return [str(i) for i in data["ids"]], data["vectors"]
