"""RAG-HPO **as published** — the pipeline that produced Garcia et al. 2025, Tables 4-5.

``rag_hpo_lib.py`` in this directory is the *current* upstream (commit of 2025-07-30, vendored
2026-08-05). It is not the code the paper reports on. Upstream's history:

    2024-12-19  6062c96  RAG-HPO.ipynb, system_prompts.json  ← preprint
    2025-01-02  d4cce83  RAG-HPO.ipynb
    2025-04-26  25c1ea7  "Updated RAG-HPO"                   ← the published tables
    2025-07-30  d2dd604  RAG-HPO.ipynb, system_prompts.json  ← what rag_hpo_lib.py is

This module is the ``25c1ea7`` notebook, exported the same way ``rag_hpo_lib.py`` exports the
newer one: retrieval / parsing / mapping logic unchanged, only the cloud-API client, the
interactive prompts and the pickle checkpointing removed. It pairs with the 2024-12-19
``system_prompts_paper.json`` — the prompts file did not change between the preprint and
2025-07-30, so those are the prompts the published runs used.

What differs from ``rag_hpo_lib.py``, and why it matters (every item suppresses recall):

  * **No category filter.** The published ``system_message_I`` returns a flat
    ``{"findings": [...]}`` list of phrases. The new one classifies each phrase as Abnormal /
    Normal / Family History / Other / Suspected and ``process_row`` keeps only *Abnormal*.
  * **No null escape hatch.** The published mapping prompt always names a term; the new
    ``system_message_II`` is instructed to return ``null`` when nothing fits well
    (→ "No Candidate Fit").
  * **Every ``HP:\\d+`` in the reply is kept**, so one phrase can yield several terms. The new
    code parses a single ``hpo_id`` and validates it against the candidate list.
  * **20 candidates** per phrase (their ``if len(unique_metadata) == 20: break``), against 15.
  * **BGE-small + ``IndexFlatL2``** over a FAISS depth of 800, against SapBERT + cosine/IP over 500.

**Candidate rows are single-key ``{<phrase>: 'HP:…'}`` dicts**, which is the shape every consumer
here is written for: :func:`generate_hpo_terms` iterates ``.items()`` and renders the key as the
term name, :func:`extract_hpo_term` fuzzy-matches the phrase against the key, and
:func:`process_unique_metadata` lower-cases the key to normalise it. All three only make sense
under that shape, and upstream's current ``_iter_term_hp`` still carries an explicit fallback
branch for it (``for k, v in d.items(): if v.startswith("HP:")``) alongside its handling of the
newer ``{'info': …, 'hp_id': …}`` rows — that fallback exists precisely because the paper-era
``G2GHPO_metadata.npy`` stored the flat mapping.

Feeding the newer two-key shape to this code instead is not a faithful reproduction, it is a
format mismatch, and it degrades every stage at once: the context block renders ``- info (…)`` /
``- hp_id (HP:…)`` line pairs in which the term name the model is asked to choose between is the
literal string ``info``, and the local exact-match shortcut can never fire. :func:`load_vector_db`
therefore converts the on-disk rows to the paper-era shape; see :func:`candidate_pair` for reading
one back.

Deviations from the notebook, all forced and none behavioural:

  * ``fastembed``/ONNX → ``sentence-transformers`` for BGE-small (fastembed is not in the
    environment and the compute nodes are offline). Both L2-normalise BGE output, and with
    normalised vectors ``IndexFlatL2`` ranks identically to cosine, so the candidate lists match.
  * ``fuzzywuzzy.fuzz.ratio`` → ``rapidfuzz.fuzz.ratio`` (same metric, packaged in this env).
  * The vector DB is stored as ``hpo_meta_paper.json`` + ``hpo_embedded_paper.npz`` rather than
    their pickled ``G2GHPO_metadata.npy`` (which was never published); the *rows* are built to
    their recipe — see ``build_vector_db_paper.py``.
  * Non-string findings are skipped instead of raising: their notebook ran interactively with
    checkpoints, ours runs unattended on a cluster.

``llm_client`` is a module global, as in ``rag_hpo_lib``: inject something exposing
``query(user_input, system_message) -> str`` before calling :func:`process_row` /
:func:`generate_hpo_terms`. Orchestration lives in ``core.rag_hpo_paper_runner``.
"""

import json
import logging
import os
import re
import string
import sys
from typing import Any

import faiss
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

#: Set by ``core.rag_hpo_paper_runner`` to an object with ``.query(user_input, system_message)``.
llm_client = None

#: Their FAISS search depth, deduplicated down to ``KEEP_TOP`` unique candidates.
FAISS_DEPTH = 800
#: "the top 20 most semantically similar vectors" (paper) = ``if len(unique_metadata) == 20``.
KEEP_TOP = 20
#: The embedding model of the published runs.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


# ======================= Embeddings & FAISS =======================

def initialize_embeddings_model(model_name: str = EMBED_MODEL):
    """Load BGE-small. ``model_name`` may be a local directory (offline compute nodes)."""
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # mirrors the notebook's hard exit on a missing model
        logger.error("[FATAL] Could not initialize embedding model %s: %s", model_name, exc)
        sys.exit(1)


def load_vector_db(meta_path: str, vec_path: str) -> tuple[list[dict[str, str]], np.ndarray]:
    """Return ``(docs, emb_matrix)`` for the paper-era DB written by ``build_vector_db_paper.py``.

    Each doc is one ``{<phrase>: 'HP:…'}`` entry — the ``unique_metadata`` payload of their
    ``G2GHPO_metadata.npy``. The file stores ``hp_id``/``info`` as separate JSON fields because
    that is the convenient layout for building and inspecting it; the conversion happens here, so
    that every consumer in this module sees the shape it was written for (module docstring).
    """
    for path in (meta_path, vec_path):
        if not os.path.exists(path):
            logger.error("[FATAL] vector DB file not found: %s — run build_vector_db_paper.py", path)
            sys.exit(1)

    with open(meta_path, encoding="utf-8") as f:
        entries = json.load(f)["entries"]
    emb_matrix = np.load(vec_path)["emb"].astype(np.float32)

    if len(entries) != emb_matrix.shape[0]:
        logger.warning("[WARN] metadata/embedding row mismatch (%d vs %d)",
                       len(entries), emb_matrix.shape[0])

    docs = [{e["info"]: e["hp_id"]} for e in entries]
    return docs, emb_matrix


def candidate_pair(item) -> tuple[str, str] | None:
    """``(phrase, hp_id)`` for one retrieved candidate, or ``None`` when it is not one.

    Mirrors upstream's ``_iter_term_hp``: paper-era rows are ``{<phrase>: 'HP:…'}``, and the newer
    ``{'info': …, 'hp_id': …}`` shape is accepted too so that anything reading a stored artifact
    keeps working. Used by the artifact writers in ``core.rag_hpo_paper_runner``; the pipeline
    itself iterates the dicts directly, exactly as the notebook does.
    """
    parsed = clean_and_parse(item) if isinstance(item, str) else item
    if not isinstance(parsed, dict):
        return None
    if "hp_id" in parsed:
        phrase, hp_id = parsed.get("info") or parsed.get("label"), parsed["hp_id"]
        return (phrase, hp_id) if phrase and hp_id else None
    for phrase, hp_id in parsed.items():
        if isinstance(hp_id, str) and hp_id.startswith("HP:"):
            return phrase, hp_id
    return None


def create_faiss_index(emb_matrix: np.ndarray) -> faiss.Index:
    """``IndexFlatL2``, as in the notebook. Vectors are BGE-normalised, so L2 ranks as cosine."""
    if emb_matrix.dtype != np.float32:
        emb_matrix = emb_matrix.astype(np.float32)
    index = faiss.IndexFlatL2(emb_matrix.shape[1])
    index.add(emb_matrix)
    return index


def embed_query(text: str, model) -> np.ndarray:
    """Embed one phrase, L2-normalised — fastembed normalises BGE output, so we do too."""
    vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32).reshape(1, -1)


# ======================= Extraction =======================

def extract_findings(response_content: str) -> list:
    """Their parser: strip code fences, take the outermost ``{...}``, read the ``findings`` list."""
    if not response_content:
        return []
    sanitized = response_content.replace("```", "").strip()
    start, end = sanitized.find("{"), sanitized.rfind("}")
    if start == -1 or end == -1 or start > end:
        return []
    try:
        data = json.loads(sanitized[start:end + 1])
    except json.JSONDecodeError:
        return []
    findings = data.get("findings", []) if isinstance(data, dict) else []
    # Diagnostic only — the return value is unchanged, so this stays faithful to upstream.
    #
    # The paper's own System Message I (Additional file 2, Fig. S1) instructs the model to return
    # "a JSON object with a single key 'finding'" — singular — while the worked example in the same
    # figure emits "findings". This parser, byte-faithful to upstream at 25c1ea7, reads only the
    # plural. So a reply that obeys the published *instruction* is silently discarded by the
    # published *code*, and the report contributes nothing. exp13_17 runs that prompt deliberately;
    # counting the drops is what tells us whether a low yield is the prompt or the parser.
    if isinstance(data, dict) and not findings and isinstance(data.get("finding"), list):
        logger.warning(
            "reply used the singular key 'finding' (%d item(s)) and was DISCARDED: upstream's "
            "parser reads only 'findings'. This is the paper's prompt disagreeing with the paper's "
            "code, not a bug here.", len(data["finding"]))
    return findings if isinstance(findings, list) else []


def _as_phrase(finding: Any) -> str | None:
    """Coerce one element of ``findings`` to a phrase.

    The notebook assumed a bare string and would raise on anything else. Unattended, a single
    malformed reply would then kill the run, so a dict is unwrapped and anything else dropped.
    """
    if isinstance(finding, str):
        return finding.strip() or None
    if isinstance(finding, dict):
        for key in ("phrase", "finding", "term", "text"):
            value = finding.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def process_findings(findings, clinical_note: str, embeddings_model, index,
                     docs: list[dict[str, str]], keep_top: int = KEEP_TOP,
                     faiss_depth: int = FAISS_DEPTH) -> pd.DataFrame:
    """Retrieve candidates per finding and attach the best-overlapping sentence.

    Dedup is on the whole ``{<phrase>: 'HP:…'}`` payload — *not* on the id — so the 20 candidates
    can contain several phrases of the same term. That is what the notebook does, and it changes
    the prompt: a term with many matching synonyms occupies several of the 20 slots.
    """
    sentences = clinical_note.split(".")
    results = []

    for finding in findings:
        phrase = _as_phrase(finding)
        if phrase is None:
            logger.warning("skipping non-string finding: %r", finding)
            continue

        query_vector = embed_query(phrase, embeddings_model)
        distances, indices = index.search(query_vector, faiss_depth)

        seen_metadata: set[str] = set()
        unique_metadata: list[str] = []
        similarities: list[float] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            metadata_str = json.dumps(docs[idx])
            if metadata_str in seen_metadata:
                continue
            seen_metadata.add(metadata_str)
            unique_metadata.append(metadata_str)
            # Both sides are unit vectors, so cos = 1 - d^2/2. Kept only for the segments
            # artifact, which reports a similarity for every method.
            similarities.append(1.0 - float(dist) / 2.0)
            if len(unique_metadata) == keep_top:
                break

        finding_words = set(re.findall(r"\b\w+\b", phrase.lower()))
        best_match_sentence, max_matching_words = None, 0
        for sentence in sentences:
            sentence_words = set(re.findall(r"\b\w+\b", sentence.lower()))
            common = len(finding_words & sentence_words)
            if common > max_matching_words:
                max_matching_words, best_match_sentence = common, sentence.strip()

        results.append({
            "phrase": phrase,
            "unique_metadata": unique_metadata,
            "similarities": similarities,
            "original_sentence": best_match_sentence,
        })

    return pd.DataFrame(results, columns=["phrase", "unique_metadata", "similarities",
                                          "original_sentence"])


def process_row(clinical_note: str, system_message_I: str, embeddings_model, index,
                docs: list[dict[str, str]], keep_top: int = KEEP_TOP,
                faiss_depth: int = FAISS_DEPTH) -> pd.DataFrame | None:
    """One clinical note → phrases (LLM) → candidates (FAISS). ``None`` when nothing is extracted.

    Note what is *absent*: the note is passed to the LLM as-is (the newer lib re-encodes it through
    latin1 first), and no phrase is filtered by category.
    """
    findings_text = llm_client.query(clinical_note, system_message_I)
    if not findings_text:
        return None
    findings = extract_findings(findings_text)
    if not findings:
        return None
    return process_findings(findings, clinical_note, embeddings_model, index, docs,
                            keep_top=keep_top, faiss_depth=faiss_depth)


# ======================= Metadata helpers =======================

def clean_and_parse(json_str: str):
    """Their lenient JSON reader for a metadata string."""
    try:
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r"\s+", " ", json_str)
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def process_unique_metadata(metadata) -> list[str]:
    """Lower-case every key of every metadata entry — i.e. normalise the candidate phrases."""
    if not isinstance(metadata, list):
        return []
    processed = []
    for item in metadata:
        try:
            item_dict = json.loads(item)
        except (json.JSONDecodeError, TypeError):
            continue
        processed.append(json.dumps({k.lower(): v for k, v in item_dict.items()}))
    return processed


def clean_text(text: str) -> str:
    """Lower-case, drop punctuation, trim — their retrieval-side normaliser."""
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()


def extract_hpo_term(phrase: str, metadata_list: list[str]) -> str | None:
    """Their local shortcut: fuzzy → substring → exact, returning an id only on exact equality.

    Reproduced verbatim, quirks included. Steps 1 and 2 append *copies* of entries already in the
    list rather than returning, so they cannot decide the outcome on their own — only step 3
    returns, and only on exact equality between the cleaned phrase and a cleaned candidate. What
    steps 1 and 2 do change is ``metadata_list``, in place: callers that read it afterwards see the
    appended copies as extra entries (see ``core.rag_hpo_paper_runner``).
    """
    cleaned_phrase = clean_text(phrase)
    fuzzy_matches = []

    # Step 1: fuzzy matching with metadata
    for metadata in metadata_list:
        try:
            metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
            for term, hp_id in metadata_dict.items():
                if fuzz.ratio(cleaned_phrase, clean_text(term)) > 80:
                    fuzzy_matches.append({term: hp_id})
        except (json.JSONDecodeError, TypeError):
            continue
    if fuzzy_matches:
        metadata_list.extend([json.dumps(match) for match in fuzzy_matches])

    # Step 2: exact substring matching
    exact_matches = []
    for metadata in metadata_list:
        try:
            metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
            for term, hp_id in metadata_dict.items():
                if clean_text(term) in cleaned_phrase:
                    exact_matches.append({term: hp_id})
        except (json.JSONDecodeError, TypeError):
            continue
    if exact_matches:
        metadata_list.extend([json.dumps(match) for match in exact_matches])

    # Step 3: exact match within the metadata list
    for metadata in metadata_list:
        if not metadata.strip():
            continue
        try:
            metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
            for term, hp_id in metadata_dict.items():
                if cleaned_phrase == clean_text(term):
                    return hp_id
        except (json.JSONDecodeError, TypeError):
            continue

    return None


# ======================= Mapping =======================

def generate_hpo_terms(df: pd.DataFrame, system_message: str) -> pd.DataFrame:
    """Ask the LLM to name the HPO term(s) for each phrase, given the retrieved candidates.

    The context block is built exactly as upstream builds it — one ``- <phrase> (HP:…)`` line per
    candidate — and **every** ``HP:\\d+`` in the reply is kept, comma-joined. There is no
    candidate-membership check and no null option: those arrived with the 2025-07-30 rewrite, and
    both cost recall.
    """
    responses = []
    for _, row in df.iterrows():
        user_input = row["phrase"]
        unique_metadata_list = row["unique_metadata"]
        original_sentence = row["original_sentence"]

        context_items = []
        for item in unique_metadata_list:
            parsed_item = clean_and_parse(item)
            if parsed_item:
                for description, hp_id in parsed_item.items():
                    context_items.append(f"- {description} ({hp_id})")
        context_text = "\n".join(context_items)

        human_message = (
            f"Query: {user_input}\n"
            f"Original Sentence: {original_sentence}\n"
            f"Context: The following related information is available to assist in determining "
            f"the appropriate HPO terms:\n"
            f"{context_text}"
        )

        response_text = llm_client.query(human_message, system_message)
        hpo_terms = re.findall(r"HP:\d+", response_text or "")
        responses.append({
            "phrase": user_input,
            "response": ", ".join(hpo_terms) if hpo_terms else "No HPO terms found",
            "raw_llm_resp": response_text,
        })

    return pd.DataFrame(responses)
