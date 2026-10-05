# -*- coding: utf-8 -*-
"""
HPO Vector DB Builder via Pronto Graph Extraction
(exported verbatim from HPO_Vectorization.ipynb, cell 0).

One-time build of the RAG-HPO vector database. Downloads the latest hp.obo, builds
per-term metadata (labels, synonyms, lineage, organ system, alt/xref ids), embeds every
label+synonym with SapBERT, and writes:
    hpo_terms_full.csv   — human-readable term table
    hpo_meta.json        — {constants, entries} metadata consumed by load_vector_db()
    hpo_embedded.npz      — float16 embedding matrix ('emb')

Run once (needs `pronto`; downloads hp.obo + the SapBERT model, ~10 min):
    conda activate PhenoRAG_env && pip install pronto
    python src/RAG-HPO/build_vector_db.py

The only changes vs the notebook cell are: `fastembed` import made optional (the SapBERT
path does not need it), and `__main__` chdir's to this file's directory + calls main() so
outputs always land in src/RAG-HPO/ regardless of the current working directory.

Features:
- Automatic OBO download & refresh (`initialize_hpo_resources`)
- Build metadata directly from the HPO OBO using pronto:
  • labels, definitions, synonyms, ALT IDs, xrefs
  • full lineage (root→term) across all inheritance paths
  • organ system(s) (direct child under phenotypic abnormality)
- Optional SBERT (BioBERT) or BGE embeddings
- Antonym `direction` flags
- Float16 quantization & compressed storage (.npz)
- Optional limit for test extraction
"""
import os
import time
import json
import re
import requests
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import pronto      # pip install pronto
try:
    from fastembed import TextEmbedding  # only needed for the use_sbert=False (BGE) path
except ImportError:
    TextEmbedding = None
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# When run as a script, pin all relative outputs (hp.obo, hpo_meta.json,
# hpo_embedded.npz, hpo_terms_full.csv) to this file's directory so the DB always
# builds inside src/RAG-HPO/. Done before the module-level ontology load below so the
# hp.obo download lands there too.
if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)

# ────────── 0. Automatic OBO Download & Load ──────────
_hpo_ontology = None

def initialize_hpo_resources(
    obo_url: str = "https://purl.obolibrary.org/obo/hp.obo",
    obo_path: str = "hp.obo",
    refresh_days: int = 14,
) -> pronto.Ontology:
    """
    Download or refresh the HPO OBO file if older than `refresh_days`,
    then load it via pronto.
    """
    global _hpo_ontology
    if _hpo_ontology is None:
        obo_file = Path(obo_path)
        if not obo_file.exists() or ((time.time() - obo_file.stat().st_mtime) / 86400) > refresh_days:
            print(f"Downloading HPO ontology from {obo_url} …")
            resp = requests.get(obo_url, timeout=30)
            resp.raise_for_status()
            obo_file.write_text(resp.text, encoding="utf-8")
        with open(obo_file, 'rb') as f:
            _hpo_ontology = pronto.Ontology(f)
        print(f"Loaded ontology with {len(list(_hpo_ontology.terms()))} terms")
    return _hpo_ontology

ROOT_ID  = 'HP:0000001'  # "All"
PHENO_ID = 'HP:0000118'  # "Phenotypic abnormality"

# load & cache ontology once
ont = initialize_hpo_resources()

# maps HP_ID → [immediate parent IDs]
parent_map = {
    term.id: [p.id for p in term.superclasses(distance=1)]
    for term in ont.terms()
}

# maps HP_ID → term name
label_map = {term.id: term.name for term in ont.terms()}

# ────────── 1. Multi-Path Lineage Helper ──────────
_lineage_memo = {}

def _build_lineage_paths(hp_id, parent_map, seen=None):
    """
    Return all paths from ROOT_ID → ... → hp_id.
    Each path is a list of HP_ID strings.
    Avoids cycles by tracking `seen`.
    """
    if seen is None:
        seen = set()
    if hp_id in seen:              # cycle guard
        return []
    seen = seen | {hp_id}

    # cached?
    if hp_id in _lineage_memo:
        return _lineage_memo[hp_id]

    # base case: we hit the root
    if hp_id == ROOT_ID:
        paths = [[ROOT_ID]]
    else:
        parents = parent_map.get(hp_id, [])
        if not parents:
            # orphan: just attach root + self
            paths = [[ROOT_ID, hp_id]]
        else:
            paths = []
            for p in parents:
                for ppath in _build_lineage_paths(p, parent_map, seen):
                    paths.append(ppath + [hp_id])

    _lineage_memo[hp_id] = paths
    return paths

# ────────── 2. Build DataFrame from OBO ──────────
CLEAN_ABNORMALITY = re.compile(r'(?i)^Abnormality of(?: the)?\s*')

def _sort_by_numeric(entries: list[str]) -> list[str]:
    def key_fn(e: str):
        digs = ''.join(filter(str.isdigit, e))
        return int(digs) if digs else float('inf')
    return sorted(entries, key=key_fn)

def build_hpo_dataframe(obo_path: str = "hp.obo", limit: int = None) -> pd.DataFrame:
    ont     = pronto.Ontology(Path(obo_path))
    records = []
    terms   = list(ont.terms())[:limit] if limit else list(ont.terms())

    for term in tqdm(terms, desc="Building HPO DataFrame", unit="term"):
        hp_id      = term.id
        label      = term.name
        definition = term.definition or ""
        synonyms   = [syn.description for syn in term.synonyms]

        # ── ALT IDs ──
        alt_ids = _sort_by_numeric(list(term.alternate_ids))

        # ── XREFS ──
        snomedct, umls = [], []
        for xr in term.xrefs:
            txt = str(xr)
            m   = re.search(r"'(.+?:.+?)'", txt)
            ent = m.group(1) if m else txt
            pre, _, _ = ent.partition(':')
            if pre.upper() == 'UMLS':
                umls.append(ent)
            elif pre.upper().startswith('SNOMED'):
                snomedct.append(ent)
        snomedct = _sort_by_numeric(snomedct)
        umls      = _sort_by_numeric(umls)

        # ── LINEAGE ──
        paths = _build_lineage_paths(hp_id, parent_map) or [[ROOT_ID, hp_id]]
        for path_ids in paths:
            lineage_str = " -> ".join(f"{label_map[i]} ({i})" for i in path_ids)

            # ── ORGAN SYSTEM ──
            if PHENO_ID in path_ids:
                idx   = path_ids.index(PHENO_ID)
                organ = label_map.get(path_ids[idx+1], "Other") if idx+1 < len(path_ids) else "Other"
            else:
                organ = "Other"
            organ_system = CLEAN_ABNORMALITY.sub("", organ).title()

            # ── EMIT ROWS ──
            for phrase in [label] + synonyms:
                if not phrase:
                    continue
                # **Only title-case** the phrase; do NOT strip anything from it
                clean_phrase = phrase.title()

                records.append({
                    "hp_id":        hp_id,
                    "phrase":       clean_phrase,
                    "organ_system": organ_system,
                    "lineage":      lineage_str,
                    "definition":   definition,
                    "alt_ids":      ";".join(alt_ids),
                    "snomedct":     ";".join(snomedct),
                    "umls":         ";".join(umls),
                })

    return pd.DataFrame(
        records,
        columns=[
            "hp_id", "phrase", "organ_system", "lineage",
            "definition", "alt_ids", "snomedct", "umls"
        ]
    )
# ────────── 3. Clean Text for Embedding ──────────
PAT = re.compile(r'\s*\([^)]*\)\s*')

def clean_text(txt: str) -> str:
    txt = PAT.sub(' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip().lower()
    txt = re.sub(r'[^\w\s]+$', '', txt)
    return txt

# ────────── 4. Embedding Model Selector ──────────
# SBERT_MODEL env var overrides the default (set it to a local directory on an offline
# cluster; must be the SAME model the experiments load via sbert_model).
DEFAULT_SBERT = os.environ.get(
    "SBERT_MODEL", "pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb"
)

def get_embedding_model(
    use_sbert: bool = True,
    sbert_model: str = DEFAULT_SBERT,

    bge_model: str = 'BAAI/bge-small-en-v1.5',
):
    if use_sbert:
        print(f"Loading SBERT model: {sbert_model}")
        return SentenceTransformer(sbert_model)
    print(f"Loading BGE model: {bge_model}")
    return TextEmbedding(model_name=bge_model)

# ────────── 5. Vectorize & Save ──────────
def vectorize_dataframe(
    df: pd.DataFrame,
    meta_out: str,
    vec_out: str,
    use_sbert: bool = True
):
    model = get_embedding_model(use_sbert)

    # ——— New: collect constant metadata once per HP term ———
    constants: dict[str, dict] = {}

    # ——— New: per-embedding metadata (minimal) ———
    entries: list[dict] = []
    embs: list[np.ndarray] = []

    # ——— Compile direction‐detection regexes once per call ———
    NEG_PATTERN = re.compile(
        r'\b(?:decreas(?:e|ed|ing)?|loss(?:es)?|hypo[-]?\w+)\b',
        re.IGNORECASE
    )
    POS_PATTERN = re.compile(
        r'\b(?:increas(?:e|ed|ing)?|gain(?:s|ed)?|hyper[-]?\w+)\b',
        re.IGNORECASE
    )

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding rows", unit="row"):
        info = clean_text(row.phrase)

        # ——— More robust direction detection ———
        direction = 0
        if NEG_PATTERN.search(info):
            direction = -1
        elif POS_PATTERN.search(info):
            direction = 1

        # ——— Embedding call unchanged ———
        if use_sbert:
            vec = model.encode(info, convert_to_numpy=True)
        else:
            vec = np.asarray(list(model.embed([info]))[0], dtype=np.float32)

        # ——— New: store only minimal per-info metadata ———
        entries.append({
            'hp_id':    row.hp_id,
            'info':     info,
            'direction':direction
        })

        # ——— New: record the constant fields once for each hp_id ———
        if row.hp_id not in constants:
            constants[row.hp_id] = {
                'organ_system': row.organ_system,
                'lineage':      row.lineage,
                'definition':   row.definition,
                'alt_ids':      row.alt_ids,
                'snomedct':     row.snomedct,
                'umls':         row.umls
            }

        embs.append(vec.astype(np.float16))

    # ——— Save embeddings as before ———
    emb_matrix = np.vstack(embs)

    # ——— Write out a single JSON with both parts ———
    combined = {
        'constants': constants,
        'entries':   entries
    }
    with open(meta_out, 'w') as f:
        json.dump(combined, f, separators=(',', ':'))

    np.savez_compressed(vec_out, emb=emb_matrix)
    print(f"Saved {len(entries)} embeddings → {meta_out}, {vec_out}")

# ────────── 6. Main Entrypoint ──────────
def main():
    df = build_hpo_dataframe("hp.obo")
    df.to_csv("hpo_terms_full.csv", index=False)
    print(f"Built DataFrame with {len(df)} rows.")
    vectorize_dataframe(
        df,
        meta_out='hpo_meta.json',
        vec_out='hpo_embedded.npz',
        use_sbert=True
    )

def main_test(
    test_limit: int = None
):
    df = build_hpo_dataframe("hp.obo", limit=test_limit)
    out_csv = "hpo_terms_full.csv" if test_limit is None else f"hpo_terms_test_{test_limit}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Built DataFrame with {len(df)} rows → {out_csv}")
    if test_limit is None:
        vectorize_dataframe(
            df,
            meta_out='hpo_meta.json',
            vec_out='hpo_embedded.npz',
            use_sbert=True
        )
    else:
        print("Test run: skipping vectorization.")

if __name__ == '__main__':
    main()
