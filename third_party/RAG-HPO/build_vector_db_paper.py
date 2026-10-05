# -*- coding: utf-8 -*-
"""Build RAG-HPO's vector DB **as it was for the paper** (HPO v2024-08-13 + BGE-small).

``build_vector_db.py`` builds the *current* upstream DB: SapBERT over the latest ``hp.obo``,
labels and synonyms only. That is not what Garcia et al. 2025 ran on. This script is their
``HPO_Vectorization.ipynb`` at commit ``25c1ea7`` (the revision behind the published tables):

  * source is **``hp.json``**, pinned to the release the paper names — "an updated HPO version
    accessed in August 2024 (v2024-08-13)";
  * every term contributes its **label, each synonym, and its definition** as separate rows —
    which is how the paper reaches ">54,000 unique phrases", far more than labels+synonyms alone;
  * **``HPO_addons.csv``** (3 079 curated phrase→HPO rows, shipped in this directory) is merged in;
  * each phrase is cleaned with their ``clean_text`` — drop parenthesised spans, ``_``→space,
    lower-case, *no* trailing strip — deduplicated per term, and embedded with
    **``BAAI/bge-small-en-v1.5``**.

Their pickled ``G2GHPO_metadata.npy`` was never published, so it has to be rebuilt; only the
storage layout differs here (a JSON + npz pair, like ``build_vector_db.py`` writes, instead of a
pickled object array). Rows, cleaning, and embedding model are theirs. Their per-row ``lineage`` /
``organ_system`` / ``depth_from_root`` fields are dropped: nothing in the retrieval or prompting
path ever reads them.

``fastembed`` is replaced by ``sentence-transformers`` (not in this environment, and the compute
nodes are offline). Both L2-normalise BGE output; ``rag_hpo_lib_paper`` normalises the query the
same way, so the FAISS ``IndexFlatL2`` ranking is unchanged.

    conda activate PhenoRAG_marc_env
    python src/RAG-HPO/build_vector_db_paper.py                    # src/RAG-HPO/hp.json + HF model
    python src/RAG-HPO/build_vector_db_paper.py \
        --hpo-json /path/to/hp.json --model /path/to/bge-small-en-v1.5   # offline

The release of ``--hpo-json`` is checked against v2024-08-13 and the build refuses to run on
anything else: a DB built from a different release still produces plausible numbers, and nothing
downstream would catch it.

Writes ``hpo_meta_paper.json`` and ``hpo_embedded_paper.npz`` next to this file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent

#: The release the paper names. Pinned, not "latest": a newer ontology is a different experiment.
#: Stored as the bare date, which is what the ontology's own version IRI carries
#: (``…/hp/releases/2024-08-13/hp.json``) and therefore what :func:`detect_release` returns; the
#: git tag prefixes it with ``v``, so only the URL adds that back.
HPO_RELEASE = "2024-08-13"
HPO_JSON_URL = (
    "https://github.com/obophenotype/human-phenotype-ontology/releases/download/"
    f"v{HPO_RELEASE}/hp.json"
)
EMBED_MODEL = os.environ.get("RAGHPO_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

#: Their cleaner: parenthesised spans removed, underscores to spaces, lower-cased. No strip —
#: keeping that faithful, since it decides whether two phrases deduplicate to one row.
_PAREN = re.compile(r"\(.*?\)")


def clean_text(text: str) -> str:
    return _PAREN.sub("", text).replace("_", " ").lower()


def fetch_hpo_json(path: Path) -> Path:
    """Download the pinned ``hp.json`` release unless it is already on disk."""
    if path.exists():
        return path
    print(f"Downloading HPO {HPO_RELEASE} from {HPO_JSON_URL} …")
    resp = requests.get(HPO_JSON_URL, timeout=120)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def load_hpo_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def detect_release(data: dict) -> str | None:
    """The release encoded in the ontology's own version IRI, e.g. ``2024-08-13``.

    Worth checking rather than trusting the filename: a DB silently built from a different HPO
    release is the one error in this pipeline that produces plausible numbers and invalidates the
    replication, and nothing downstream would ever notice.
    """
    for graph in data.get("graphs", []):
        version = (graph.get("meta") or {}).get("version", "")
        match = re.search(r"releases/(\d{4}-\d{2}-\d{2})", version)
        if match:
            return match.group(1)
    return None


def extract_terms(data: dict) -> dict[str, dict]:
    """``{HP_0001234: {label, definition, synonyms}}`` — their ``process_json_file``."""
    terms: dict[str, dict] = {}
    for graph in data.get("graphs", []):
        for node in graph.get("nodes", []):
            match = re.search(r"(HP_\d+)", node.get("id", ""))
            if not match:
                continue
            meta = node.get("meta", {}) or {}
            info = {
                "label": node.get("lbl", "") or "",
                "definition": (meta.get("definition") or {}).get("val", "") or "",
                "synonyms": [s.get("val", "") for s in meta.get("synonyms", []) if s.get("val")],
            }
            if info["label"] or info["definition"] or info["synonyms"]:
                terms[match.group(1)] = info
    return terms


def build_rows(terms: dict[str, dict], addons: pd.DataFrame) -> list[dict[str, str]]:
    """One row per (term, distinct cleaned phrase), label + synonyms + definition + add-ons."""
    rows: list[dict[str, str]] = []
    for hp_key, details in terms.items():
        hp_id = hp_key.replace("_", ":")
        unique_info: set[str] = set()

        if details["label"]:
            unique_info.add(clean_text(details["label"]))
        for synonym in details["synonyms"]:
            unique_info.add(clean_text(synonym))
        if details["definition"]:
            unique_info.add(clean_text(details["definition"]))
        for addon in addons.loc[addons["HP_ID"] == hp_id, "info"].tolist():
            unique_info.add(clean_text(str(addon)))

        # Sorted so a rebuild is byte-identical; upstream iterated a set, whose order only
        # affects tie-breaking between two identical-distance rows of the same term.
        for info in sorted(unique_info):
            rows.append({"hp_id": hp_id, "info": info})
    return rows


def embed_rows(rows: list[dict[str, str]], model_name: str, batch_size: int) -> np.ndarray:
    """Embed every phrase with BGE-small, L2-normalised (what fastembed returns for BGE)."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    texts = [r["info"] for r in rows]
    vecs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        vecs.append(model.encode(
            texts[start:start + batch_size],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ))
    return np.vstack(vecs).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hpo-json", default=str(_HERE / "hp.json"),
                        help="hp.json to build from; the pinned release is downloaded if absent")
    parser.add_argument("--allow-any-release", action="store_true",
                        help=f"build even if the file is not HPO {HPO_RELEASE} (not a replication)")
    parser.add_argument("--addons", default=str(_HERE / "HPO_addons.csv"))
    parser.add_argument("--model", default=EMBED_MODEL,
                        help="BGE-small: HF id, or a local directory on offline nodes")
    parser.add_argument("--out-meta", default=str(_HERE / "hpo_meta_paper.json"))
    parser.add_argument("--out-vec", default=str(_HERE / "hpo_embedded_paper.npz"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None,
                        help="only embed the first N rows — smoke test, not a usable DB")
    args = parser.parse_args()

    hpo_json = fetch_hpo_json(Path(args.hpo_json))
    data = load_hpo_json(hpo_json)
    release = detect_release(data)
    if release != HPO_RELEASE and not args.allow_any_release:
        raise SystemExit(
            f"{hpo_json} is HPO release {release or 'unknown'}, not {HPO_RELEASE}. The paper ran "
            f"on {HPO_RELEASE}; building on another release is a different experiment. Download "
            f"{HPO_JSON_URL}, or pass --allow-any-release if you mean it."
        )
    terms = extract_terms(data)
    print(f"Loaded {len(terms)} HPO terms from {hpo_json.name} (release {release})")

    addons = pd.read_csv(args.addons)
    print(f"Loaded {len(addons)} add-on phrases from {Path(args.addons).name}")

    rows = build_rows(terms, addons)
    if args.limit:
        rows = rows[: args.limit]
        print(f"LIMIT: {len(rows)} rows only — smoke test")
    print(f"{len(rows)} phrase rows ready for embedding "
          f"({len({r['hp_id'] for r in rows})} distinct HPO ids)")

    emb = embed_rows(rows, args.model, args.batch_size)

    meta = {
        "source": {
            "hpo_release": HPO_RELEASE,
            "hpo_json": hpo_json.name,
            "addons": Path(args.addons).name,
            "embed_model": args.model,
            "recipe": "RAG-HPO HPO_Vectorization.ipynb @ 25c1ea7 (label + synonyms + definition "
                      "+ HPO_addons, clean_text, BGE-small, L2-normalised)",
        },
        "entries": rows,
    }
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"))
    np.savez_compressed(args.out_vec, emb=emb)
    print(f"Saved {len(rows)} embeddings (dim {emb.shape[1]}) → {args.out_meta}, {args.out_vec}")


if __name__ == "__main__":
    main()
