"""PhenoRAG glue for the AutoPCR benchmark (parallels hpo_extraction.baselines.rag_hpo_runner / hpo_extraction.phenojury.phenobert).

AutoPCR's own logic lives in ``src/AutoPCR/``, vendored, with every deviation listed in
``third_party/AutoPCR/PATCHES.md``. This module supplies the four things it needs to run here and nothing
else:

* **a local entity linker.** Upstream's linking step calls a hosted model (OpenAI / Groq / Together
  / vLLM). HCY is KiSpi patient data and may not be read by a hosted model, so :class:`LocalPrompter`
  replaces that client with a local model, the same seam ``rag_hpo_runner.LocalLlamaClient``
  uses. It refuses to construct if handed an API provider, so the boundary is mechanical rather
  than a comment.
* **a corpus in AutoPCR's format.** :func:`stage_corpus` writes the NCBI-style ``pmid\\ntext``
  blocks upstream reads, from the ``{report_id: text}`` mapping ``hpo_extraction.data.loading.load_cohort``
  returns.
* **candidate capture.** ``run_gsc_test_ner`` keeps only the top-1 concept per phrase, but the
  ranked candidates are what ``*_retrieved_segments.jsonl`` is made of. :class:`RecordingSapBERT`
  keeps them.
* **an output reader.** :func:`parse_autopcr_tsv` turns upstream's TSV back into records.

Everything in between is upstream's ``HPO_evaluation.run_gsc_test_ner``, called unmodified.

The AutoPCR imports are **function-local**: they pull in faiss, torch and transformers,
and the parsing/staging functions above are the only part with logic worth unit-testing.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from hpo_extraction.paths import AUTOPCR_DIR as _AUTOPCR_DIR

logger = logging.getLogger(__name__)


def _ensure_autopcr_path() -> None:
    """Put ``src/AutoPCR`` on ``sys.path``, only when AutoPCR is actually about to be imported.

    Its modules are flat (``from utils import prompt``, ``from build_dict import lemmatizer``), so
    the directory has to be on the path itself, as ``rag_hpo_runner`` does for the
    hyphenated ``src/RAG-HPO``. It is done lazily because ``utils`` and ``ee`` are generic enough
    names to shadow somebody else's, and the staging/parsing helpers in this module, the part
    worth unit-testing, need none of it.
    """
    if str(_AUTOPCR_DIR) not in sys.path:
        sys.path.insert(0, str(_AUTOPCR_DIR))


#: Score AutoPCR writes for a concept the LLM chose, not retrieval (``nn_model2.predict_llm``
#: prepends ``(concept, id, -1.0)``). It is a sentinel, not a similarity.
LLM_CHOSEN_SCORE = -1.0


# --------------------------------------------------------------------------------------------
# The entity-linking client
# --------------------------------------------------------------------------------------------
class LocalPrompter:
    """Drop-in replacement for ``AutoPCR.utils.llm.prompt``, backed by a local model.

    Upstream's signature is ``prompt(pr, model, api_provider, api_key, seed, max_tokens) -> {i: str}``
    where ``pr`` maps an index to ``{"system": ..., "user": ...}``. That is reproduced, so
    ``nn_model2.predict_llm`` cannot tell the difference.

    Decoding is greedy: upstream sends ``temperature=0`` to every provider, and every other earlier method here is greedy too. The API client's rate-limiting sleep and its ``seed`` are dropped, neither means anything for deterministic local generation, and ``seed`` is accepted only to
    keep the signature.
    """

    def __init__(self, llm, max_new_tokens: int = 32):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.n_calls = 0

    def __call__(self, pr, model=None, api_provider=None, api_key=None, seed=0, max_tokens=None):
        if api_provider:
            raise ValueError(
                f"LocalPrompter was handed api_provider={api_provider!r}. AutoPCR's hosted backends "
                "are disabled in this fork: HCY report text may not be sent to a hosted model "
                "(see docs/cluster.md, the patient-data boundary). Leave api_provider unset."
            )
        max_new_tokens = int(max_tokens or self.max_new_tokens)
        responses: dict = {}
        for i, item in pr.items():
            responses[i] = self.llm.generate(
                item["user"], item["system"], max_new_tokens=max_new_tokens
            )
            self.n_calls += 1
        return responses


def load_local_linker(llama_dir: str, load_in_4bit: bool = False, max_new_tokens: int = 32):
    """``LocalPrompter`` over a local LLaMA, the AutoPCR 8B baseline / the AutoPCR 70B baseline entity linker."""
    from hpo_extraction.models.llama import LlamaLLM, load_llama

    tokenizer, model = load_llama(llama_dir, load_in_4bit=load_in_4bit)
    return LocalPrompter(LlamaLLM(model, tokenizer), max_new_tokens=max_new_tokens)


# --------------------------------------------------------------------------------------------
# Staging the cohort into AutoPCR's corpus format
# --------------------------------------------------------------------------------------------
def flatten_for_corpus(text: str) -> str:
    """Collapse line breaks to spaces, preserving every character offset.

    AutoPCR's corpus format is ``pmid`` on one line and the **whole document** on the next, blocks
    separated by a blank line, so a multi-line HCY report cannot be staged verbatim the way
    ``phenobert_experiment.stage_reports`` stages one. Each line terminator is replaced by a single
    space, which is length-preserving, so the character offsets AutoPCR emits still index the
    original report. Nothing else is touched: normalising the text would invalidate every offset in
    the detections artifact.
    """
    return text.replace("\r\n", "  ").replace("\n", " ").replace("\r", " ")


def stage_corpus(reports: dict[str, str], report_ids: list[str], corpus_path: str) -> dict[str, str]:
    """Write the corpus file and return ``{pmid: report_id}``.

    The id map is returned, not re-derived, for the same reason
    ``phenobert_experiment.stage_reports`` returns one: the mapping is not invertible in general.
    Here it happens to be the identity, and that is asserted, not assumed, a report id
    carrying a newline or a tab would silently corrupt the corpus.
    """
    os.makedirs(os.path.dirname(corpus_path) or ".", exist_ok=True)
    pmid_to_id: dict[str, str] = {}
    blocks = []
    for report_id in report_ids:
        pmid = str(report_id)
        if any(c in pmid for c in "\r\n\t"):
            raise ValueError(
                f"report id {report_id!r} contains a newline or tab — it cannot be a pmid in "
                "AutoPCR's line-oriented corpus format"
            )
        if pmid in pmid_to_id:
            raise ValueError(f"duplicate report id {report_id!r} staged twice")
        pmid_to_id[pmid] = report_id
        blocks.append(pmid + "\n" + flatten_for_corpus(reports[report_id]))
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + "\n")
    return pmid_to_id


PHRASE_CACHE_FILES = ("phrases.json", "phrases_abbr.json", "phrases_benepar.json",
                      "phrases_conjunct.json")


def invalidate_stale_phrase_cache(corpus_dir: str, pmids: set[str]) -> list[str]:
    """Drop cached phrase extractions that do not cover the staged corpus.

    ``run_gsc_test_ner`` caches its extracted phrases in ``phrases.json`` beside the corpus and
    reuses the file whenever it exists, without checking which documents it holds. Re-running a
    cohort after a ``max_patients=2`` pilot would therefore load a two-document cache and then
    ``KeyError`` on the third report, a confusing crash standing in for a stale cache.

    Reusing the cache when it *does* match is worth keeping: phrase extraction is the Stanza pass,
    the expensive half of a run with no LLM calls in it. So the file is validated, not
    deleted unconditionally, in the same spirit as ``phenobert_experiment._output_is_complete``:
    a cache is reused only when it covers every staged report and nothing else.

    Returns the names of the files removed.
    """
    removed = []
    for name in PHRASE_CACHE_FILES:
        path = os.path.join(corpus_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                cached = set(json.load(f))
        except (ValueError, OSError):
            cached = None            # unreadable is stale by definition
        if cached != pmids:
            os.remove(path)
            removed.append(name)
    return removed


#: The two files the constituency-parse path (``ee="neural+"/"neural++"``) needs beside the corpus.
#: Upstream opens BOTH whenever the first exists, even at ``neural+``.
PARSER_CACHE_FILES = ("phrases_benepar.json", "phrases_conjunct.json")


def require_parser_cache(corpus_dir: str, ee: str, dataset: str = "<ds>") -> None:
    """``SystemExit`` unless the benepar phrase cache is in place for a ``neural+``/``neural++`` run.

    The parser does not live in this environment: upstream pins transformers 4.49 / protobuf 6,
    so it runs in ``autopcr_ee_venv`` via ``third_party/AutoPCR/extract_phrases.py`` and leaves these two
    files, which ``run_gsc_test_ner`` then loads instead of parsing. Without them upstream would try
    to import benepar here and die with an ``ImportError``, after a 70B had spent tens of minutes
    being placed. Call this after :func:`invalidate_stale_phrase_cache`, so a cache for a different
    corpus (a pilot's, say) has already been dropped and is reported as missing, not reused.
    """
    if ee not in ("neural+", "neural++"):
        return
    missing = [n for n in PARSER_CACHE_FILES if not os.path.isfile(os.path.join(corpus_dir, n))]
    if missing:
        raise SystemExit(
            f"ee={ee!r} needs the constituency-parse cache, and {missing} is missing from "
            f"{corpus_dir}. It is produced in autopcr_ee_venv, not here:\n"
            f"  DATASET={dataset} sbatch slurm/autopcr_parse.sbatch   (same max_patients as this run)\n"
            "which stages the corpus for both AutoPCR baselines and parses it once for both."
        )


def phrase_counts(corpus_dir: str) -> dict[str, int]:
    """Phrases per extraction source in the cache files present, what each ``ee`` step added."""
    counts = {}
    for name in PHRASE_CACHE_FILES:
        path = os.path.join(corpus_dir, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                counts[name.removesuffix(".json")] = sum(len(v["phrases"]) for v in json.load(f).values())
    return counts


# --------------------------------------------------------------------------------------------
# Reading AutoPCR's output
# --------------------------------------------------------------------------------------------
def normalise_code(code: str) -> str:
    """``HP_0001250`` / ``HP:0001250`` -> ``HP:0001250``.

    Which form appears depends on the path: retrieval hits carry ``index_to_id``'s colon form, while
    an LLM-chosen id comes back through ``id.replace('_', ':')``, but the model is prompted with
    the _ character form, so a truncated or reformatted answer can arrive either way.
    """
    return code.strip().replace("HP_", "HP:", 1)


def parse_autopcr_tsv(raw: str):
    """Yield ``(pmid, start, end, phrase, hpo_id, score)`` for every annotation row.

    Upstream's output is blank-line-separated blocks: pmid, document text, then one
    ``start \\t end \\t phrase \\t id \\t score`` row per accepted mention. Rows that do not parse are
    skipped and reported by the caller through :func:`diagnose_autopcr_output`, never dropped
    silently.
    """
    for block in raw.strip().split("\n\n"):
        lines = block.split("\n")
        if len(lines) < 2:
            continue
        pmid = lines[0]
        for line in lines[2:]:
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            try:
                start, end = int(parts[0]), int(parts[1])
                score = float(parts[4]) if len(parts) > 4 else None
            except ValueError:
                continue
            yield pmid, start, end, parts[2], normalise_code(parts[3]), score


def diagnose_autopcr_output(path: str) -> dict:
    """Counters over the output file, how many blocks, rows, and rows carrying an ``HP:`` code.

    ``phenojury_generation_free_listing`` once shipped a whole grid of green jobs that had grounded zero HPO terms before
    anyone noticed. A file full of rows that parse to nothing looks like a cohort with no
    phenotypes, so the driver turns ``n_hp_rows == 0`` into a hard failure using this.
    """
    counts = {"n_blocks": 0, "n_rows": 0, "n_hp_rows": 0, "n_unparsed": 0}
    if not os.path.isfile(path):
        return counts
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    for block in raw.strip().split("\n\n"):
        lines = block.split("\n")
        if len(lines) < 2:
            continue
        counts["n_blocks"] += 1
        for line in lines[2:]:
            counts["n_rows"] += 1
            parts = line.split("\t")
            if len(parts) >= 4 and normalise_code(parts[3]).startswith("HP:"):
                counts["n_hp_rows"] += 1
            else:
                counts["n_unparsed"] += 1
    return counts


def link_source(score) -> str:
    """``"llm"`` when AutoPCR's linker chose the concept, ``"retrieval"`` when SapBERT did.

    ``nn_model2.predict_llm`` prepends the LLM's answer with a score of ``-1.0``. Everything
    else is a cosine similarity, or ``1.0`` for the abbreviation / first-word dictionary shortcuts.
    This column is the whole point of comparing AutoPCR against RAG-HPO: it separates what the
    ontology index found from what the model decided.

    The ``-1.0`` sentinel is unambiguous. ``"dictionary"`` is not quite: a cosine of 1.0
    would be read as a dictionary hit. Nothing else in the output TSV distinguishes the two, and an
    L2-normalised inner product landing on 1.0 is rare, but it is a boundary, not a proof,
    and the llm/retrieval split (the one the finding rests on) is unaffected either way.
    """
    if score is None:
        return "unknown"
    if float(score) == LLM_CHOSEN_SCORE:
        return "llm"
    if float(score) >= 1.0:
        return "dictionary"
    return "retrieval"


# --------------------------------------------------------------------------------------------
# Running it
# --------------------------------------------------------------------------------------------
def _recording_sapbert_class():
    """``bioTag_SapBERT`` subclass that keeps the ranked candidates ``run_gsc_test_ner`` discards.

    Defined inside a function so importing this module does not pull in faiss/torch.
    """
    _ensure_autopcr_path()
    from nn_model2 import bioTag_SapBERT  # AutoPCR's own

    class RecordingSapBERT(bioTag_SapBERT):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls: list[tuple[list[str], list]] = []

        def predict_llm(self, x, texts, confidence=("HIGH")):
            anns = super().predict_llm(x, texts, confidence)
            # One call per document, in the order run_gsc_test_ner walks the corpus. The driver
            # asserts len(calls) == number of staged reports before pairing them up.
            self.calls.append((list(x), anns))
            return anns

    return RecordingSapBERT


def check_dictionary(ontology_dict: str) -> dict[str, str]:
    """The six files ``bioTag_SapBERT`` opens, or ``SystemExit`` naming the missing ones.

    Called by the driver **before** the linker is loaded: a 70B takes tens of minutes to place, and
    discovering a missing index afterwards wastes the whole allocation for a check that costs six
    ``stat`` calls.
    """
    vocabfiles = {
        "index": f"{ontology_dict}/main_lable.index",
        "index_to_id": f"{ontology_dict}/index_to_id_lable.json",
        "index_to_term": f"{ontology_dict}/index_to_term_lable.json",
        "id_to_concept": f"{ontology_dict}/id_to_concept.json",
        "firstword_to_id": f"{ontology_dict}/firstword_id_map.json",
        "abbr": f"{ontology_dict}/abbr.json",
    }
    missing = [p for p in vocabfiles.values() if not os.path.isfile(p)]
    if missing:
        raise SystemExit(
            f"AutoPCR dictionary incomplete: {missing}. Build it with\n"
            f"  python third_party/AutoPCR/build_dict_from_hpo_json.py -o {ontology_dict}\n"
            f"  python third_party/AutoPCR/build_index.py --ontology_dict {ontology_dict} "
            "--ccr <local SapBERT dir>\n"
            "(see third_party/README.md)"
        )
    return vocabfiles


def build_model(
    ontology_dict: str,
    ccr_model: str,
    tau_1: float,
    tau_2: float,
    k: int,
    prompt_fn,
    el: str = "local",
    seed: int = 0,
    use_cache: bool = True,
    device: str | None = None,
    use_gpu_index: bool = False,
):
    """AutoPCR's retriever+linker, wired to a local encoder and a local linking client."""
    vocabfiles = check_dictionary(ontology_dict)
    cls = _recording_sapbert_class()
    return cls(
        vocabfiles, ccr_model, tau_1, tau_2, k, el,
        None,          # api_provider, LocalPrompter refuses anything else
        None,          # api_key
        seed, use_cache,
        prompt_fn=prompt_fn, device=device, use_gpu_index=use_gpu_index,
    )


def run_autopcr(model, corpus_path: str, out_path: str, ontology_dict: str,
                tau_1: float, ee: str = "neural", only_longest: bool = False,
                abbr_recog: bool = False, stanza_dir: str | None = None) -> None:
    """Call upstream's ``run_gsc_test_ner`` unmodified over the staged corpus.

    ``out_path`` must end in ``.tsv``: upstream derives its second output file as
    ``outfile[:-4] + '.raw.tsv'``.
    """
    if not out_path.endswith(".tsv"):
        raise ValueError(f"out_path must end in .tsv (upstream slices it): {out_path!r}")
    if ee not in ("neural", "neural+", "neural++"):
        raise ValueError(
            f"ee={ee!r} unsupported here. 'rule' needs the PhenoTagger stack this fork does not "
            "vendor."
        )
    # 'neural+'/'neural++': the parse was done in autopcr_ee_venv (require_parser_cache has checked
    # its files are here), so run_gsc_test_ner loads them and never imports benepar in this env.
    require_parser_cache(os.path.dirname(corpus_path), ee)

    _ensure_autopcr_path()
    import ee as ee_pkg  # AutoPCR's package

    if stanza_dir:
        ee_pkg.set_stanza_dir(stanza_dir)

    from HPO_evaluation import run_gsc_test_ner

    files = {
        # Forward slashes on purpose: upstream derives the phrase-cache directory with
        # "/".join(testfile.split("/")[:-1]).
        "testfile": Path(corpus_path).as_posix(),
        "outfile": Path(out_path).as_posix(),
        "ontology_dict": ontology_dict,
        # No 'goldfile': scoring is the result-table library's job over the predictions contract, and the staged
        # corpus carries no ground truth lines. See third_party/AutoPCR/PATCHES.md.
    }
    run_gsc_test_ner(files, model, tau_1, only_longest, abbr_recog, ee, None)


# --------------------------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------------------------
def selfcheck(ontology_dict: str, ccr_model: str, device: str | None = None) -> int:
    """Load the index, retrieve for a handful of phrases, print the top hits.

    Exercises the whole offline chain, SapBERT model, FAISS file, the dictionary built from
    the fixed ontology, in seconds, which is the cheap way to find out that a path is wrong
    before a GPU allocation does.
    """
    probes = {"seizures": "HP:0001250", "intellectual disability": "HP:0001249",
              "short stature": "HP:0004322"}
    model = build_model(ontology_dict, ccr_model, 0.95, 0.85, 5, prompt_fn=LocalPrompter(None),
                        el="none", device=device)
    cands = model.get_cand(list(probes), 5)
    failures = 0
    for i, (phrase, expected) in enumerate(probes.items()):
        top = [(c[1], round(c[2], 3), c[0].get("label")) for c in cands[i]]
        hit = "OK " if top and top[0][0] == expected else "MISS"
        if hit == "MISS":
            failures += 1
        print(f"{hit} {phrase!r} -> {top}")
    print(f"selfcheck: {len(probes) - failures}/{len(probes)} top-1 as expected")
    return failures


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="AutoPCR retrieval self-check")
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--ontology_dict", default=str(_AUTOPCR_DIR / "dict" / "HPO"))
    p.add_argument("--ccr", required=True, help="local SapBERT directory")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    if args.selfcheck:
        raise SystemExit(1 if selfcheck(args.ontology_dict, args.ccr, args.device) else 0)
    p.error("nothing to do — pass --selfcheck")
