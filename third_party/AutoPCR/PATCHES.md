# Deviations from upstream AutoPCR

Upstream commit: `c2b46a9c088c696533a43c5cb8e50c7500f055fe`. Every change below is marked in the source
with a `# PhenoRAG patch:` comment, so `grep -rn "PhenoRAG patch" third_party/AutoPCR/` is the complete list.
Nothing in the method, thresholds, prompt, phrase enumeration, candidate ranking, overlap
resolution, has been altered.

Diff against upstream to confirm:

```bash
git diff --no-index <(curl -s https://raw.githubusercontent.com/yctao7/AutoPCR/c2b46a9c088c696533a43c5cb8e50c7500f055fe/src/nn_model2.py) src/AutoPCR/nn_model2.py
```

---

## 1. `utils/llm.py`, lazy provider imports

`httpx`, `dotenv`, `openai`, `groq` and `together` were module-level imports. Only `openai` is in
`environment.yaml`, so importing this module failed before any code ran, including the
local-model path, which needs no provider at all. Each import moved into the branch that uses
it. The hosted code paths are otherwise untouched and still work if the packages are installed.

## 2. `nn_model2.py`, three keyword-only arguments on `bioTag_SapBERT.__init__`

Upstream defaults are preserved in all three cases.

**`prompt_fn`**, the entity-linking client. Upstream always called the module-level
`utils.llm.prompt`, i.e. a hosted model over the network. HCY is KiSpi patient data and may not be
read by a hosted model (docs/cluster.md), so `hpo_extraction.baselines.autopcr_runner.LocalPrompter`
is injected instead: same signature, a local model behind it, and it raises if handed an API
provider. This is the same seam `hpo_extraction/baselines/rag_hpo_runner.py`'s `LocalLlamaClient` uses for RAG-HPO.

**`device`**, the SapBERT encoder was hard-wired to `.cuda()`. It has to be able to sit on CPU
beside a 70B linker (the AutoPCR 70B baseline config sets `ccr_device: cpu`). Encoding also runs under `torch.no_grad()`,
which changes no output.

**`use_gpu_index`**, the FAISS index was unconditionally copied to GPU 0. It is a 40 000 × 768
`IndexFlatIP`, i.e. milliseconds on CPU, and `faiss-gpu==1.7.2` against this environment's
torch/CUDA is an avoidable failure mode. **Default off**, so the shipped behaviour differs from
upstream here, the search results are identical, only the device is not.

## 3. `ee/utils.py`, data paths and NLTK bootstrap

`stopwords.txt` and `NUM.txt` now come from `resources/util/` instead of AutoPCR's `data/`. Both
files are **byte-identical** to upstream's (md5 `cb46557f…` / `60f55459…`), both projects vendor
PhenoBERT's copies, so one file on disk serves both.

`import hpo_extraction.ontology.nltk_data` was added at the top: `PhraseItem.StopWords = stopwords.words("english")`
runs in a **class body**, so a machine that cannot reach the corpora fails at import, before first use.

## 4. `ee/pbert.py`, the Stanza pipeline is built lazily, through this repo's loader

Upstream constructed `stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'})` at module
import with no resources directory, so it reached for `~/stanza_resources`, and the network, the
moment anything imported `ee`. This repo already builds the identical pipeline in
`hpo_extraction.data.segmentation.load_stanza`, including `download_method=None`, which is the only form that works
on the offline compute nodes. Construction is deferred to first use via `get_clinical_ner_model()`,
and `set_stanza_dir()` points it at a resources tree.

## 5. `ee/__init__.py`, benepar imported lazily

Importing the package pulled in `benepar` / `spacy` / `torch-struct`. Those live in a separate
environment (`autopcr_ee_venv`, `requirements_ee.txt`): upstream pins transformers 4.49 and
protobuf 6, which PhenoRAG_marc_env cannot also hold. `process_text2phrases_benepar` and
`process_text2phrases_abbr` are thin wrappers that import on call, so the linker run in the main env
never needs the parser stack. It never calls them either: the parse is done beforehand by
`extract_phrases.py`, and `run_gsc_test_ner` loads the cache it leaves (see §6 and §8).

## 6. `HPO_evaluation.py`, two import moves, a guarded evaluator, the benepar cache loop lifted out

`dic_ner` / `tagging_text` were module-level imports belonging to the rule-based extraction path,
which this fork does not vendor. Moved into `run_gsc_test`, their only caller.

The trailing `GSCplus_corpus(...)` call is now guarded on `files['goldfile']`. Upstream always ran
its own evaluator against the ground truth TSV it had read the document text from. The baseline runs stage a
corpus with **no ground truth lines**, because scoring is the comparison's job over the predictions
contract. Passing a `goldfile` still runs upstream's metric, which is how the independent
cross-check on GSC+ is done.

**The benepar cache loop is lifted to module level, verbatim.** `remove_dup` (nested in
`run_gsc_test_ner`) and the loop that builds `phrases_benepar.json` / `phrases_conjunct.json` are now
the module-level `remove_dup` and `build_benepar_phrase_cache(all_test, test_path)`.
`run_gsc_test_ner` calls the latter where the loop used to run. The point is that
`extract_phrases.py` can run *the same code* in `autopcr_ee_venv`: one copy, so the parse done in
another environment is upstream's parse. Checked by
`tests/unit/test_autopcr.py::TestParserCache::test_cache_builder_is_upstreams_loop`.

## 7. `build_index.py`, explicit UMLS path, explicit device, no network fallback

Upstream decided whether to attach the `UMLS synonyms` linking field by comparing the
`ontology_dict` **string** to the literal `'../dict/HPO'`, and fell back to live UMLS API calls when
the file was absent, using a `requests` it never imported. Both are unusable here. The path is now
an explicit optional `--umls_synonyms` flag. Absent, the index is built without that field, which
`slurm/autopcr_index.sbatch` reports loudly because it changes the linker prompt.

`from dotenv import load_dotenv` moved into `get_umls_name` for the same reason as patch 1, and a
`--device` flag was added so the index can be built on CPU.

## 8. `ee/benepar.py`, vendored, with model loading made lazy and network-free

Upstream loaded three models **at import** and, on a miss, fetched them: `spacy.cli.download`
(`en_core_web_trf`), `benepar.download` (`benepar_en3_large`), and
`os.system('pip install <S3 url>')` (scispaCy's `en_core_sci_scibert`). None of those hosts is
reachable from the cluster. The changes:

- **No network.** Each fallback now raises an error that names `experiments/06_comparison/baselines/stage_autopcr_parser_assets.sh`,
  which downloads both models on a machine with internet and proves offline completeness (see
  README, "The constituency parser").
- **Lazy construction** through `get_parser_model()` / `get_abbr_model()`, the same shape as
  `ee/pbert.py`'s Stanza pipeline (§4). `spacy`, `benepar` and `scispacy` are imported inside them.
- **scispaCy only for `abbr_recog`.** The abbreviation model is built on first use by
  `process_text2phrases_abbr`, and never at `abbr_recog=False`. That is the published GSC-2024
  setting (upstream's README passes `--abbr_recog` only for NCBI), so scispaCy is not installed.

The model names, `generate_spans`, the conjunct decomposition and everything else in the file are
upstream's, character for character: `diff` against the fix shows only the loading block and the two
call sites (`clinical_ner_model` → `get_parser_model()`, `abbr_model` → `get_abbr_model()`).

---

## NOT changed

**`ee/pbert.py`'s phrase extraction is not replaced by another clinical NLP pipeline.** The two are
the same PhenoBERT lineage and even share the n-gram enumeration tail, so substituting them looks
like obvious de-duplication. It is not: `ontology/clinical_nlp.py` lowercases, strips accents and
maps hyphens to spaces *before* NER, where AutoPCR moved that into `normalize_with_char_map` and
keeps hyphens. Swapping them would silently change which phrases AutoPCR proposes, a different
method, reported under AutoPCR's name.

**`run_gsc_test_ner` passes `lines[1] * len(phrases)` as the `texts` argument.** That repeats the
document string rather than building a list, which looks like a bug, and is one, but `texts` is
unused: `gen_grounding_prompt2` has the context line commented out. Left verbatim, because "fixing"
it would add report context to the linker prompt and stop reproducing the published method.

## Added files (ours, not upstream's)

- `build_dict_from_hpo_json.py`, builds upstream's dictionary from this repo's fixed
  `resources/util/hpo.json` instead of an `hp.obo` download. See `README.md` for why.
- `extract_phrases.py`, runs `build_benepar_phrase_cache` in `autopcr_ee_venv` and copies the
  result to every experiment staged on a byte-identical corpus.
- `requirements_ee.txt`, that environment, upstream's own pins from its `environment.yml`.
- `README.md`, `PATCHES.md`, this documentation.
