# Changes to upstream PhenoBERT

Upstream: <https://github.com/EclipseCN/PhenoBERT>, commit
`195d5df2ad9c6265e1de523f139694c7646eee29`. The files here are that commit, with the changes below.
The changes were made on 2026-06-20 in the checkout the thesis runs used, so that PhenoBERT loads
under PyTorch 2.6 and later. They change how files are loaded, not what PhenoBERT computes.

```bash
git diff b0ad5510 -- third_party/PhenoBERT    # every change, line by line
```

## `torch.load(..., weights_only=False)`

PyTorch 2.6 changed the default of `torch.load` to `weights_only=True`, which refuses PhenoBERT's
model files: they are whole pickled models, not state dicts. Every call that loads them now passes
`weights_only=False`, as earlier PyTorch versions did by default.

| File | Calls changed |
|---|---|
| `phenobert/utils/util.py` | `ModelLoader.load_params`, `ModelLoader.load_all` (3) |
| `phenobert/utils/my_bert_match.py` | `test` (1) |
| `phenobert/utils/fastNLP/core/callback.py` | 1 |
| `phenobert/utils/fastNLP/core/dist_trainer.py` | 1 |
| `phenobert/utils/fastNLP/core/trainer.py` | 2 |
| `phenobert/utils/fastNLP/embeddings/elmo_embedding.py` | 1 |
| `phenobert/utils/fastNLP/io/model_io.py` | 2 |
| `phenobert/utils/fastNLP/modules/encoder/bert.py` | 1 |

In annotation, only the model files published by the PhenoBERT authors pass through these calls.

## `fastNLP/core/batch.py`: `SamplerAdapter.__init__`

`super().__init__(dataset)` became `super().__init__()`. Newer PyTorch versions dropped the
`data_source` argument of `torch.utils.data.Sampler.__init__`, and the argument was never used.

## Not copied

- training data and evaluation corpora (`phenobert/models/*.txt`, `phenobert/data/GSC+`, `ID-68`,
  `val`, `GeneReviews`, `gene_reviews.idx`). Only the training and evaluation scripts read them.
- `phenobert/img/`, `setup.py`, `.travis.yml`, the prebuilt executable `phenobert/utils/handle_hpo`
  (its Go source, `handle_hpo.go`, is copied), and the `__pycache__` files upstream tracks.
- the model weights and embeddings (`phenobert/models/`, `phenobert/embeddings/`), which are large
  and not tracked. `third_party/README.md` says where they come from.
