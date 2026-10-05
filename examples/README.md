# examples

| File | Contents |
|---|---|
| `synthetic_reports/*.txt` | Two short reports written for this folder. They describe no real patient. |
| `stand_in_demo.py` | Runs TreePhenoRAG and PhenoJury on the two reports with stand-in models (no GPU, no model files, about ten seconds). |
| `treephenorag_stand_in.jsonl`, `phenojury_stand_in.jsonl` | What `stand_in_demo.py` writes. |

```bash
python examples/stand_in_demo.py --out output/stand_in_demo
```

The stand-ins replace the language models and the sentence encoder with simple rules (a verifier
that checks whether the words of a term's label occur in the segment, and three jurors with scripted
replies), so the terms they find say nothing about the quality of either method. Everything between
the models is the real code: segmentation, retrieval, prompts, traversal, pooling, parsing,
normalisation and voting. The output format is described in `docs/data_formats.md`.

To run the methods with the real models, see "Run on new text" in the root README.
