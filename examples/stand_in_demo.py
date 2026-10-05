"""Run TreePhenoRAG and PhenoJury on the synthetic reports, with stand-in models.

    python examples/stand_in_demo.py [--out output/stand_in_demo]

Needs no GPU, no model files and no network. The language models and the sentence encoder are
replaced by small stand-ins, so the terms it finds say nothing about the quality of either method.
What it shows is the input and output format, and that every step between the models runs:
segmentation, retrieval, prompts, traversal of the ontology, pooling, parsing, normalisation and
voting. The stand-ins follow the application tests (``tests/unit/test_applications.py``):

* encoder: hashed word counts;
* TreePhenoRAG verifier: answers *Yes* (margin +6) when every word of five or more letters of the
  term's label occurs in the segment, otherwise *No* (margin -6);
* PhenoJury jurors: three scripted jurors that name *Seizure* when a sentence mentions seizures,
  and one of them also names *Hypotonia* when a sentence says *hypotonic*.

TreePhenoRAG searches only the subtree below *Seizure* (HP:0001250) to keep the run short.
"""
from __future__ import annotations

import argparse
import re
import zlib
from pathlib import Path

import numpy as np

from hpo_extraction.ontology.hpo_tree import HPOTree
from hpo_extraction.results import read_reports, write_jsonl

HERE = Path(__file__).resolve().parent
SEIZURE = "HP:0001250"


class BagOfWords:
    """Stand-in sentence encoder: hashed word counts, one constant dimension against zero rows."""

    def encode(self, texts, **_):
        """One 257-dimensional count vector per text."""
        out = np.zeros((len(texts), 257), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[i, zlib.crc32(word.encode()) % 256] += 1.0
            out[i, 256] = 0.1
        return out


class WordVerifier:
    """Stand-in verifier: margin +6 when every label word of five or more letters is in the segment."""

    def generate_batch(self, prompts, system_prompt):
        """One ``{"margin": float}`` per prompt (log-odds of *Yes* against *No*)."""
        out = []
        for prompt in prompts:
            label = re.match(r"The symptom (.+?)(?: is defined as|\. | is also)", prompt).group(1)
            segment = prompt.rsplit(":'", 1)[1].rsplit("'.", 1)[0].lower()
            words = [w for w in re.findall(r"[a-z]+", label.lower()) if len(w) >= 5]
            out.append({"margin": 6.0 if words and all(w in segment for w in words) else -6.0})
        return out


class ScriptedJuror:
    """Stand-in juror: names Seizure, and optionally Hypotonia, when the sentence mentions them."""

    supports_batching = True

    def __init__(self, names_hypotonia: bool):
        self.names_hypotonia = names_hypotonia

    def generate_many(self, system_prompt, user_msgs, max_new_tokens):
        """One reply per message, one phenotype per line, ``NONE`` when there is none."""
        replies = []
        for msg in user_msgs:
            sentence = msg.lower()
            lines = []
            if "seizure" in sentence:
                lines.append("Seizure")
            if self.names_hypotonia and "hypotonic" in sentence:
                lines.append("Hypotonia")
            replies.append("\n".join(lines) or "NONE")
        return replies

    def unload(self):
        """Nothing to free."""


def run_treephenorag(tree: HPOTree, reports: dict[str, str]) -> list:
    """TreePhenoRAG results of *reports* with the stand-in encoder and verifier."""
    from hpo_extraction.treephenorag import pipeline as tp
    from hpo_extraction.treephenorag import verifier_prompt

    # The prompt builder loads the ontology file on every call. Handing it the loaded tree keeps
    # the demo fast and changes nothing else.
    verifier_prompt.HPOTree = lambda: tree
    settings = tp.TreePhenoRAGSettings(subtree_root=SEIZURE, segments_per_term=3)
    method = tp.TreePhenoRAG(tree, BagOfWords(), WordVerifier(), settings)
    return [method.extract(text, report_id) for report_id, text in reports.items()]


def run_phenojury(tree: HPOTree, reports: dict[str, str]) -> list:
    """PhenoJury results of *reports* with three scripted jurors and the dictionary normaliser."""
    from hpo_extraction.phenojury import generation
    from hpo_extraction.phenojury import pipeline as pj

    jurors = {"a": ScriptedJuror(True), "b": ScriptedJuror(False), "c": ScriptedJuror(False)}
    generation.load_slm = lambda key, path, logger=None, deterministic=False: jurors[key]
    settings = pj.PhenoJurySettings(jurors=list(jurors), k=2, unit="segment")
    method = pj.PhenoJury(tree, {key: f"stand-in/{key}" for key in jurors},
                          pj.dictionary_normaliser(tree), settings)
    return [method.extract(text, report_id) for report_id, text in reports.items()]


def main() -> int:
    """Run both methods on ``examples/synthetic_reports`` and write one JSON Lines file each."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="output/stand_in_demo", help="output folder")
    args = parser.parse_args()
    out = Path(args.out)
    reports = read_reports(HERE / "synthetic_reports")
    tree = HPOTree()
    tree.buildHPOTree()
    for name, run in (("treephenorag", run_treephenorag), ("phenojury", run_phenojury)):
        results = run(tree, reports)
        path = write_jsonl(results, out / f"{name}_stand_in.jsonl")
        print(f"{name}: wrote {path}")
        for result in results:
            terms = ", ".join(f"{t.hpo_id} {t.label} ({t.score:.2f})" for t in result.terms)
            print(f"  {result.report_id}: {terms or 'no terms'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
