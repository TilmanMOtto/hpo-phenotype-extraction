"""A synthetic cohort in the five drivers' exact on-disk schemas, small enough to reason about.

The builder joins five artifact families, an ontology, a curated ground truth and a sampling frame. Testing
that against the cluster is not testing -- it is watching. So this writes a three-report cohort
where every interesting case is present **by design** and its expected outcome is arithmetic
somebody worked out by hand:

``SYN001`` carries all of them::

    ground truth = {C (Seizure), D, F}

    PhenoBERT    predicts {C, F'}   C tp; F' fp; D detected and NEGATED -> fn; F never seen -> fn
    AutoPCR      predicts {C, D}    C by dictionary, D by the LLM route; F was a losing candidate
    RAG-HPO      predicts {C', D}   C' is C's ALT ID -- a tp only if resolution works
    PhenoJury    predicts {C, D, F} three jurors on C, two on D, two on F
    TreePhenoRAG predicts {C, M}    C accepted; M accepted and wrong; D scored but under
                                    tau_accept; F pruned, blocked at its unexpanded parent E

``SYN002`` is the easy case -- one annotated term everybody finds -- so a view cannot pass by being
right only about failure. ``SYN003`` is in the frame and in the ground truth and its segmentation does
**not** align to its text, which must make it a named skip rather than a drawn report.

Three properties are essential and easy to lose:

**The report is written latin1 and contains umlauts.** That is how ``hpo_extraction.data.loading.load_txt``
reads HCY, every HCY report is German, and a builder that read them as UTF-8 would silently drop
all of them. If this fixture were ASCII the regression would pass.

**RAG-HPO predicts the alt id.** A ground truth file predates the release it is scored against, so the same
phenotype gets two spellings. Comparing raw strings paints a found term red. This row fails loudly
when someone removes the resolution.

**The tree cache is built by ``ingest_score_cache`` from a real calls JSONL**, not hand-written as
an ``.npz``. The archive layout is that function's business, and a fixture that wrote the arrays
directly would keep passing after the producer changed.

The ontology is ``tests/fixtures/toy_ontology`` -- ten nodes, a multi-parent node, an alt id and an
out-of-subtree term, all hand-checked. A second toy ontology here would be a second thing to keep
true.

The GSC+ cohort
---------------
:func:`write_gsc` writes a second, smaller cohort in the *other* shape the builder supports, and
it exists to exercise the parts that are not shared. Two abstracts, **UTF-8**, a corpus
annotation with character offsets, and RAG-HPO's own re-annotation on top of it -- which is what
makes all four rungs of ``gold.RagHpoGold``'s placement ladder present by design::

    GSC001  ground truth = {C, D, F}
            C  annotated by the corpus with offsets that check out  -> drawn on them
            D  annotated by RAG-HPO only, description occurs verbatim -> drawn, `raghpo-only`
            F  annotated by RAG-HPO only, description occurs nowhere  -> unplaced, in the footer
    GSC002  ground truth = {C}
            C  annotated by the corpus with offsets that do NOT spell their own mention
               -> located by the mention instead, and badged as such

TreePhenoRAG is present with **only** a transferred prediction set and no score cache, because
that is what GSC+ actually has: the TreePhenoRAG protocol carries the HCY-selected configuration over unchanged,
so there are no folds and nothing to re-run. The column must draw a verdict and say why it can
say no more, which is a degradation path with no HCY equivalent.
"""

from __future__ import annotations

import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


def toy_ontology():
    """The shared toy ontology module, with ``tests/fixtures`` put on the path to reach it."""
    path = os.path.join(_REPO, "tests", "fixtures")
    if path not in sys.path:
        sys.path.insert(0, path)
    import toy_ontology  # noqa: F401

    return toy_ontology


def toy_view():
    """``OntologyView`` over the toy tree -- the argument every adapter takes as ``ctx.view``."""
    from hpo_extraction.evaluation.metrics import OntologyView

    return OntologyView(toy_ontology().build_toy_tree())


# -- the cohort --------------------------------------------------------------

#: Report text, latin1 and German, one entry per sentence. The blank line between paragraphs is
#: deliberate: ``spans.align_segments`` has to step over text that belongs to no segment.
REPORTS = {
    "SYN001": [
        "Der Junge zeigt eine ungewöhnliche Bewegungsstörung.",
        "Er hatte Anfälle im Alter von zwei Monaten.",
        "Ein feines Zittern der Hände wurde beobachtet.",
        "Sonst unauffällig.",
    ],
    "SYN002": [
        "Das Mädchen hatte wiederholte Anfälle.",
        "Die Eltern sind gesund.",
    ],
    "SYN003": [
        "Ein kurzer Bericht ohne Auffälligkeiten.",
    ],
}

#: What ``segmented_reports.csv`` claims. SYN003's row names a sentence that is not in its report,
#: which is what makes it the un-alignable case.
SEGMENTS = dict(REPORTS, SYN003=["Eine Zeile die im Bericht gar nicht vorkommt."])

GOLD_DIR = "curated_ground_truth_2026-01-01"

JURORS = ("apertus", "deepseek", "intelligent_internet", "openbiollm",
          "llama", "medpsy", "medgemma", "phi4")

PROMPT = "q2_sentence_last"
NORMALISER = "phenobert_candidates"


def _codes():
    toy = toy_ontology()
    return {
        "A": toy.A, "B": toy.B, "C": toy.C, "D": toy.D, "E": toy.E,
        "F": toy.F, "G": toy.G, "H": toy.H, "M": toy.M,
        "C_ALT": toy.OBSOLETE_C,
    }


def report_text(report_id: str) -> str:
    """Two paragraphs joined by a blank line, so segment ranges are not contiguous."""
    lines = REPORTS[report_id]
    if len(lines) < 3:
        return "\n".join(lines) + "\n"
    return " ".join(lines[:2]) + "\n\n" + " ".join(lines[2:]) + "\n"


def write(root: str) -> dict:
    """Write the whole cohort under *root*. Returns the paths a :class:`Paths` needs."""
    hcy = os.path.join(root, "hcy")
    output = os.path.join(root, "output")
    os.makedirs(os.path.join(hcy, "input"), exist_ok=True)

    for report_id in REPORTS:
        with open(os.path.join(hcy, "input", report_id + ".txt"), "w",
                  encoding="latin1") as handle:
            handle.write(report_text(report_id))

    _write_segments(os.path.join(hcy, "segmented_reports.csv"))
    _write_gold(os.path.join(hcy, GOLD_DIR))
    _write_frame(os.path.join(output, "hcy_deepdive_frame"))
    _write_phenobert(os.path.join(output, "baseline_phenobert", "hcy"))
    _write_autopcr(os.path.join(output, "baseline_autopcr_70b", "hcy"))
    _write_raghpo(os.path.join(output, "baseline_raghpo_70b", "hcy"))
    _write_phenojury(output)
    _write_tree(output)
    return {"root": root, "hcy_dir": hcy, "output_base": output,
            "gold_dir": os.path.join(hcy, GOLD_DIR),
            "frame_dir": os.path.join(output, "hcy_deepdive_frame")}


def _rows(path: str, fieldnames, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _jsonl(path: str, records) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_segments(path: str) -> None:
    rows = []
    for report_id, sentences in SEGMENTS.items():
        for idx, sentence in enumerate(sentences):
            rows.append({"patient_id": report_id, "sentence_idx": idx, "sentence": sentence})
    _rows(path, ["patient_id", "sentence_idx", "sentence"], rows)


# -- the curated ground truth --------------------------------------------------------

ANNOTATION_FIELDS = [
    "patient_id", "hpo_code", "hpo_name", "in_gold", "exclude_reason", "source", "status",
    "segment_idx", "trigger_word", "segment_text", "anchored", "anchor_how", "qualifiers",
    "note", "edited", "original_hpo_code", "key",
]

REPORT_FIELDS = ["patient_id", "in_cohort", "cohort_reason", "n_segments", "n_annotations",
                 "n_gold_terms", "difficulty", "report_labels"]


def _write_gold(directory: str) -> None:
    codes = _codes()
    gold = {
        "SYN001": [codes["C"], codes["D"], codes["F"]],
        "SYN002": [codes["C"]],
        "SYN003": [codes["C"]],
    }
    _rows(os.path.join(directory, "hcy_ground_truth_curated.csv"),
          ["patient_id", "hpo_codes"],
          [{"patient_id": r, "hpo_codes": ";".join(sorted(v))} for r, v in sorted(gold.items())])

    annotations = [
        # Placed on its own trigger word, in the segment the curator named.
        _annotation("SYN001", codes["C"], "Seizure", 1, "Anfälle", SEGMENTS["SYN001"][1]),
        # A qualifier that the ground truth column shows as a chip.
        _annotation("SYN001", codes["D"], "Tremor", 2, "Zittern", SEGMENTS["SYN001"][2],
                    qualifiers="implicit"),
        # No trigger word located: the unplaced case, which must be listed and not drawn.
        _annotation("SYN001", codes["F"], "Spasticity", None, "", ""),
        _annotation("SYN002", codes["C"], "Seizure", 0, "Anfälle", SEGMENTS["SYN002"][0]),
        _annotation("SYN003", codes["C"], "Seizure", 0, "Anfälle", SEGMENTS["SYN003"][0]),
    ]
    _rows(os.path.join(directory, "hcy_curated_annotations.csv"), ANNOTATION_FIELDS, annotations)

    _rows(os.path.join(directory, "hcy_curated_reports.csv"), REPORT_FIELDS, [
        {"patient_id": r, "in_cohort": 1, "cohort_reason": "original_gold",
         "n_segments": len(SEGMENTS[r]), "n_annotations": len(gold[r]),
         "n_gold_terms": len(gold[r]), "difficulty": "medium", "report_labels": ""}
        for r in sorted(gold)])

    _json(os.path.join(directory, "manifest.json"),
          {"dataset": "hcy", "date": "2026-01-01",
           "counts": {"n_reports_in_cohort": len(gold),
                      "n_gold_pairs": sum(len(v) for v in gold.values())},
           "inputs": {"curation_log_sha256_16": "0123456789abcdef"}})

    # Every report is evaluated in outer fold 0 of repetition 0, which is the only repetition that
    # contributes to a pooled prediction set -- see methods/phenojury.py.
    _rows(os.path.join(directory, "folds_hcy.csv"),
          ["repetition", "outer_fold", "inner_fold", "report_id", "stratum", "role"],
          [{"repetition": 0, "outer_fold": 0, "inner_fold": -1, "report_id": r,
            "stratum": 0, "role": "eval"} for r in sorted(gold)])


def _annotation(patient, code, name, segment_idx, trigger, segment_text, qualifiers="") -> dict:
    return {
        "patient_id": patient, "hpo_code": code, "hpo_name": name, "in_gold": 1,
        "exclude_reason": "", "source": "daphne", "status": "unruled",
        "segment_idx": "" if segment_idx is None else segment_idx,
        "trigger_word": trigger, "segment_text": segment_text,
        "anchored": 1 if trigger else 0, "anchor_how": "segment" if trigger else "",
        "qualifiers": qualifiers, "note": "", "edited": 0, "original_hpo_code": "",
        "key": "daphne|{}|{}".format(patient, code),
    }


def _write_frame(directory: str) -> None:
    selected = ["SYN001", "SYN002", "SYN003"]
    _json(os.path.join(directory, "frame.json"), {
        "generated": "2026-01-01",
        "params": {"seed": 1, "min_gold": 1, "metric": "f1"},
        "inputs": {"gold": "deadbeefdeadbeef"},
        "cell_sizes": {"pb_best": 1, "pb_worst": 1, "implicit": 1},
        "pools": {"pb_best": ["SYN002"], "pb_worst": ["SYN001"], "implicit": ["SYN003"]},
        "picks": {"pb_best": ["SYN002"], "pb_worst": ["SYN001"], "implicit": ["SYN003"]},
        "selected": selected,
        "trace": [], "duplicates": [], "unannotated": [],
    })
    _rows(os.path.join(directory, "pools.csv"),
          ["patient_id", "selected", "cell", "n_gold", "n_pred", "tp", "fp", "fn",
           "precision", "recall", "f1", "family", "lab_value", "implicit", "negated"],
          [{"patient_id": "SYN001", "selected": 1, "cell": "pb_worst", "n_gold": 3, "n_pred": 2,
            "tp": 1, "fp": 1, "fn": 2, "precision": 0.5, "recall": 0.333, "f1": 0.4,
            "family": 0, "lab_value": 0, "implicit": 1, "negated": 0},
           {"patient_id": "SYN002", "selected": 1, "cell": "pb_best", "n_gold": 1, "n_pred": 1,
            "tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "family": 0, "lab_value": 0, "implicit": 0, "negated": 0},
           {"patient_id": "SYN003", "selected": 1, "cell": "implicit", "n_gold": 1, "n_pred": 0,
            "tp": 0, "fp": 0, "fn": 1, "precision": "", "recall": 0.0, "f1": 0.0,
            "family": 0, "lab_value": 0, "implicit": 1, "negated": 0}])


# -- the five methods --------------------------------------------------------

def _offsets(report_id: str, phrase: str):
    """Where a phrase sits in the report as ``load_txt`` reads it -- the frame both taggers use."""
    text = report_text(report_id)
    at = text.find(phrase)
    return (at, at + len(phrase)) if at >= 0 else (None, None)


def _predictions(path: str, per_report: dict, gold: dict) -> None:
    """The predictions contract: per-term lines, then one authoritative summary line per report."""
    records = []
    for report_id in sorted(per_report):
        codes = sorted(per_report[report_id])
        for code in codes:
            records.append({"report_id": report_id, "hpo_id": code, "hpo_label": code,
                            "prediction": 1,
                            "ground_truth": 1 if code in gold.get(report_id, ()) else 0})
        records.append({"report_id": report_id, "summary": True, "predicted_set": codes,
                        "gold_set": sorted(gold.get(report_id, ()))})
    _jsonl(path, records)


def _ground_truth_sets() -> dict:
    codes = _codes()
    return {"SYN001": {codes["C"], codes["D"], codes["F"]},
            "SYN002": {codes["C"]}, "SYN003": {codes["C"]}}


def _write_phenobert(directory: str) -> None:
    codes = _codes()
    predicted = {"SYN001": {codes["C"], codes["A"]},
                 "SYN002": {codes["C"]}, "SYN003": {codes["C"]}}
    _predictions(os.path.join(directory, "phenobert_predictions.jsonl"), predicted, _ground_truth_sets())

    detections = []
    for report_id, phrase, code, negated in (
        ("SYN001", "Anfälle", codes["C"], False),
        # Detected and discarded as negated: the miss this artifact can explain and no other can.
        ("SYN001", "Zittern", codes["D"], True),
        ("SYN001", "Bewegungsstörung", codes["A"], False),
        ("SYN002", "Anfälle", codes["C"], False),
    ):
        start, end = _offsets(report_id, phrase)
        detections.append({"report_id": report_id, "hpo_id": code, "hpo_label": code,
                           "raw_hpo_id": code, "resolved": True, "phrase": phrase,
                           "start": start, "end": end, "score": 0.91, "negated": negated})
    _jsonl(os.path.join(directory, "phenobert_detections.jsonl"), detections)


def _write_autopcr(directory: str) -> None:
    codes = _codes()
    predicted = {"SYN001": {codes["C"], codes["D"]},
                 "SYN002": {codes["C"]}, "SYN003": {codes["C"]}}
    _predictions(os.path.join(directory, "autopcr_predictions.jsonl"), predicted, _ground_truth_sets())

    detections = []
    for report_id, phrase, code, route, score in (
        ("SYN001", "Anfälle", codes["C"], "dictionary", 1.0),
        ("SYN001", "Zittern", codes["D"], "llm", -1.0),
        ("SYN002", "Anfälle", codes["C"], "retrieval", 0.96),
    ):
        start, end = _offsets(report_id, phrase)
        detections.append({"report_id": report_id, "hpo_id": code, "hpo_label": code,
                           "raw_hpo_id": code, "resolved": True, "phrase": phrase,
                           "start": start, "end": end, "score": score, "linked_by": route})
    _jsonl(os.path.join(directory, "autopcr_detections.jsonl"), detections)

    # F appears on a menu and loses: "was a candidate and not chosen", not "never retrieved".
    _jsonl(os.path.join(directory, "autopcr_retrieved_segments.jsonl"), [
        {"report_id": "SYN001", "hpo_id": codes["D"], "hpo_label": codes["D"], "rank": 1,
         "text": "Zittern", "cosine_sim": 0.88, "slm_verdict": "Yes"},
        {"report_id": "SYN001", "hpo_id": codes["F"], "hpo_label": codes["F"], "rank": 2,
         "text": "Zittern", "cosine_sim": 0.86, "slm_verdict": "No"},
        {"report_id": "SYN001", "hpo_id": codes["C"], "hpo_label": codes["C"], "rank": 1,
         "text": "Anfälle", "cosine_sim": 1.0, "slm_verdict": "Yes"},
        {"report_id": "SYN002", "hpo_id": codes["C"], "hpo_label": codes["C"], "rank": 1,
         "text": "Anfälle", "cosine_sim": 0.96, "slm_verdict": "Yes"},
    ])


def _write_raghpo(directory: str) -> None:
    codes = _codes()
    # The alt id,: a true positive only if resolution works.
    predicted = {"SYN001": {codes["C_ALT"], codes["D"]},
                 "SYN002": {codes["C_ALT"]}, "SYN003": {codes["C"]}}
    _predictions(os.path.join(directory, "rag_hpo_predictions.jsonl"), predicted, _ground_truth_sets())

    _jsonl(os.path.join(directory, "rag_hpo_retrieved_segments.jsonl"), [
        {"report_id": "SYN001", "hpo_id": codes["C_ALT"], "hpo_label": "Seizures", "rank": 1,
         "text": REPORTS["SYN001"][1], "cosine_sim": 0.74, "slm_verdict": "Yes"},
        {"report_id": "SYN001", "hpo_id": codes["B"], "hpo_label": codes["B"], "rank": 2,
         "text": REPORTS["SYN001"][1], "cosine_sim": 0.61, "slm_verdict": "No"},
        {"report_id": "SYN001", "hpo_id": codes["D"], "hpo_label": codes["D"], "rank": 1,
         "text": REPORTS["SYN001"][2], "cosine_sim": 0.69, "slm_verdict": "Yes"},
        {"report_id": "SYN002", "hpo_id": codes["C_ALT"], "hpo_label": "Seizures", "rank": 1,
         "text": REPORTS["SYN002"][0], "cosine_sim": 0.81, "slm_verdict": "Yes"},
    ])


def _write_phenojury(output: str) -> None:
    codes = _codes()
    cell = os.path.join(output, "phenojury_generation_other_prompts", "hcy", PROMPT)

    # Who said what, per (model, report, sentence). Three jurors on C, two on D, two on F.
    votes = {
        "apertus": {"SYN001": {1: [codes["C"]], 2: [codes["D"]]}, "SYN002": {0: [codes["C"]]}},
        "deepseek": {"SYN001": {1: [codes["C"]], 2: [codes["D"], codes["F"]]},
                     "SYN002": {0: [codes["C"]]}},
        "llama": {"SYN001": {1: [codes["C"]], 2: [codes["F"]]}, "SYN002": {0: [codes["C"]]}},
        "medgemma": {"SYN001": {1: [codes["B"]]}, "SYN002": {0: [codes["C"]]}},
    }
    for model in JURORS:
        by_report = votes.get(model, {})
        _jsonl(os.path.join(cell, "detections_{}.jsonl".format(model)), [
            {"report_id": report_id, "model": model, "sentence_number": sent,
             "hpo_id": code, "count": 1}
            for report_id, sentences in sorted(by_report.items())
            for sent, terms in sorted(sentences.items())
            for code in terms
        ])
        _jsonl(os.path.join(cell, "llm_extractions_{}.jsonl".format(model)), [
            {"model": model, "patient_id": report_id, "sentence_number": sent,
             "sentence_text": SEGMENTS[report_id][sent],
             "llm_output": "\n".join(terms) if terms else "NONE"}
            for report_id, sentences in sorted(by_report.items())
            for sent, terms in sorted(sentences.items())
        ])

    protocol = os.path.join(output, "phenojury_protocol", "hcy")
    _json(os.path.join(protocol, "manifest.json"),
          {"cohort": "hcy", "selected_prompt": PROMPT, "selected_normaliser": NORMALISER,
           "n_reports": 3, "n_jurors": len(JURORS)})
    _rows(os.path.join(protocol, "tables", "s5_selection.csv"),
          ["k", "outer_fold", "repetition", "rule", "size", "subset", "unit"],
          [{"k": 2, "outer_fold": 0, "repetition": 0, "rule": "exact", "size": 4,
            "subset": "apertus;deepseek;llama;medgemma", "unit": "segment"},
           # A second repetition that must be ignored: only repetition 0 pools.
           {"k": 3, "outer_fold": 0, "repetition": 1, "rule": "closure_reduced", "size": 4,
            "subset": "apertus;deepseek;llama;medgemma", "unit": "report"}])

    predictions = os.path.join(output, "comparison_inputs", "phenojury_protocol",
                               "hcy", "predictions")
    # The full pool, as roster.py reads it -- chapter 6's row, not a forced-prompt pool.
    _rows(os.path.join(predictions, "full_pool.csv"),
          ["patient_id", "hpo_codes"],
          [{"patient_id": "SYN001",
            "hpo_codes": ";".join(sorted([codes["C"], codes["D"], codes["F"]]))},
           {"patient_id": "SYN002", "hpo_codes": codes["C"]},
           {"patient_id": "SYN003", "hpo_codes": codes["C"]}])


# -- TreePhenoRAG ------------------------------------------------------------

#: Per-node ``[margin, margin]`` over two retrieved segments, chosen so that at ``tau_prune=0.1``
#: (pooled P1 = max sigmoid) and ``tau_accept=0.9`` (lse_beta1) the re-run accepts
#: ``{C, M}``: A and B expand without being accepted, E does not expand so F is pruned behind it,
#: D is scored and falls short of acceptance, and M is accepted although it is not in the ground truth.
TREE_MARGINS = {
    "A": (1.5, 0.0),
    "B": (1.8, 0.0),
    "C": (4.0, -1.0),
    "D": (0.5, -1.0),
    "E": (-3.0, -3.0),
    "F": (-3.0, -3.0),
    "G": (-4.0, -4.0),
    "H": (-3.5, -3.5),
    "M": (2.5, 2.4),
}

#: Which sentence each rank was scored on, so a call can be shown beside the words it judged.
TREE_SENTENCES = {"SYN001": (1, 2), "SYN002": (0, 1), "SYN003": (0, 0)}

TAU_PRUNE = 0.1
TAU_ACCEPT = 0.9


def _write_tree(output: str) -> None:
    codes = _codes()
    directory = os.path.join(output, "treephenorag_protocol")
    calls = os.path.join(directory, "toy_calls.jsonl")

    records = []
    for report_id in sorted(REPORTS):
        sentences = TREE_SENTENCES[report_id]
        for name in sorted(TREE_MARGINS):
            for rank, margin in enumerate(TREE_MARGINS[name], start=1):
                sent = sentences[min(rank - 1, len(sentences) - 1)]
                # The driver records the two logits and the margin between them; ``ingest`` reads
                # all three, so a fixture that wrote only the margin would exercise a code path
                # The real cache never takes.
                logit_no = 10.0
                records.append({
                    "report_id": report_id, "hpo_id": codes[name], "ctx_type": "union",
                    "rank": rank, "sent_index": sent, "cosine_sim": 0.7 - 0.05 * rank,
                    "logit_yes": logit_no + float(margin), "logit_no": logit_no,
                    "logsumexp_all": logit_no + float(margin) + 1.0,
                    "margin": float(margin), "captured_mass": 0.42,
                    "verdict": "Yes" if margin > 0 else "No",
                    "wall_clock_s": 0.1, "n_prompt_tokens": 120,
                })
    _jsonl(calls, records)

    from hpo_extraction.treephenorag.stored_scores import ingest_score_cache

    ingest_score_cache(calls, os.path.join(directory, "cache_toy.npz"), ctx_type="union")
    os.remove(calls)

    _rows(os.path.join(directory, "tables", "selection_choices.csv"),
          ["repetition", "outer_fold", "retrieval_index", "pool_pr", "tau_prune", "pool_acc",
           "tau_accept", "S", "inner_f1", "n_eval"],
          [{"repetition": 0, "outer_fold": 0, "retrieval_index": "toy", "pool_pr": "P1",
            "tau_prune": TAU_PRUNE, "pool_acc": "lse_beta1", "tau_accept": TAU_ACCEPT,
            "S": 2, "inner_f1": 0.5, "n_eval": 3},
           {"repetition": 1, "outer_fold": 0, "retrieval_index": "toy", "pool_pr": "P2",
            "tau_prune": 0.9, "pool_acc": "P1", "tau_accept": 0.1, "S": 2,
            "inner_f1": 0.1, "n_eval": 3}])

    accepted = ";".join(sorted([codes["C"], codes["M"]]))
    _rows(os.path.join(directory, "predictions", "nested_cv_pooled.csv"),
          ["patient_id", "hpo_codes"],
          [{"patient_id": "SYN001", "hpo_codes": accepted},
           {"patient_id": "SYN002", "hpo_codes": accepted},
           {"patient_id": "SYN003", "hpo_codes": accepted}])


# -- the GSC+ cohort ---------------------------------------------------------

#: Two abstracts, one sentence list each. English and ASCII -- unlike HCY, whose umlauts are
#: essential. What this cohort is testing is the other decoder and the other ground truth, and an
#: abstract that needed latin1 would not be a GSC+ abstract.
GSC_REPORTS = {
    "GSC001": [
        "A boy with seizures and a fine tremor is reported.",
        "He also showed spasticity of the lower limbs.",
        "The parents are healthy.",
    ],
    "GSC002": [
        "A girl with recurrent seizures is described.",
        "No other abnormality was found.",
    ],
}

#: The GSC+ cohort's ground truth, in RAG-HPO's own shape: a code and their wording for it, no
#: position. Which rung of the placement ladder each one lands on is the point -- see the
#: module docstring.
GSC_RAGHPO_GOLD = {
    "GSC001": [("C", "seizures"), ("D", "fine tremor"),
               ("F", "reduced muscle bulk of the calves")],
    "GSC002": [("C", "recurrent seizures")],
}

#: The corpus's own mention-level annotation: ``(code alias, mention, offsets_agree)``. The
#: false flag writes offsets that do not spell their own mention, which is a real failure mode
#: (a differently staged corpus) and must fall back to locating the mention, not
#: underlining three arbitrary characters.
GSC_CORPUS_ANNOTATION = {
    "GSC001": [("C", "seizures", True), ("M", "spasticity", True)],
    "GSC002": [("C", "seizures", False)],
}


def gsc_text(report_id: str) -> str:
    """The abstract as ``load_gsc_reports`` reads it -- two paragraphs, so segment ranges are
    not contiguous and ``align_segments`` has to step over text belonging to no segment."""
    lines = GSC_REPORTS[report_id]
    if len(lines) < 3:
        return "\n".join(lines) + "\n"
    return " ".join(lines[:2]) + "\n\n" + " ".join(lines[2:]) + "\n"


def write_gsc(root: str) -> dict:
    """Write the GSC+ cohort under *root*. Returns the paths a :class:`Paths` needs."""
    codes = _codes()
    gsc = os.path.join(root, "gsc")
    raghpo = os.path.join(root, "raghpo")
    output = os.path.join(root, "output")

    for report_id in GSC_REPORTS:
        _write_text(os.path.join(gsc, "Text", report_id), gsc_text(report_id))
        _write_text(os.path.join(gsc, "Annotations", report_id),
                    _gsc_annotation_file(report_id, codes))
        # The junk sidecar a Windows checkout leaves beside every file. The corpus reader
        # filters it. If it ever stops, this fixture reads as four documents and says so.
        _write_text(os.path.join(gsc, "Text", report_id + ":Zone.Identifier"), "[ZoneTransfer]")

    _rows(os.path.join(raghpo, "annotations.csv"), ["doc_id", "hpo_id", "hpo_description"],
          [{"doc_id": report_id, "hpo_id": codes[alias], "hpo_description": description}
           for report_id, entries in sorted(GSC_RAGHPO_GOLD.items())
           for alias, description in entries])
    _write_text(os.path.join(raghpo, "document_ids.txt"),
                "\n".join(sorted(GSC_REPORTS)) + "\n")

    segments = os.path.join(output, "gsc_segmentation", "segmented_reports.csv")
    _rows(segments, ["patient_id", "sentence_idx", "sentence"],
          [{"patient_id": report_id, "sentence_idx": idx, "sentence": sentence}
           for report_id, sentences in sorted(GSC_REPORTS.items())
           for idx, sentence in enumerate(sentences)])

    _write_gsc_methods(output, codes)
    _write_gsc_frame(os.path.join(output, "gsc_compare_frame"))
    return {"root": root, "gsc_dir": gsc, "raghpo_dir": raghpo, "output_base": output,
            "segments": segments,
            "frame_dir": os.path.join(output, "gsc_compare_frame")}


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def _gsc_annotation_file(report_id: str, codes: dict) -> str:
    """``start:end\\tHP:#######\\tmention`` per line, the corpus's own format."""
    text = gsc_text(report_id)
    lines = []
    for alias, mention, agrees in GSC_CORPUS_ANNOTATION.get(report_id, ()):  # noqa: B007
        at = text.find(mention)
        if at < 0:
            continue
        start, end = (at, at + len(mention)) if agrees else (at + 3, at + 3 + len(mention))
        lines.append("{}:{}\t{}\t{}".format(start, end, codes[alias], mention))
    return "\n".join(lines) + "\n"


#: What each column predicts per abstract. Worked out so the four cells of the GSC+ frame are
#: present: on GSC001, C is found by all three frame methods, D by PhenoJury alone and F by
#: nobody. On GSC002, PhenoJury misses the one annotated term both others find. RAG-HPO predicts
#: The ALT id here too, so resolution is exercised on this cohort as well.
GSC_PREDICTIONS = {
    "phenobert": {"GSC001": ["C"], "GSC002": ["C"]},
    "autopcr": {"GSC001": ["C", "M"], "GSC002": ["C"]},
    "rag_hpo": {"GSC001": ["C_ALT"], "GSC002": ["C_ALT"]},
    "phenojury": {"GSC001": ["C", "D"], "GSC002": []},
    "tree": {"GSC001": ["C"], "GSC002": ["C"]},
}


def _gsc_ground_truth_sets(codes: dict) -> dict:
    return {report_id: {codes[alias] for alias, _d in entries}
            for report_id, entries in GSC_RAGHPO_GOLD.items()}


def _gsc_set(codes: dict, name: str, report_id: str) -> set:
    return {codes[alias] for alias in GSC_PREDICTIONS[name][report_id]}


def _gsc_offsets(report_id: str, phrase: str):
    text = gsc_text(report_id)
    at = text.find(phrase)
    return (at, at + len(phrase)) if at >= 0 else (None, None)


def _write_gsc_methods(output: str, codes: dict) -> None:
    gold = _gsc_ground_truth_sets(codes)
    reports = sorted(GSC_REPORTS)

    # -- PhenoBERT
    directory = os.path.join(output, "baseline_phenobert", "gsc")
    _predictions(os.path.join(directory, "phenobert_predictions.jsonl"),
                 {r: _gsc_set(codes, "phenobert", r) for r in reports}, gold)
    detections = []
    for report_id in reports:
        start, end = _gsc_offsets(report_id, "seizures")
        detections.append({"report_id": report_id, "hpo_id": codes["C"],
                           "hpo_label": "Seizure", "raw_hpo_id": codes["C"],
                           "resolved": True, "phrase": "seizures", "start": start,
                           "end": end, "score": 0.93, "negated": False})
    _jsonl(os.path.join(directory, "phenobert_detections.jsonl"), detections)

    # -- AutoPCR
    directory = os.path.join(output, "baseline_autopcr_70b", "gsc")
    _predictions(os.path.join(directory, "autopcr_predictions.jsonl"),
                 {r: _gsc_set(codes, "autopcr", r) for r in reports}, gold)
    detections = []
    for report_id, phrase, alias, route, score in (
            ("GSC001", "seizures", "C", "dictionary", 1.0),
            ("GSC001", "spasticity", "M", "llm", -1.0),
            ("GSC002", "seizures", "C", "retrieval", 0.97)):
        start, end = _gsc_offsets(report_id, phrase)
        detections.append({"report_id": report_id, "hpo_id": codes[alias],
                           "hpo_label": codes[alias], "raw_hpo_id": codes[alias],
                           "resolved": True, "phrase": phrase, "start": start, "end": end,
                           "score": score, "linked_by": route})
    _jsonl(os.path.join(directory, "autopcr_detections.jsonl"), detections)
    _jsonl(os.path.join(directory, "autopcr_retrieved_segments.jsonl"), [
        {"report_id": "GSC001", "hpo_id": codes["M"], "hpo_label": codes["M"], "rank": 1,
         "text": "spasticity", "cosine_sim": 0.84, "slm_verdict": "Yes"},
        {"report_id": "GSC001", "hpo_id": codes["F"], "hpo_label": codes["F"], "rank": 2,
         "text": "spasticity", "cosine_sim": 0.80, "slm_verdict": "No"},
    ])

    # -- RAG-HPO
    directory = os.path.join(output, "baseline_raghpo_70b", "gsc")
    _predictions(os.path.join(directory, "rag_hpo_predictions.jsonl"),
                 {r: _gsc_set(codes, "rag_hpo", r) for r in reports}, gold)
    _jsonl(os.path.join(directory, "rag_hpo_retrieved_segments.jsonl"), [
        {"report_id": report_id, "hpo_id": codes["C_ALT"], "hpo_label": "Seizures",
         "rank": 1, "text": GSC_REPORTS[report_id][0], "cosine_sim": 0.78,
         "slm_verdict": "Yes"}
        for report_id in reports])

    # -- PhenoJury: the jurors, the protocol run, and the folds beside the predictions
    cell = os.path.join(output, "phenojury_generation_other_prompts", "gsc", PROMPT)
    votes = {
        "apertus": {"GSC001": {0: ["C", "D"]}},
        "deepseek": {"GSC001": {0: ["C", "D"]}},
        "llama": {"GSC001": {0: ["C"]}, "GSC002": {0: ["B"]}},
    }
    for model in JURORS:
        by_report = votes.get(model, {})
        _jsonl(os.path.join(cell, "detections_{}.jsonl".format(model)), [
            {"report_id": report_id, "model": model, "sentence_number": sent,
             "hpo_id": codes[alias], "count": 1}
            for report_id, sentences in sorted(by_report.items())
            for sent, aliases in sorted(sentences.items())
            for alias in aliases])
        _jsonl(os.path.join(cell, "llm_extractions_{}.jsonl".format(model)), [
            {"model": model, "patient_id": report_id, "sentence_number": sent,
             "sentence_text": GSC_REPORTS[report_id][sent],
             "llm_output": ", ".join(aliases) if aliases else "NONE"}
            for report_id, sentences in sorted(by_report.items())
            for sent, aliases in sorted(sentences.items())])

    protocol = os.path.join(output, "phenojury_protocol", "gsc")
    _json(os.path.join(protocol, "manifest.json"),
          {"cohort": "gsc", "selected_prompt": PROMPT, "selected_normaliser": NORMALISER,
           "n_reports": len(reports), "n_jurors": len(JURORS)})
    _rows(os.path.join(protocol, "tables", "s5_selection.csv"),
          ["k", "outer_fold", "repetition", "rule", "size", "subset", "unit"],
          [{"k": 2, "outer_fold": 0, "repetition": 0, "rule": "exact", "size": 3,
            "subset": "apertus;deepseek;llama", "unit": "segment"}])

    inputs = os.path.join(output, "comparison_inputs", "phenojury_protocol", "gsc")
    _rows(os.path.join(inputs, "predictions", "full_pool.csv"),
          ["patient_id", "hpo_codes"],
          [{"patient_id": r, "hpo_codes": ";".join(sorted(_gsc_set(codes, "phenojury", r)))}
           for r in reports])
    # Beside the predictions, which is where the reader looks first: the folds have to be the
    # ones this pooled set was pooled over, and two the PhenoJury protocol runs both write a folds_gsc.csv.
    _rows(os.path.join(inputs, "folds_gsc.csv"),
          ["repetition", "outer_fold", "inner_fold", "report_id", "stratum", "role"],
          [{"repetition": 0, "outer_fold": 0, "inner_fold": -1, "report_id": r,
            "stratum": 0, "role": "eval"} for r in reports])

    # -- TreePhenoRAG: a transferred prediction set and nothing else. See the docstring.
    _rows(os.path.join(output, "treephenorag_protocol", "predictions",
                       "gsc_raghpo_ann_transferred.csv"),
          ["patient_id", "hpo_codes"],
          [{"patient_id": r, "hpo_codes": ";".join(sorted(_gsc_set(codes, "tree", r)))}
           for r in reports])


def _write_gsc_frame(directory: str) -> None:
    """A two-document draw in the shape ``apps/compare_ui/select_gsc_documents.py`` writes.

    Written out, not drawn, because the draw is that script's own selftest to check.
    What this one has to exercise is the *reading* side: a frame whose cells are named after
    the three-method agreement pattern, with two of the four cells empty.
    """
    _json(os.path.join(directory, "frame.json"), {
        "generated": "2026-01-02",
        "cohort": "gsc",
        "scored_cohort": "gsc_raghpo_ann",
        "params": {"seed": 1, "resolved": True,
                   "methods": {"pj": "phenojury", "rag": "raghpo_70b",
                               "apc": "autopcr_70b"}},
        "inputs": {"raghpo_annotations": "feedfacefeedface"},
        "cell_sizes": {"all_right": 1, "pj_only": 1, "pj_wrong": 1, "all_wrong": 1},
        "pools": {"all_right": ["GSC001"], "pj_only": ["GSC001"],
                  "pj_wrong": ["GSC002"], "all_wrong": ["GSC001"]},
        "picks": {"all_right": ["GSC001"], "pj_only": [], "pj_wrong": ["GSC002"],
                  "all_wrong": []},
        "selected": ["GSC001", "GSC002"],
        "tier_of": {"GSC001": "strict", "GSC002": "strict"},
        "trace": [],
    })


# -- wiring ------------------------------------------------------------------

def paths_for(written: dict):
    """A :class:`~apps.compare_ui.sources.Paths` over what :func:`write` produced."""
    from apps.compare_ui import sources

    return sources.Paths(output_base=written["output_base"], hcy_dir=written["hcy_dir"],
                         gold_dir=written["gold_dir"], frame_dir=written["frame_dir"])


def context_for(written: dict):
    """A loaded :class:`~apps.compare_ui.sources.Context`, with the toy ontology injected.

    The real ``registry.get_view`` builds an ``HPOTree``, which imports nltk and stanza at module
    scope -- absent on most machines this will run on. Injecting the toy view is the seam that lets
    the whole builder be exercised anywhere, and ``registry.set_shared`` puts the same view behind
    the tree adapter's traversal graph so the two cannot disagree about the ontology.
    """
    from apps.compare_ui import sources
    from apps.treephenorag_ui import pruning

    view = toy_view()
    children_map, roots = toy_ontology().toy_children_map()
    sources.set_shared(
        view=view, graph=(children_map, roots, pruning.bfs_depths(children_map, roots)))
    return sources.build_context(paths_for(written), view=view)


def build(written: dict):
    """``(context, bundles)`` -- the HCY fixture cohort, built. Used by the selftests and pytest."""
    from apps.compare_ui import build as build_mod, sources as sources_mod

    ctx = context_for(written)
    frame = sources_mod.frame_of(paths_for(written))
    report_ids = list((frame or {}).get("selected") or sorted(REPORTS))
    cells = dict((frame or {}).get("cell_of") or {})
    return ctx, build_mod.build_all(ctx, report_ids, cells)


def paths_for_gsc(written: dict):
    """A :class:`~apps.compare_ui.sources.Paths` over what :func:`write_gsc` produced."""
    from apps.compare_ui import cohorts, sources

    return sources.Paths(output_base=written["output_base"], cohort=cohorts.GSC.key,
                         gsc_dir=written["gsc_dir"], raghpo_dir=written["raghpo_dir"],
                         segments=written["segments"], frame_dir=written["frame_dir"])


def build_gsc(written: dict):
    """``(context, bundles)`` -- the GSC+ fixture cohort, built."""
    from apps.compare_ui import build as build_mod, sources as sources_mod
    from apps.treephenorag_ui import pruning

    view = toy_view()
    children_map, roots = toy_ontology().toy_children_map()
    sources_mod.set_shared(
        view=view, graph=(children_map, roots, pruning.bfs_depths(children_map, roots)))

    paths = paths_for_gsc(written)
    ctx = sources_mod.build_context(paths, view=view)
    frame = sources_mod.frame_of(paths)
    report_ids = list((frame or {}).get("selected") or sorted(GSC_REPORTS))
    cells = dict((frame or {}).get("cell_of") or {})
    return ctx, build_mod.build_all(ctx, report_ids, cells)
