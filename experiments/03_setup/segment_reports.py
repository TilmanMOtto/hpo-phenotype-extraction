"""Segmentation-only utility: run the pipeline's own sentence split and write it to a CSV.

The CSV this writes **is** the coordinate system every downstream surface indexes into.
``sent_index`` in a tree score cache, ``sentence_number`` in a juror's generation and
``segment_idx`` in the curated ground truth are one integer with three names, and that integer is a
position in the list ``split_sents(segment_dict(...))`` returns. So a second, cheaper sentence
splitter is not an option anywhere: it would renumber every recorded position and attribute
real decisions to the wrong sentences.

Two corpora, and they differ only in how the text is read -- which is the part that
cannot be guessed. HCY reports are German and go through ``hpo_extraction.data.loading.load_txt``, which
decodes **latin1**; GSC+ abstracts go through ``hpo_extraction.evaluation.datasets.gsc.load_gsc_reports``,
which decodes **UTF-8** from files that have no extension and skips the Windows
``*:Zone.Identifier`` sidecars that would otherwise double the corpus. Both loaders are
imported rather than retyped.

Usage::

    python experiments/03_setup/segment_reports.py --corpus hcy \
        --input_dir /path/to/patient_texts \
        --stanza_dir /path/to/stanza_resources \
        --output_path /path/to/hcy/segmented_reports.csv

    python experiments/03_setup/segment_reports.py --corpus gsc \
        --input_dir $REPO/resources/data/GSC_2024 \
        --stanza_dir /path/to/stanza_resources \
        --output_path $REPO/output/gsc_segmentation/segmented_reports.csv

For GSC+, ``--input_dir`` is the corpus root (the directory holding ``Text/``), not a
directory of text files.
"""

import argparse
import csv
import logging
from pathlib import Path

# Ensure src/ is importable when running as a script (before pip install -e .)

from hpo_extraction.data.segmentation import load_stanza, segment_dict, split_sents
from hpo_extraction.paths import lookup

#: ``{corpus: (loader, what --input_dir means)}``. Adding a third corpus is a line here.
CORPORA = {
    "hcy": ("core.data_loading.load_txt (latin1)", "a directory of .txt reports"),
    "gsc": ("evaluation.datasets.gsc.load_gsc_reports (UTF-8)",
            "the GSC+ corpus root, which holds Text/"),
}


def load_corpus(corpus: str, input_dir: str) -> dict:
    """``{document_id: text}``, read the way that corpus's drivers read it."""
    if corpus == "hcy":
        from hpo_extraction.data.loading import load_txt

        return load_txt(input_dir)
    from hpo_extraction.evaluation.datasets.gsc import load_gsc_reports

    return load_gsc_reports(input_dir)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Segment a corpus with the Stanza clinical tokenizer and write ``segmented_reports.csv``."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="hcy", choices=sorted(CORPORA),
                        help="which corpus, and therefore how the text is decoded")
    parser.add_argument("--input_dir", required=True,
                        help="for hcy a directory of .txt reports; for gsc the corpus root")
    parser.add_argument("--stanza_dir", required=True, help="Path to Stanza resources")
    parser.add_argument(
        "--output_path",
        default=lookup("hcy.segmented_reports"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader, means = CORPORA[args.corpus]
    logger.info("Loading %s texts from %s (%s; --input_dir is %s)",
                args.corpus, args.input_dir, reader, means)
    text_dict = load_corpus(args.corpus, args.input_dir)
    if not text_dict:
        raise SystemExit("no documents under {} -- for --corpus gsc this must be the corpus "
                         "root holding Text/".format(args.input_dir))
    logger.info("Loaded %d documents", len(text_dict))

    logger.info("Loading Stanza tokenizer from %s", args.stanza_dir)
    nlp = load_stanza(stanza_dir=args.stanza_dir, mode="TOKENIZER")

    logger.info("Segmenting reports...")
    segmented = segment_dict(text_dict, nlp)
    segmented = split_sents(segmented)

    total_sentences = sum(len(sents) for sents in segmented.values())
    logger.info(
        "Segmentation complete: %d documents, %d sentences total",
        len(segmented),
        total_sentences,
    )

    logger.info("Writing CSV to %s", output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "sentence_idx", "sentence"])
        for patient_id in sorted(segmented.keys()):
            for idx, sentence in enumerate(segmented[patient_id]):
                writer.writerow([patient_id, idx, sentence])

    logger.info("Done. Wrote %d rows to %s", total_sentences, output_path)


if __name__ == "__main__":
    main()
