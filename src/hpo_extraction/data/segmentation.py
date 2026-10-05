"""Sentence segmentation using Stanza NLP pipelines."""

from typing import TYPE_CHECKING

import stanza

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, runtime import is inside the function
    from hpo_extraction.phenojury.span_detection import Sentence


def load_stanza(stanza_dir: str, mode: str = "TOKENIZER") -> stanza.Pipeline:
    """
    Load a Stanza pipeline for sentence tokenization or NER.

    Args:
        stanza_dir: Path to the directory containing Stanza resources.
        mode: "TOKENIZER" for sentence splitting only, "NER" for i2b2 NER.

    Returns:
        Configured stanza.Pipeline.
    """
    if mode == "NER":
        processors = {"ner": "i2b2"}
    elif mode == "TOKENIZER":
        processors = "tokenize"
    else:
        raise ValueError(f"mode must be 'NER' or 'TOKENIZER', got '{mode}'")

    return stanza.Pipeline(
        lang="en",
        package="mimic",
        processors=processors,
        dir=stanza_dir,
        download_method=None,
        verbose=True,
    )


def segment_dict(text_dict: dict[str, str], stanza_pipeline: stanza.Pipeline) -> dict[str, list[str]]:
    """
    Segment each text in text_dict into sentences using stanza_pipeline.

    Returns:
        {id → [sentence_strings]}
    """
    sent_dict: dict[str, list[str]] = {}
    for key, text in text_dict.items():
        seg_output = stanza_pipeline(text)
        sent_dict[key] = [sentence.text for sentence in seg_output.sentences]
    return sent_dict


def split_sents(segment_dict: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Split sentences at newline characters and filter out very short segments (≤3 chars).

    Returns:
        {id → [filtered_sentence_strings]}
    """
    split_sent_dict: dict[str, list[str]] = {}
    for key, sent_list in segment_dict.items():
        split_list = [s.split("\n") for s in sent_list]
        flattened = [sub for item in split_list for sub in item]
        split_sent_dict[key] = [s for s in flattened if len(s) > 3]
    return split_sent_dict


def segment_with_offsets(
    text_dict: dict[str, str], stanza_pipeline: stanza.Pipeline
) -> dict[str, list["Sentence"]]:
    """Segment into sentences that carry their character offsets into the source text.

    ``segment_dict`` keeps only ``sentence.text`` and throws Stanza's ``start_char``/``end_char``
    away; ``split_sents`` then splits on newlines and drops short fragments, so even index-based
    back-mapping is gone by the end. Every offset-bearing thing this project has wanted, a
    mention-level metric, the HCY half of ``an earlier exploratory run``'s fate ladder, the whole earlier record schema, has been blocked on that.

    This function does **not** re-segment. It runs the identical legacy chain
    (``split_sents(segment_dict(...))``) and then evidence locations the resulting strings, so the sentence at
    index *i* here is byte-for-byte the sentence at index *i* there. That counts concretely: the
    ``sent_index`` field of every ``*_calls.jsonl`` on disk, and ``result_tables/segments.py``'s
    re-derivation, both index that exact list. Offsets are therefore *added* to the existing
    coordinate space rather than defining a competing one.

    Raises:
        hpo_extraction.phenojury.span_detection.OffsetError: if a sentence is not a verbatim slice of its report,
            naming the report. That cannot happen while the chain above only slices and drops, so
            it firing means the chain changed and every offset downstream is void.
    """
    from hpo_extraction.phenojury.span_detection import OffsetError, location_sentences

    sent_texts = split_sents(segment_dict(text_dict, stanza_pipeline))
    out: dict[str, list["Sentence"]] = {}
    for report_id, sents in sent_texts.items():
        try:
            out[report_id] = location_sentences(text_dict[report_id], sents)
        except OffsetError as exc:
            raise OffsetError(f"{report_id}: {exc}") from exc
    return out
