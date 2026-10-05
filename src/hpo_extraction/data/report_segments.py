"""Split reports into the segments both methods read.

The thesis segments every report with the Stanza clinical tokenizer (``mimic`` package), splits the
sentences again at line breaks and drops segments of three or fewer characters
(:func:`hpo_extraction.data.segmentation.split_sents`). :class:`Segmenter` does the same when it is
given a Stanza resource directory. Without one it uses a simple rule-based sentence splitter
followed by the same line-break split and length filter. That fallback needs no model download and
is meant for quick trials. Results reported in the thesis always used Stanza.
"""
from __future__ import annotations

import logging
import re

from hpo_extraction.data.segmentation import split_sents

logger = logging.getLogger(__name__)

# A sentence ends at ., ! or ? followed by white space and an upper-case letter or a digit.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class Segmenter:
    """Segment report texts.

    Args:
        stanza_dir: directory holding the Stanza English resources with the ``mimic`` package.
            ``None`` selects the rule-based fallback.
    """

    def __init__(self, stanza_dir: str | None = None):
        self.stanza_dir = stanza_dir
        self._nlp = None
        if stanza_dir is None:
            logger.warning("no Stanza directory given: using the rule-based sentence splitter, "
                           "which differs from the segmentation used for the thesis results")

    def segment(self, texts: dict[str, str]) -> dict[str, list[str]]:
        """Segments of every report.

        Args:
            texts: ``{report_id: text}``.

        Returns:
            ``{report_id: [segment, ...]}`` in reading order.
        """
        if self.stanza_dir is not None:
            from hpo_extraction.data.segmentation import load_stanza, segment_dict

            if self._nlp is None:
                self._nlp = load_stanza(stanza_dir=self.stanza_dir, mode="TOKENIZER")
            return split_sents(segment_dict(texts, self._nlp))
        sentences = {rid: [s for s in _SENTENCE_END.split(text) if s.strip()]
                     for rid, text in texts.items()}
        return split_sents(sentences)
