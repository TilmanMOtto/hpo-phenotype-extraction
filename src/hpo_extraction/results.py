"""Output format shared by the TreePhenoRAG and PhenoJury applications.

Both applications return one :class:`ReportResult` per report: the HPO terms predicted for it,
each with a score and the segments of the report that support it. ``write_jsonl`` stores a list
of results as JSON Lines, one report per line, in the layout ``docs/data_formats.md`` describes.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class Evidence:
    """One segment of the report that supports a predicted term.

    Attributes:
        segment_index: position of the segment in the report's segmentation, starting at 0.
        text: the segment text.
        score: method-specific support of this segment. For TreePhenoRAG it is the verifier's
            probability of *Yes* for this segment, between 0 and 1. For PhenoJury it is the number
            of jurors that named the term in this segment, between 1 and the jury size.
    """

    segment_index: int
    text: str
    score: float


@dataclass
class TermScore:
    """One predicted HPO term.

    Attributes:
        hpo_id: HPO identifier, for example ``HP:0001250``.
        label: the term's label in the ontology file.
        score: method-specific score between 0 and 1. For TreePhenoRAG it is the pooled acceptance
            score, which was compared with the acceptance threshold. For PhenoJury it is the share
            of jurors that support the term (votes divided by jury size).
        evidence: supporting segments, most supportive first.
    """

    hpo_id: str
    label: str
    score: float
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class ReportResult:
    """All predictions for one report.

    Attributes:
        report_id: identifier of the report (the file name without extension when read from a
            folder).
        terms: predicted terms, highest score first.
        n_segments: number of segments the report was split into.
        n_model_calls: language-model calls spent on this report (verifier calls for
            TreePhenoRAG, juror generations for PhenoJury).
        method: ``"treephenorag"`` or ``"phenojury"``.
        settings: the configuration values the prediction was made with.
    """

    report_id: str
    terms: list[TermScore]
    n_segments: int
    n_model_calls: int
    method: str
    settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """The result as plain Python types, ready for ``json.dumps``."""
        return asdict(self)


def write_jsonl(results: Iterable[ReportResult], path: str | Path) -> Path:
    """Write results as JSON Lines (one report per line) and return the path.

    Args:
        results: the results to write.
        path: output file. Parent folders are created.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_reports(source: str | Path, encoding: str = "utf-8") -> dict[str, str]:
    """Read reports from a ``.txt`` file or from a folder of ``.txt`` files.

    Args:
        source: a single text file, or a folder whose ``*.txt`` files are read in name order.
        encoding: text encoding of the files. The HCY reports on the cluster are latin-1, which
            is how the thesis scripts read them (``hpo_extraction.data.loading.load_txt``).

    Returns:
        ``{report_id: text}``, where the report id is the file name without ``.txt``.
    """
    source = Path(source)
    files = [source] if source.is_file() else sorted(source.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt report found at {source}")
    return {p.stem: p.read_text(encoding=encoding) for p in files}
