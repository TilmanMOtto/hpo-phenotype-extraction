"""The read-only inputs, loaded once per process.

==================  ======================================================================
segments            ``segmented_reports.csv``, the Stanza segmentation every screen is laid out on
``prior_annotation``        ``hcy_holistic_ground_truth.csv``, one row per *annotation*: code, trigger
                    word, the sentence it sits in, a character offset, and the verbatim report
``daphne``          ``annotations_confirmed.csv``, one row per annotation, already located to a
                    segment index, carrying a provenance and a confirmed flag
``prior_annotation_2``           ``hcy_ground_truth_marc2.csv``, two columns, code-only, the second annotator
PhenoBERT           ``baseline_phenobert/<cohort>/``, read through :mod:`hpo_extraction.curation.phenobert_output`
==================  ======================================================================

The first two replaced ``hcy_ground_truth_raw.csv``, and the difference is the whole point: a
code-only file records that ``SYN004`` has ``HP:0001250`` and nothing else, so every question since
asked of it, is this term right, did the model have a chance, is this a labelling error or a
retrieval failure, needed a lexical guess to even ask. These two carry the evidence, so the screen
can *draw* the annotation on the sentence it came from rather than offering a candidate origin.

``prior_annotation_2`` stays code-only, so it keeps the guessing path. That path is now the exception, not
the rule, which is what :mod:`locate` exists for.

The three are also not equals. ``daphne`` is a **confirmed pass over the prior_annotation set**, so it is
what Approve mode rules on. Where it dropped a term the prior_annotation file still carries, the prior_annotation
row stands in its place, because that drop is itself a judgement somebody has to confirm. ``prior_annotation_2``
is a second annotator's code list, a cross-check on every row, never a row of its own. That
ordering is :data:`ADJUDICATION_ORDER` and :func:`adjudicated`, and it is data, not a branch
in each panel.

Everything a rich source yields is an **annotation record**, the one shape the rest of the app
reads:

.. code-block:: python

    {"source": "prior_annotation", "patient_id": "SYN004", "hpo_code": "HP:0001250",
     "hpo_name": "Seizure", "trigger_word": "convulsions",
     "segment_idx": None,          # daphne carries one. Prior_annotation does not
     "char_offset": 300,           # into ``report_text``. Prior_annotation only
     "sentence_context": "…she had convulsions starting after…",
     "confirmed": True, "provenance": "manual", "slot": None}

``slot`` disambiguates a code a source carries **twice** for one patient, two triggers for one
phenotype is a normal annotation, and a curator keeping one must not implicitly keep the other. The
first occurrence gets ``None`` so its key stays ``prior_annotation|SYN004|HP:0001250``. Only the second and
later ones get an index, so adding an occurrence never re-keys the one already adjudicated.

Two of these are shared code paths, not lookalikes, on purpose:

* the code-cell parser is the one from ``scripts/generate_target_symptoms_all.py``, a ground truth file
  may hold a Python list literal, a semicolon list or a comma list, and a UI that guessed
  differently from the scorer would show a term set nothing else in the repo agrees with;
* PhenoBERT is read by ``pbstandalone.load_detections`` / ``load_texts``, which already yield the
  offsets, the ``negated`` flag and the verbatim staged report. Reimplementing that would be a
  second parser of the same files, free to drift.

Everything here is stdlib + pandas. No torch, no stanza, no mlflow: this app runs on a login node.
"""

from __future__ import annotations

import ast
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

#: Where the cluster keeps the cohort. Every path below is derived from it unless overridden.
from hpo_extraction.paths import lookup as _lookup  # noqa: E402

DEFAULT_HCY_DIR = _lookup("hcy.dir")
DEFAULT_PB_DIR = (
    os.path.join(_lookup("results_dir"), "baseline_phenobert", "hcy")
)

SEGMENTS_FILE = "segmented_reports.csv"
HOLISTIC_FILE = "hcy_holistic_ground_truth.csv"
CONFIRMED_FILE = "annotations_confirmed.csv"
MARC2_FILE = "hcy_ground_truth_marc2.csv"

#: The annotation sources, in the order every screen lists them.
#:
#: ``rich``        the file carries its own evidence, a trigger word and a position, so its
#:                 annotations are *drawn on the report*. A source that is not rich hands its codes
#:                 to :mod:`locate` for a candidate origin instead.
#: ``precedence``  where the source sits in the **adjudication order**, or ``None`` if it is
#:                 reference only. See :func:`adjudicated`.
GOLD_SOURCES: tuple[dict, ...] = (
    {"id": "prior_annotation", "path_key": "prior_annotation", "file": HOLISTIC_FILE, "rich": True,
     "precedence": 1, "label": "hcy_holistic_ground_truth.csv"},
    {"id": "daphne", "path_key": "confirmed", "file": CONFIRMED_FILE, "rich": True,
     "precedence": 0, "label": "annotations_confirmed.csv"},
    {"id": "prior_annotation_2", "path_key": "prior_annotation_2", "file": MARC2_FILE, "rich": False,
     "precedence": None, "label": "hcy_ground_truth_marc2.csv"},
)

GOLD_SOURCE_IDS = tuple(spec["id"] for spec in GOLD_SOURCES)

#: The adjudication order, best first. ``daphne`` is the confirmed pass over the holistic set, so
#: where it has an annotation for a code, that annotation *is* the one to rule on. Where it has
#: none, the holistic annotation stands in its place. ``marc2`` is in neither position, it is a
#: second annotator's code list, useful as a cross-check and not a thing to pass a verdict on.
ADJUDICATION_ORDER = tuple(
    spec["id"] for spec in sorted(
        (s for s in GOLD_SOURCES if s["precedence"] is not None),
        key=lambda s: s["precedence"],
    )
)


def precedence(source: str):
    """Where *source* sits in the adjudication order, or ``None`` if it is reference only."""
    for spec in GOLD_SOURCES:
        if spec["id"] == source:
            return spec["precedence"]
    return None


def adjudicated(records, loaded=None) -> list[dict]:
    """The records Approve mode rules on: **per code, the best source that has one**.

    ``daphne`` is a confirmed pass over the prior_annotation set, so a term it carries is a term somebody
    has already looked at once and the prior_annotation row for the same code is that row's predecessor, not
    a second annotation. Showing both would ask the curator the same question twice and let the two
    answers disagree.

    Where ``daphne`` dropped a term the prior_annotation file still carries, the prior_annotation annotation takes
    its place. That is the case that most needs adjudicating, the drop is itself a judgement
    somebody made, and it has to be confirmable, not invisible.

    Everything else, ``prior_annotation_2``, PhenoBERT, is **reference**. It is still drawn on the report, still
    listed in the document panel and still counted in the agreement label on every row, because
    knowing that a second annotator also has this code is what makes a verdict easy. What it
    is not is a row to rule on, because nothing about it would enter the curated ground truth.

    Order is the input's, so a screen built from this reads in file order, not in whatever
    order the codes happened to group.
    """
    loaded = set(loaded) if loaded is not None else None
    best: dict[str, int] = {}
    for record in records:
        rank = precedence(record.get("source", ""))
        if rank is None or (loaded is not None and record["source"] not in loaded):
            continue
        code = record["hpo_code"]
        if code not in best or rank < best[code]:
            best[code] = rank
    return [
        record for record in records
        if precedence(record.get("source", "")) is not None
        and (loaded is None or record["source"] in loaded)
        and precedence(record["source"]) == best.get(record["hpo_code"])
    ]


# ──────────────────────────────────────────────────────────────────────────────
# HPO code cells
# ──────────────────────────────────────────────────────────────────────────────
def parse_hpo_codes(raw: str) -> list[str]:
    """Parse one ground truth code cell: list literal, semicolon-separated, or comma-separated.

    Byte-for-byte the rule in ``scripts/generate_target_symptoms_all.py``. Kept identical so the
    term set shown here is the term set the scorer sees.
    """
    raw = str(raw).strip()
    if not raw or raw.lower() == "nan":
        return []
    try:
        evaluated = ast.literal_eval(raw)
        if isinstance(evaluated, list):
            return [str(c).strip() for c in evaluated if str(c).strip()]
        return [str(evaluated).strip()]
    except (ValueError, SyntaxError):
        sep = ";" if ";" in raw else ","
        return [c.strip() for c in raw.split(sep) if c.strip()]


def fix_hpo_id(code: str) -> str:
    """``HP0001234`` → ``HP:0001234``. Some annotation exports drop the colon."""
    if ":" not in code and len(code) > 2:
        return code[:2] + ":" + code[2:]
    return code


# ──────────────────────────────────────────────────────────────────────────────
# small cell readers, every one of these tolerates a missing or empty column
# ──────────────────────────────────────────────────────────────────────────────
def _text(row, column: str) -> str:
    """One cell as a stripped string. A missing column and an empty cell read the same."""
    if column not in row:
        return ""
    value = row[column]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _int_or_none(row, column: str):
    """One cell as an ``int``, or ``None``. ``"300 "``, ``300.0`` and ``""`` all behave."""
    text = _text(row, column)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _flag(row, column: str):
    """A truthy cell as ``bool``, or ``None`` when the column is absent or empty.

    loose: the confirmed column has been seen holding ``True``, ``TRUE``, ``1`` and
    ``True.``, a trailing full stop from a spreadsheet, and reading that last one as *unconfirmed*
    would silently demote the whole file.
    """
    text = _text(row, column).lower().rstrip(".")
    if not text:
        return None
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    return None


def _claim(record: dict) -> tuple:
    """The ``(segment, trigger)`` a record's file *states*, normalised for display.

    Whitespace and case are collapsed on the trigger because the two files were written by different
    pipelines and neither preserved the other's. ``None`` means the file said nothing, and note
    that segment ``0`` is a claim, so this tests against ``None``, not for truthiness.

    A claim is **not** what the merge compares. See :func:`merge_identical`.
    """
    segment = record.get("segment_idx")
    try:
        segment = None if segment is None else int(segment)
    except (TypeError, ValueError):
        segment = None
    trigger = " ".join(str(record.get("trigger_word") or "").split()).lower() or None
    return segment, trigger


def same_annotation(one, other) -> bool:
    """Are these two rows the same annotation, given where each one actually sits?

    Both arguments are **spans**, ``(segment, start, end)`` for a row that sits on words in the
    report, or ``None`` for one that does not.

    Two annotations are different only when both of them landed, on different words. That is the
    real case, and it is worth keeping apart: ``developmental delay`` and ``delay`` in one report
    are two claims about two stretches of text, and merging them would lose one.

    Everything else is one annotation. A row whose trigger word occurs nowhere, or whose segment
    index belongs to a different segmentation run, has told us **nothing about which words**, so it
    cannot disagree with anything. Two such rows for one code are not two annotations. They are two
    broken records of one, and the curator has one thing to do about them, not two. And a broken row
    beside one that *did* land is the same annotation recorded twice, once badly, which is
    the row that should stop being asked about.

    Comparing the files' *claims* instead, which this used to do, got the first case right and
    both of the others wrong: two failed attempts naming different words looked like two different
    annotations, because neither of the names worked.
    """
    return one is None or other is None or one == other


def _richness(span, record: dict) -> tuple:
    """Sort key: rows that landed first, then the most complete claim, then the best source.

    The host of a merge is therefore an annotation that is already located wherever one exists,
    which is what makes a broken duplicate fold into the working row, not the other way
    round.
    """
    segment, trigger = _claim(record)
    rank = precedence(record.get("source", ""))
    return (span is None, trigger is None, segment is None, rank is None, rank or 0)


def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def merge_identical(records, spans_by_key, mergeable=None) -> list[dict]:
    """One row per *annotation*, where the files carried the same one more than once.

    ``prior_annotation`` and ``daphne`` overlap by design, the confirmed set is a pass over the
    prior_annotation file, so a term both carry is one annotation with two rows behind it. Listing both is
    right on a screen about *provenance* and wrong on a screen about *work*: it asks a curator to
    place the same phenotype twice, and the second time there is nothing left to place.

    *spans_by_key* says where each row **actually sits**: ``{key: (segment, start, end)}``, with a
    key absent or ``None`` for a row that sits nowhere. It is the placement of the file's claim, or
    this app's own evidence location where a curator has since supplied one, see ``reader.realised_spans``.
    Rows for one code are folded together whenever those spans do not conflict
    (:func:`same_annotation`), rows that landed first, and the merged row **keeps the most complete
    value for every field**: a segment index from the row that had one, a trigger word from the row
    that had one, the sentence context from whichever file recorded it.

    Nothing is silently dropped. Where a folded row's own claim differed from the host's it is kept
    in ``claims`` and shown on the row, because when a term is hard to place each file's guess about
    where it came from is a clue, and the merge must not throw those away.

    The merged row is the richest member with three extra fields, ``sources``, ``keys`` and
    ``claims``, naming everything folded into it. ``source`` and ``key`` stay scalar and name the
    **primary**: the row that landed if one did, and otherwise the best source under
    :data:`ADJUDICATION_ORDER`. So anything addressing a row by key still works, and a verdict or an
    evidence location lands on the row the curated ground truth would carry.

    ``char_offset`` is **not** merged across sources: it indexes the prior_annotation file's
    own copy of the report, so on a row whose primary is ``daphne`` it would be a position in a
    coordinate system nobody promised it indexes, the one thing :mod:`spans` exists to refuse.

    *mergeable* is the set of sources allowed to take part. It defaults to the **adjudication
    order**. A ``prior_annotation_2`` row states a code and nothing else, so it lands nowhere, is compatible with
    every row for that code, and would be absorbed into all of them, and then reported as part of
    an annotation it says nothing about. It is a cross-check, and a cross-check that has merged into
    the thing it checks is not one. Reference rows pass through as themselves.

    A caller whose question is *what is left to place* passes ``new`` as well, so that a term this
    app has already located itself absorbs the file row that could not be placed. Nothing else can
    close that one: the file's own claim will never start working, and without the fold the queue
    goes on asking for a phenotype that is already sitting on the words it came from.
    """
    allowed = frozenset(mergeable) if mergeable is not None else frozenset(ADJUDICATION_ORDER)
    by_code: dict[str, list[dict]] = {}
    for record in records:
        by_code.setdefault(record.get("hpo_code", ""), []).append(record)

    merged: list[dict] = []
    for members in by_code.values():
        kept: list[tuple] = []          # (span, merged record)
        for record in sorted(members, key=lambda r: _richness(_span(r, spans_by_key), r)):
            span = _span(record, spans_by_key)
            joins = record.get("source", "") in allowed
            host = next((entry for entry in kept
                         if joins and entry[1]["source"] in allowed
                         and same_annotation(entry[0], span)), None)
            if host is None:
                first = dict(record)
                first["sources"] = [record.get("source", "")]
                first["keys"] = [record.get("key", "")]
                first["claims"] = []
                kept.append((span, first))
                continue
            _absorb(host[1], record)
        merged.extend(entry[1] for entry in kept)
    return merged


def _span(record: dict, spans_by_key):
    return (spans_by_key or {}).get(record.get("key", ""))


def _absorb(host: dict, record: dict) -> None:
    """Fold *record* into *host*, taking any field the host does not already have."""
    for field, value in (("sources", record.get("source", "")), ("keys", record.get("key", ""))):
        if value and value not in host[field]:
            host[field].append(value)
    # Only a genuine second opinion: a field both rows state, and state differently. A row that
    # merely says *less* than the host is not another account of where the term came from, and
    # noting it would put a line under half the panel saying nothing. Computed before the fields
    # below are filled in, or every folded row would look like it agreed.
    if any(mine is not None and theirs is not None and mine != theirs
           for mine, theirs in zip(_claim(record), _claim(host))):
        host["claims"].append({"source": record.get("source", ""),
                               "trigger_word": record.get("trigger_word", ""),
                               "segment_idx": record.get("segment_idx")})
    for field in ("trigger_word", "segment_idx", "sentence_context", "hpo_name", "provenance"):
        if _blank(host.get(field)) and not _blank(record.get(field)):
            host[field] = record[field]
    # ``confirmed`` is three-valued: True, False, and "this file does not say". Only the last is
    # missing information, so a False is never overwritten by a True from another file.
    if host.get("confirmed") is None and record.get("confirmed") is not None:
        host["confirmed"] = record["confirmed"]


def assign_slots(records: list[dict]) -> list[dict]:
    """Give every record after the first of its ``(patient, code)`` an index, in file order.

    A source carrying one code twice for one patient is two annotations, two triggers for one
    phenotype, and they must key apart or a verdict on one silently settles the other. The **first**
    keeps ``slot=None`` so its key stays the short one: appending a second occurrence to the file
    must not re-key the annotation somebody already adjudicated.
    """
    seen: dict[tuple[str, str], int] = {}
    for record in records:
        pair = (record["patient_id"], record["hpo_code"])
        count = seen.get(pair, 0)
        record["slot"] = None if count == 0 else count
        seen[pair] = count + 1
    return records


def _by_patient(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for record in assign_slots(records):
        out.setdefault(record["patient_id"], []).append(record)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# loaders
# ──────────────────────────────────────────────────────────────────────────────
def load_segments(path: str) -> dict[str, list[str]]:
    """``segmented_reports.csv`` → ``{patient_id: [sentence, …]}`` in sentence order."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"segmented reports not found: {path}")
    frame = pd.read_csv(path)
    missing = {"patient_id", "sentence_idx", "sentence"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
    out: dict[str, list[str]] = {}
    for pid, group in frame.groupby("patient_id"):
        out[str(pid)] = (
            group.sort_values("sentence_idx")["sentence"].dropna().astype(str).tolist()
        )
    return out


def load_prior_annotation(path: str) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """``hcy_holistic_ground_truth.csv`` → ``({patient_id: [record, …]}, {patient_id: report})``.

    The second map is the file's own ``report_text`` column, which is worth keeping for two
    reasons: it is the coordinate system ``char_offset`` indexes, and it is a **verbatim report for
    a patient PhenoBERT was never run on**, so a cohort with no staged the PhenoBERT baseline output still gets
    prose, not a list of tokenized sentences.

    A missing file yields empty maps, not raising: curating with only some of the sources
    present is a legitimate session, and the sidebar says which ones are missing.
    """
    if not os.path.isfile(path):
        logger.warning("prior_annotation ground truth not found, continuing without it: %s", path)
        return {}, {}
    frame = pd.read_csv(path)
    if "patient_id" not in frame.columns or "hpo_code" not in frame.columns:
        raise ValueError(f"{path}: expected at least patient_id and hpo_code columns, "
                         f"found {list(frame.columns)}")

    records: list[dict] = []
    texts: dict[str, str] = {}
    for _, row in frame.iterrows():
        patient_id = _text(row, "patient_id")
        code = fix_hpo_id(_text(row, "hpo_code"))
        if not patient_id or not code:
            continue
        report_text = _text(row, "report_text")
        # One report per patient, and the file repeats it on every row. The first non-empty one
        # wins, not the last: a truncated repeat later in the file must not replace it.
        if report_text and not texts.get(patient_id):
            texts[patient_id] = report_text
        records.append({
            "source": "prior_annotation",
            "patient_id": patient_id,
            "hpo_code": code,
            "hpo_name": _text(row, "hpo_name"),
            "trigger_word": _text(row, "trigger_word"),
            "segment_idx": None,
            "char_offset": _int_or_none(row, "char_offset"),
            "sentence_context": _text(row, "sentence_context"),
            "confirmed": None,
            "provenance": "",
        })
    logger.info("prior_annotation: %d annotation(s) over %d patient(s), %d report text(s)",
                len(records), len({r["patient_id"] for r in records}), len(texts))
    return _by_patient(records), texts


def load_confirmed(path: str) -> dict[str, list[dict]]:
    """``annotations_confirmed.csv`` → ``{patient_id: [record, …]}``.

    Already segment-located, which makes it the one source whose annotations need no locating at
    all when its segmentation agrees with ``segmented_reports.csv``. Where it does not, the
    ``segment`` column travels with the record as context so the evidence location can be recovered, see
    :mod:`hpo_extraction.curation.evidence_location`.

    ``confirmed`` is carried, never filtered on. An unconfirmed row is an annotation somebody
    entered and nobody has signed off, which is the queue this app exists to work through. Dropping it here would hide it from the only screen that could resolve it.
    """
    if not os.path.isfile(path):
        logger.warning("confirmed annotations not found, continuing without them: %s", path)
        return {}
    frame = pd.read_csv(path)
    if "patient_id" not in frame.columns or "hpo_code" not in frame.columns:
        raise ValueError(f"{path}: expected at least patient_id and hpo_code columns, "
                         f"found {list(frame.columns)}")

    records: list[dict] = []
    for _, row in frame.iterrows():
        patient_id = _text(row, "patient_id")
        code = fix_hpo_id(_text(row, "hpo_code"))
        if not patient_id or not code:
            continue
        records.append({
            "source": "daphne",
            "patient_id": patient_id,
            "hpo_code": code,
            "hpo_name": _text(row, "hpo_name"),
            "trigger_word": _text(row, "trigger_word"),
            "segment_idx": _int_or_none(row, "segment_idx"),
            "char_offset": None,
            "sentence_context": _text(row, "segment"),
            "confirmed": _flag(row, "confirmed"),
            "provenance": _text(row, "provenance"),
        })
    logger.info("daphne: %d annotation(s) over %d patient(s)",
                len(records), len({r["patient_id"] for r in records}))
    return _by_patient(records)


def load_gold(path: str) -> dict[str, list[str]]:
    """A two-column code-only ground truth file → ``{patient_id: [hpo_code, …]}``, de-duplicated.

    Column *names* are not trusted, the first column is the id and the second the codes, which is
    what ``HCYDataset.load_ground_truth`` assumes too. A missing file yields ``{}``, not
    raising: curating with only some of the sources present is a legitimate session.
    """
    if not os.path.isfile(path):
        logger.warning("gold file not found, continuing without it: %s", path)
        return {}
    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError(f"{path}: expected at least 2 columns, found {frame.shape[1]}")
    id_col, code_col = frame.columns[0], frame.columns[1]

    out: dict[str, list[str]] = {}
    for _, row in frame.iterrows():
        pid = str(row[id_col]).strip()
        if not pid or pid.lower() == "nan":
            continue
        codes = out.setdefault(pid, [])
        for code in parse_hpo_codes(row[code_col]):
            code = fix_hpo_id(code)
            if code and code not in codes:
                codes.append(code)
    return out


def load_code_only(path: str, source: str) -> dict[str, list[dict]]:
    """A code-only ground truth file, as annotation records with nothing but the code filled in.

    One shape for every source is what lets the screens stop asking which file a term came from
    before they know how to draw it. What a code-only source contributes is an empty trigger and no
    segment, which is the honest statement of what that file knows, and what sends the term to
    :mod:`locate` for a candidate origin instead of a mark.
    """
    records = [
        {
            "source": source,
            "patient_id": patient_id,
            "hpo_code": code,
            "hpo_name": "",
            "trigger_word": "",
            "segment_idx": None,
            "char_offset": None,
            "sentence_context": "",
            "confirmed": None,
            "provenance": "",
        }
        for patient_id, codes in load_gold(path).items()
        for code in codes
    ]
    return _by_patient(records)


def load_phenobert(run_dir: str, patient_ids) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """``({patient_id: [detection, …]}, {patient_id: verbatim report text})``.

    Delegates to :mod:`hpo_extraction.curation.phenobert_output`. Detections keep their ``start``/``end``
    offsets into the returned text, plus ``negated`` (PhenoBERT found the phenotype and ruled it
    out) and ``resolved`` (the code exists in this HPO release), both of which a curator needs and
    the predicted set throws away.

    A missing or empty run directory yields empty maps. PhenoBERT is a reference column here, not
    a dependency: the app must open on a machine where the PhenoBERT baseline has not been run.
    """
    from hpo_extraction.curation import phenobert_output as pbstandalone

    if not run_dir or not pbstandalone.is_run_dir(run_dir):
        if run_dir:
            logger.warning("no PhenoBERT run at %s — continuing without that column", run_dir)
        return {}, {}

    by_patient: dict[str, list[dict]] = {}
    for row in pbstandalone.load_detections(run_dir):
        by_patient.setdefault(row["report_id"], []).append(row)
    for rows in by_patient.values():
        rows.sort(key=lambda r: (r["start"], r["end"]))

    # Texts are looked up per known id, never by inverting a filename, an HCY id may contain a
    # colon that the staged stem replaced with an _ character.
    wanted = list(dict.fromkeys(list(patient_ids) + list(by_patient)))
    texts = pbstandalone.load_texts(run_dir, wanted)
    return by_patient, texts


def default_paths(hcy_dir: str = DEFAULT_HCY_DIR, pb_dir: str = DEFAULT_PB_DIR) -> dict:
    """The path set the cluster layout produces, as the sidebar's starting values."""
    paths = {"hcy_dir": hcy_dir, "segments": os.path.join(hcy_dir, SEGMENTS_FILE)}
    for spec in GOLD_SOURCES:
        paths[spec["path_key"]] = os.path.join(hcy_dir, spec["file"])
    paths["phenobert"] = pb_dir
    return paths
