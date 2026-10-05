"""
PhenoBERT subprocess wrapper for an earlier exploratory run.

PhenoBERT requires stanza==1.4.1 (patched), which conflicts with the PhenoRAG env
(stanza 1.10.1). Pass phenobert_python pointing to a dedicated PhenoBERT conda env
so the subprocess runs under the correct interpreter.

Critical: annotate.py uses bare imports (from util import ...) and CWD-relative model
paths (../models/HPOModel_H/...), so it MUST be invoked with cwd set to the directory
that contains annotate.py (i.e. third_party/PhenoBERT/phenobert/utils/).
"""

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

logger = logging.getLogger(__name__)


#: Markers of a CUDA out-of-memory in the PhenoBERT subprocess's stderr. Matched as substrings
#: because the message reaches us as third-party text, not as an exception we can catch by type:
#: stanza re-raises torch's OOM wrapped in a plain RuntimeError with advice appended.
#:
#: The cuDNN entries are the same failure wearing a different name. cuDNN's RNN path reports a
#: workspace it could not allocate as a *status* code, not as an OOM, a char-LSTM that cannot get
#: its scratch buffer on a busy card raises ``CUDNN_STATUS_NOT_SUPPORTED``, whose text advertises a
#: non-contiguous input and says nothing about memory. Without these the CPU retry below never
#: fires for the one failure it would fix: an earlier exploratory run's ``q7_recall x apertus`` cell died this way on
#: two different nodes (9571247_56 on gpu-biomed-19, 9571927_56 on gpu-biomed-17) while the seven
#: other models on the same prompt, including one with 7x the characters and one with 4x the
#: lines, all passed, which is what rules out the input and leaves contention.
_CUDA_OOM_MARKERS = (
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    "CUDA error: out of memory",
    "CUDNN_STATUS_NOT_SUPPORTED",
    "CUDNN_STATUS_ALLOC_FAILED",
    "CUDNN_STATUS_EXECUTION_FAILED",
)


def _is_cuda_oom(stderr: str) -> bool:
    return any(m in (stderr or "") for m in _CUDA_OOM_MARKERS)


#: Return codes that mean the *host* killed the process rather than the program failing. A cgroup
#: OOM kill delivers SIGKILL, which ``subprocess`` reports as ``-9``. A shell in between reports the
#: same event as ``137``. Neither carries a Python traceback, so the stderr markers above cannot see
#: it, an OOM-killed annotate.py returns an empty stderr and a negative code.
#:
#: This is the second half of an earlier exploratory run's D1. The CPU retry above fixes the cuDNN workspace failure,
#: and then the CPU pass itself was OOM-killed at 90.7 GB of 96 GB requested (job 9571996_56,
#: ``q7_recall x apertus``): PhenoBERT holds the whole input directory's annotation in memory, so a
#: recall-first prompt writing 8 lines per reply on 68 % of sentences exceeds any per-node envelope
#: once the cohort grows. 20 reports already did. 118 and 228 will do it harder, which is why the
#: retry below splits the directory instead of asking for more RAM.
_HOST_OOM_RETURNCODES = (-9, -137, 137)

#: Input files per subprocess when the chunked retry fires. PhenoBERT annotates each file
#: independently and writes one output file per input, so chunking is a pure memory transform: the
#: union of the chunks' outputs is byte-identical to what one pass would have written. Small enough
#: that a chunk of the most verbose model still fits. Large enough that model load-up (~40 s of
#: stanza + BERT + CNN per subprocess) is amortised, not paid per report.
DEFAULT_PHENOBERT_CHUNK_SIZE = 16


def _is_host_oom(returncode: int, stderr: str) -> bool:
    """True when the OS killed annotate.py, an OOM kill, not a Python-level failure."""
    if returncode in _HOST_OOM_RETURNCODES:
        return True
    # Some launchers surface it as text instead of a signal.
    return any(m in (stderr or "") for m in ("Killed", "MemoryError", "Cannot allocate memory"))


def _chunk_input_dir(input_dir: str, chunk_size: int, staging_dir: str) -> list[str]:
    """Split *input_dir* into directories of at most *chunk_size* files under *staging_dir*.

    Hard-linked where the filesystem allows it and copied otherwise, so the chunk directories cost
    nothing on the cluster's shared scratch but still work across mount points. File **names** are
    preserved, which is essential: ``parse_phenobert_output_dir`` keys the detections by output
    filename, and a renamed chunk member would silently detach a report from its annotations.
    """
    names = sorted(
        fn for fn in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, fn))
    )
    chunks: list[str] = []
    for start in range(0, len(names), chunk_size):
        chunk_dir = os.path.join(staging_dir, f"chunk_{len(chunks):04d}")
        os.makedirs(chunk_dir, exist_ok=True)
        for fn in names[start : start + chunk_size]:
            src = os.path.join(input_dir, fn)
            dst = os.path.join(chunk_dir, fn)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
        chunks.append(chunk_dir)
    return chunks


def run_phenobert(
    phenobert_dir: str,
    input_dir: str,
    output_dir: str,
    phenobert_python: str | None = None,
    stanza_dir: str | None = None,
    p1: float = 0.8,
    p2: float = 0.6,
    p3: float = 0.9,
    n_threads: int = 4,
    chunk_size: int | None = None,
) -> None:
    """
    Call PhenoBERT annotate.py as a subprocess.

    Args:
        phenobert_dir: Directory containing annotate.py
                       (third_party/PhenoBERT/phenobert/utils/, relative or absolute).
        input_dir: Directory with input UTF-8 .txt files (one per patient).
        output_dir: Directory where PhenoBERT writes output TSV files.
        phenobert_python: Path to the Python interpreter that has stanza==1.4.1
                          installed with the required patches. Defaults to
                          sys.executable (works locally if the active env is the
                          PhenoBERT venv. On the cluster set this to
                          /path/to/conda/envs/PhenoBERT_env/bin/python).
        stanza_dir: Path to stanza model directory. Passed as STANZA_RESOURCES_DIR
                    env var so annotate.py finds the mimic NER model without
                    re-downloading. If None, stanza uses its default ~/stanza_resources/.
        p1: CNN layer-1 confidence threshold.
        p2: Sub-model confidence threshold.
        p3: BERT semantic filter threshold.
        n_threads: OMP/MKL thread limit passed via -t flag. Match to SLURM
                   --cpus-per-task allocation.
        chunk_size: input files per subprocess when the host-OOM retry fires. None uses
                    :data:`DEFAULT_PHENOBERT_CHUNK_SIZE`; 0 disables the retry, so an OOM kill
                    raises as before. It is a *retry* parameter and never a first-attempt one, a cell that fits in one pass must keep paying one model load, not N.
    """
    python_exe = phenobert_python or sys.executable
    annotate_script = os.path.join(phenobert_dir, "annotate.py")

    if not os.path.isfile(annotate_script):
        raise FileNotFoundError(
            f"PhenoBERT annotate.py not found: {annotate_script}\n"
            "Set phenobert_dir to the directory containing annotate.py "
            "(third_party/PhenoBERT/phenobert/utils/)."
        )

    os.makedirs(output_dir, exist_ok=True)

    # Launched through phenobert_launcher, not directly: it raises the recursion limit
    # before annotate.py imports stanza, so a single very long line cannot kill the run inside
    # stanza's recursive Chu-Liu/Edmonds cycle detection. See that module for the full story.
    # The launcher is stdlib-only, so running it under the PhenoBERT interpreter is safe, and it
    # is passed by absolute path because cwd below is phenobert_dir.
    launcher = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phenobert_launcher.py")

    # Parameterised by input directory so the host-OOM retry below can re-run the identical
    # command over a subset of the files, same thresholds, same output directory.
    def _build_cmd(in_dir: str) -> list:
        return [
            python_exe, launcher, "annotate.py",
            "-i", in_dir,
            "-o", output_dir,
            "-p1", str(p1),
            "-p2", str(p2),
            "-p3", str(p3),
            "-t", str(n_threads),
        ]

    cmd = _build_cmd(input_dir)
    logger.info(
        "Running PhenoBERT | python=%s input=%s output=%s threads=%d",
        python_exe, input_dir, output_dir, n_threads,
    )
    logger.debug("Command: %s", " ".join(cmd))

    env = os.environ.copy()
    if stanza_dir is not None:
        # Tells stanza where to find mimic NER model without re-downloading.
        env["STANZA_RESOURCES_DIR"] = os.path.abspath(stanza_dir)

    # cwd must be phenobert_dir so that bare imports (util, model) and
    # CWD-relative asset paths (../models/, ../embeddings/) resolve correctly.
    def _annotate(run_env, in_dir=input_dir):
        return subprocess.run(
            _build_cmd(in_dir), cwd=phenobert_dir, env=run_env, capture_output=True, text=True,
        )

    result = _annotate(env)

    if result.returncode != 0 and _is_cuda_oom(result.stderr):
        # One pathological line can make stanza's dependency parser ask for more VRAM than the card
        # has, a single sentence of a few thousand tokens tried to allocate 22.6 GiB on a 23.5 GiB
        # 4090 (an earlier exploratory run's p3_two_column x medgemma cell). CPU has no such ceiling.
        #
        # This is a retry, not a policy because it must not perturb the cells that already
        # work: PhenoBERT picks its device purely from torch.cuda.is_available() (utils/util.py,
        # utils/model.py) and stanza does the same, so hiding the GPUs moves the identical models at
        # The identical thresholds onto the CPU. The result is the same annotation, produced slowly
        #, not an approximation of it.
        logger.warning(
            "PhenoBERT ran out of GPU memory on %s — retrying on CPU. Same models and thresholds, "
            "so the output is unchanged; it is only slower. The underlying cause is an input line "
            "long enough to blow up the dependency parse (see "
            "experiments/findings/exp13_phenobert_input_format.md).",
            os.path.basename(output_dir),
        )
        # Rebound, not just passed: if the chunked retry below also fires, it must take over
        # The CPU-only environment, not walking back onto the card that just failed.
        env = dict(env, CUDA_VISIBLE_DEVICES="")
        result = _annotate(env)

    n_per_chunk = DEFAULT_PHENOBERT_CHUNK_SIZE if chunk_size is None else int(chunk_size)
    host_oom = _is_host_oom(result.returncode, result.stderr)
    if result.returncode != 0 and n_per_chunk > 0 and host_oom:
        # The host OOM-killed annotate.py. PhenoBERT holds the whole input directory's annotation
        # in memory, so the fix is fewer files per process, not a bigger allocation, and because
        # every input file is annotated independently into its own output file, the union of the
        # chunks is what one pass would have produced. See _HOST_OOM_RETURNCODES.
        with tempfile.TemporaryDirectory(prefix="phenobert_chunks_") as staging:
            chunks = _chunk_input_dir(input_dir, n_per_chunk, staging)
            logger.warning(
                "PhenoBERT was OOM-killed on %s (exit %d) — retrying in %d chunk(s) of <=%d "
                "file(s). Each input file is annotated independently, so the combined output is "
                "identical to a single pass; it only costs one model load per chunk.",
                os.path.basename(output_dir), result.returncode, len(chunks), n_per_chunk,
            )
            for i, chunk_dir in enumerate(chunks, start=1):
                result = _annotate(env, chunk_dir)
                if result.returncode != 0:
                    # Name the members: at this size the failure is one report's text, and the
                    # operator needs to know which one, not that "the cell died".
                    members = sorted(os.listdir(chunk_dir))
                    raise RuntimeError(
                        f"PhenoBERT annotate.py exited with code {result.returncode} on chunk "
                        f"{i}/{len(chunks)} of {os.path.basename(output_dir)} even after the "
                        f"host-OOM split. Chunk members: {members}\n"
                        f"stderr (last 2000 chars): {result.stderr[-2000:]}"
                    )
                logger.info("PhenoBERT chunk %d/%d done", i, len(chunks))

    if result.stdout:
        logger.debug("PhenoBERT stdout:\n%s", result.stdout)
    if result.stderr:
        logger.debug("PhenoBERT stderr:\n%s", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"PhenoBERT annotate.py exited with code {result.returncode}.\n"
            f"stderr (last 2000 chars): {result.stderr[-2000:]}"
        )
    logger.info("PhenoBERT annotation complete → %s", output_dir)


#: Punctuation that already ends a line for PhenoBERT's purposes, see :func:`terminate_lines`.
_LINE_TERMINATORS = ".;:,!?"


def terminate_lines(text: str) -> str:
    """End every line with punctuation so PhenoBERT cannot fuse it into the next one.

    ``process_text2phrases`` rewrites a newline that follows a word character as a bare period
    **with no space after it** (``re.sub(r"(?<=[\\w])[\\r\\n]", ".")``), so
    ``"Microcephaly\\nHypotonia"`` reaches the phrase extractor as ``microcephaly.hypotonia``, one
    token, never generated as a candidate phrase, impossible to normalise. A line that already ends
    in punctuation defeats that lookbehind and keeps its newline, so only the undecorated lines need
    a ``;``.

    Trailing whitespace is stripped first, and that is not cosmetic: a line ending in a space also
    defeats the lookbehind, but then the two lines run together with *no* separator at all, which is
    worse than the fusion it avoids.

    Measured over 691 HPO term names on the local PhenoBERT: bare newlines recover **9.4 %** of the
    intended HPO ids, terminated lines **82.1 %**, see
    ``experiments/findings/exp13_phenobert_input_format.md``.
    """
    lines = []
    for line in text.split("\n"):
        line = line.rstrip()
        if line and line[-1] not in _LINE_TERMINATORS:
            line += ";"
        lines.append(line)
    return "\n".join(lines)


def build_phenobert_input(records: list[dict], patient_id) -> tuple[str, list[tuple[int, int, int]]]:
    """One patient's PhenoBERT input text, plus the character span of each sentence block.

    Returns ``(text, spans)`` where ``spans = [(block_start, block_end, sentence_number)]``.

    This is the single definition of the input layout: :func:`write_phenobert_input` writes
    this text, and :func:`parse_phenobert_sentences` re-derives it to turn a detection's character
    offset back into a sentence number. Keeping both on one function is what makes the offset
    attribution trustworthy, the writer and the reader cannot drift apart.

    Each record contributes a marked block::

        sentence_number: {sentence_number};
        {llm_output, one terminated line per line}

    joined by newlines, in ascending ``sentence_number`` order. Every line is run through
    :func:`terminate_lines`, the marker included, an unterminated marker line fuses its own digit
    into the first phenotype word of the block (``12\\nUrinary retention`` →
    ``12.urinary retention``), which cost this layout all but 3.8 % of the terms it was handed.
    """
    patient_records = sorted(
        (r for r in records if str(r["patient_id"]) == str(patient_id)),
        key=lambda r: int(r["sentence_number"]),
    )
    blocks: list[str] = []
    spans: list[tuple[int, int, int]] = []
    pos = 0
    for r in patient_records:
        body = terminate_lines(r.get("llm_output") or "")
        block = f"sentence_number: {r['sentence_number']};\n{body}"
        blocks.append(block)
        spans.append((pos, pos + len(block), int(r["sentence_number"])))
        pos += len(block) + 1          # +1 for the "\n" that joins the blocks
    return "\n".join(blocks), spans


def write_phenobert_input(
    records: list[dict], patient_ids: list[str], txt_dir: str
) -> None:
    """
    Write per-patient PhenoBERT input files with sentence-number markers.

    A *patched* PhenoBERT (util.py `_build_metadata_index`) scans the input for
    ``sentence_number: <int>`` markers and emits the sentence number in the TSV
    `sentence_count` column. Stock PhenoBERT ignores them and writes five columns, in that case the markers still pay off, because
    :func:`parse_phenobert_sentences` recovers the same attribution from the
    detection's character offsets against :func:`build_phenobert_input`.

    Records are written in ascending `sentence_number` order, one file per patient.
    """
    os.makedirs(txt_dir, exist_ok=True)
    for patient_id in patient_ids:
        text, _spans = build_phenobert_input(records, patient_id)
        with open(os.path.join(txt_dir, f"{patient_id}.txt"), "w", encoding="utf-8") as f:
            f.write(text)


# A record starts only where a line begins with two integers. Re.S so the phrase group can span
# newlines once continuation lines have been folded back in.
_ROW_RE = re.compile(r"^(\d+)\t(\d+)\t(.*)$", re.S)


def iter_phenobert_rows(raw: str):
    """Yield ``(start, end, phrase, hpo_id, after)`` for every detection in a PhenoBERT TSV.

    ``after`` is the list of fields following the HPO id (score, and on a patched install the
    sentence index and/or a trailing ``Neg``).

    PhenoBERT writes the matched *phrase* verbatim, so a phrase containing a newline (common when
    the SLM answered with a numbered list) is split across physical lines, line-by-line parsing
    silently loses those detections. Records are therefore reassembled first.
    """
    records: list[str] = []
    pending = ""
    for line in raw.split("\n"):
        if _ROW_RE.match(line):
            if pending:
                records.append(pending)
            pending = line
        elif pending:
            pending += "\n" + line
    if pending:
        records.append(pending)

    for rec in records:
        m = _ROW_RE.match(rec)
        if not m:
            continue
        fields = m.group(3).split("\t")
        hpo = next((f for f in fields if f.strip().startswith("HP:")), None)
        if hpo is None:
            continue
        idx = fields.index(hpo)
        yield int(m.group(1)), int(m.group(2)), "\t".join(fields[:idx]), hpo.strip(), fields[idx + 1:]


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _sentence_from_offset(start: int, end: int, phrase: str, text: str, spans) -> int | None:
    """Sentence number for a detection at ``[start, end)``, or None if the offset can't be trusted.

    The offset is *verified* against the reconstructed input before it is used: if
    ``text[start:end]`` is not the phrase PhenoBERT reported, the offsets do not refer to this text
    (different extraction cache, different PhenoBERT version, …) and attributing on them would
    invent per-sentence structure that isn't there.
    """
    got = text[start:end]
    if got != phrase and _normalise(got) != _normalise(phrase):
        return None
    for block_start, block_end, sent_num in spans:
        if block_start <= start < block_end:
            return sent_num
    return None


def parse_phenobert_sentences(
    output_dir: str,
    inputs: dict | None = None,
) -> dict[str, dict[int, dict[str, int]]]:
    """
    Parse PhenoBERT output with sentence-level attribution::

        {patient_id: {sentence_number: {hpo_id: detection_count}}}

    Two ways to reach the sentence number, in order:

    1. the ``sentence_count`` column a *patched* PhenoBERT writes (the field after the score);
    2. failing that, the detection's **character offsets** (columns 1–2) resolved against the
       input this pipeline generated, passed in as ``inputs``, ``{patient_id: (text, spans)}`` from :func:`build_phenobert_input`.

    Route 2 exists because stock PhenoBERT emits only five columns, so route 1 silently discarded
    *every* detection, the Free Listing generation run "N sentences → 0 detections" failure. Each offset is verified
    against the reconstructed text before it is trusted (see :func:`_sentence_from_offset`), so a
    mismatched cache yields no attribution, not a wrong one.

    Negated detections (last field == "Neg") are skipped.
    """
    results: dict[str, dict[int, dict[str, int]]] = {}
    n_indexed = n_offset = n_dropped = 0

    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        if not os.path.isfile(fpath):
            continue
        patient_id = os.path.splitext(fname)[0]
        per_sentence: dict[int, dict[str, int]] = {}
        text, spans = (inputs or {}).get(patient_id, ("", []))

        with open(fpath, "r", encoding="utf-8") as f:
            raw = f.read()

        for start, end, phrase, hpo_id, after in iter_phenobert_rows(raw):
            if after and after[-1].strip().lower() == "neg":
                continue
            # The sentence index sits right after the score on a patched install.
            sent_num = None
            if len(after) >= 2 and after[1].strip().isdigit():
                sent_num = int(after[1].strip())
                n_indexed += 1
            elif spans:
                sent_num = _sentence_from_offset(start, end, phrase, text, spans)
                if sent_num is not None:
                    n_offset += 1
            if sent_num is None:
                n_dropped += 1
                continue
            per_sentence.setdefault(sent_num, {})
            per_sentence[sent_num][hpo_id] = per_sentence[sent_num].get(hpo_id, 0) + 1

        results[patient_id] = per_sentence
        logger.debug(
            "Patient %s: detections across %d sentences", patient_id, len(per_sentence)
        )

    if n_indexed or n_offset or n_dropped:
        logger.info(
            "Sentence attribution in %s | %d from the sentence_count column, %d from offsets, "
            "%d unattributable", os.path.basename(output_dir), n_indexed, n_offset, n_dropped,
        )
    return results


def diagnose_phenobert_output(output_dir: str) -> dict:
    """Counters describing what PhenoBERT actually wrote, the difference between "found nothing"
    and "wrote something this parser cannot read".

    ``parse_phenobert_sentences`` drops any row without an ``HP:`` id in column 4 or a numeric
    ``sentence_count`` in column 6, and it drops them silently: an unpatched PhenoBERT (no
    ``sentence_number:`` marker support in ``util.py``) yields a full output directory that parses
    to zero detections, which is how the Free Listing generation run came to log
    ``2713 sentences → 0 detections`` for every model. These counters make the two cases
    distinguishable, and callers turn them into a hard error instead of an empty result::

        {"n_files": 135, "n_rows": 8421, "n_hp_rows": 8421, "n_indexed_rows": 0}
                                                             ^ patch missing
    """
    out = {"n_files": 0, "n_rows": 0, "n_hp_rows": 0, "n_indexed_rows": 0}
    if not os.path.isdir(output_dir):
        return out
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        if not os.path.isfile(fpath):
            continue
        out["n_files"] += 1
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out["n_rows"] += 1
                parts = line.split("\t")
                # Count HPO rows at *any* width: stock PhenoBERT writes five columns
                # (start end phrase HP:id score), and requiring six here would report zero HPO rows
                # for a directory full of perfectly good detections, hiding the very case this
                # function exists to name.
                if len(parts) < 4 or not parts[3].strip().startswith("HP:"):
                    continue
                out["n_hp_rows"] += 1
                if len(parts) >= 6 and parts[5].strip().isdigit():
                    out["n_indexed_rows"] += 1
    return out


def phenobert_output_is_usable(output_dir: str) -> bool:
    """True if this output directory holds HPO detections worth reusing.

    Weaker than :func:`phenobert_output_has_sentence_index` on purpose: with offset attribution a
    stock 5-column cache is perfectly usable, and demanding a sentence index would re-run PhenoBERT
    (minutes per model) on every invocation for nothing. A cache built from *different* input is
    still rejected downstream, the offsets simply fail verification and attribute nothing.
    """
    return diagnose_phenobert_output(output_dir)["n_hp_rows"] > 0


def phenobert_output_matches_input(output_dir: str, inputs: dict) -> bool:
    """True if this cached output was produced from *these* inputs.

    :func:`phenobert_output_is_usable` only asks whether detections exist, so it happily green-lights
    a cache built from different text, and offset attribution then verifies every row, finds none of
    them landing on their own phrase, and yields an empty result. The run does not fail. It reports
    zero detections, which is indistinguishable from "PhenoBERT found nothing".

    Any change to :func:`build_phenobert_input` shifts every offset, so this check is what makes the
    layout safe to change: a stale cache is re-run instead of silently parsing to nothing.

    ``inputs`` is ``{patient_id: (text, spans)}`` from :func:`build_phenobert_input`. A row is
    matched when ``text[start:end]`` is the phrase the row reports. Returns False for an output
    directory holding no checkable rows at all.

    Coverage is checked before content, and that ordering is the point. A run killed part-way
    through annotation leaves a *consistent* cache of the patients it got to: every row in it lands
    on its own phrase, so a content-only check green-lights it and the cohort silently shrinks. That
    is not hypothetical, the an earlier exploratory run medgemma cells died in stanza after two of twenty patients,
    and the backfill then scored p2_canonical over those two.
    """
    present = {
        os.path.splitext(f)[0]
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    }
    missing = [pid for pid in inputs if str(pid) not in present]
    if missing:
        logger.info(
            "Cached PhenoBERT output in %s covers %d/%d patient(s) — missing %s%s. Re-running.",
            os.path.basename(output_dir), len(inputs) - len(missing), len(inputs),
            ", ".join(map(str, missing[:5])), " …" if len(missing) > 5 else "",
        )
        return False

    n_checked = n_ok = 0
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        if not os.path.isfile(fpath):
            continue
        text = (inputs.get(os.path.splitext(fname)[0]) or ("", []))[0]
        if not text:
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            raw = f.read()
        for start, end, phrase, _hpo, _after in iter_phenobert_rows(raw):
            n_checked += 1
            got = text[start:end]
            if got == phrase or _normalise(got) == _normalise(phrase):
                n_ok += 1
    if n_checked == 0:
        return False
    # Not all-or-nothing: a handful of rows can legitimately fail (accented characters shift the
    # offsets, since strip_accents is not length-preserving). A layout change fails essentially all.
    share = n_ok / n_checked
    logger.info(
        "Cached PhenoBERT output in %s: %d/%d rows land on their own phrase (%.1f%%)",
        os.path.basename(output_dir), n_ok, n_checked, 100 * share,
    )
    return share >= 0.9


def phenobert_output_has_sentence_index(output_dir: str) -> bool:
    """
    True if `output_dir` holds PhenoBERT TSVs with a populated `sentence_count`
    column (new marker-augmented format).

    Used to invalidate stale caches: outputs generated from input without
    `sentence_number:` markers carry "None" in the `sentence_count` column and
    cannot support per-sentence attribution, so they must be re-run.
    Returns False when the directory is missing or contains no data rows.
    """
    if not os.path.isdir(output_dir):
        return False
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 6 and parts[5].strip().isdigit():
                    return True
    return False


def parse_phenobert_tsv(tsv_path: str) -> dict[str, int]:
    """
    Parse one PhenoBERT output TSV file → {hpo_id: detection_count}.

    TSV columns (tab-separated, no header):
        start_loc  end_loc  phrase  HPO_ID  confidence  [sentence_count  sentence_source]  [Neg]

    Only positive (non-negated) detections are counted.
    """
    hpo_counts: dict[str, int] = defaultdict(int)

    if not os.path.isfile(tsv_path):
        logger.warning("PhenoBERT output not found: %s", tsv_path)
        return {}

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            hpo_id = parts[3].strip()
            # Skip negated entries (PhenoBERT appends "Neg" as the last field)
            if parts[-1].strip().lower() == "neg":
                continue
            if hpo_id and hpo_id.startswith("HP:"):
                hpo_counts[hpo_id] += 1

    return dict(hpo_counts)


def parse_phenobert_output_dir(output_dir: str) -> dict[str, dict[str, int]]:
    """
    Parse all PhenoBERT output files in output_dir.

    Returns:
        {patient_id: {hpo_id: detection_count}}
    """
    results: dict[str, dict[str, int]] = {}

    for fname in os.listdir(output_dir):
        patient_id = os.path.splitext(fname)[0]
        results[patient_id] = parse_phenobert_tsv(os.path.join(output_dir, fname))
        logger.debug(
            "Patient %s: %d unique HPOs from PhenoBERT",
            patient_id, len(results[patient_id]),
        )

    return results
