"""Yes/No verdict parsing and tokenizer helpers shared across the earlier decoding studies.

The baseline pipeline extracts a verdict with ``response.strip().lower().startswith("yes")``
while the scored path uses substring ``"Yes" in response``, both break on models that emit
reasoning or prose before the answer (see ``exp08_findings.md``). These helpers give a single,
reliable parse plus the instrumentation flags used by ``an earlier exploratory run`` to
characterize each model's raw output, and the single-token lookup shared by the logit scorers
(``LlamaLogitsLLM``, ``SLMLogitsLLM``).
"""

import logging
import re

# <think>...</think> reasoning block emitted by R1-distilled / Qwen3 reasoning models.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# An unterminated think block (verdict never reached, e.g. truncated at max_new_tokens).
_OPEN_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
# MedGemma (Gemma-3) emits its reasoning as a <unused94>thought...<unused95> block instead of
# <think>. Handle the closed form and the truncated (unterminated) form. The close token is
# accepted with or without a slash (</unused95>), pending confirmation against a real sample.
_UNUSED_THINK_RE = re.compile(r"<unused94>.*?</?unused95>", re.DOTALL | re.IGNORECASE)
_OPEN_UNUSED_RE = re.compile(r"<unused94>.*\Z", re.DOTALL | re.IGNORECASE)
# DeepSeek-R1-Distill emits its reasoning terminated by a bare </think> with NO opening <think>
# tag. Drop everything up to and including the first closing tag so only the verdict remains.
_LEADING_CLOSE_RE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)
# \boxed{...} final-answer convention (II-Medical / math-style reasoning models).
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)
# First standalone yes/no word (word-boundary, case-insensitive).
_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def has_think_block(text: str) -> bool:
    """True if the text contains a ``<think>`` reasoning marker (open or closed)."""
    return "<think>" in text.lower()


def has_boxed(text: str) -> bool:
    """True if the text contains a ``\\boxed{...}`` final-answer marker."""
    return bool(_BOXED_RE.search(text))


def strip_think(text: str) -> str:
    """Remove reasoning blocks so only the post-reasoning answer remains.

    Handles every reasoning convention seen in the earlier model sweep (see ``exp08_findings.md``):
    ``<think>...</think>`` (Qwen3 / R1), MedGemma's ``<unused94>...<unused95>``, and DeepSeek's
    bare closing ``</think>`` with no opening tag. Closed blocks are dropped entirely. An
    unterminated block (model cut off before its closing tag) is also dropped, there is no
    verdict to salvage in that case.
    """
    without_closed = _THINK_RE.sub(" ", text)
    without_unused = _UNUSED_THINK_RE.sub(" ", without_closed)
    # Drop a leading reasoning trace terminated by a bare </think> (DeepSeek: no opening tag).
    # Runs after the paired-<think> pass so already-removed blocks cannot leave a stray </think>.
    without_leading = _LEADING_CLOSE_RE.sub(" ", without_unused)
    without_open = _OPEN_THINK_RE.sub(" ", without_leading)
    without_open_unused = _OPEN_UNUSED_RE.sub(" ", without_open)
    return without_open_unused.strip()


def parse_yes_no(text: str) -> str:
    """Best-effort verdict from free-form model output.

    Returns ``"Yes"``, ``"No"``, or ``"AMBIGUOUS"``. Strategy: drop reasoning blocks, then
    prefer an explicit ``\\boxed{...}`` answer, else fall back to the *last* standalone
    yes/no word in the remaining text (the final answer, not one mentioned mid-reasoning).
    """
    cleaned = strip_think(text)

    boxed = _BOXED_RE.search(cleaned)
    if boxed:
        inner = boxed.group(1).strip().lower()
        if inner.startswith("yes"):
            return "Yes"
        if inner.startswith("no"):
            return "No"

    matches = _YESNO_RE.findall(cleaned)
    if not matches:
        return "AMBIGUOUS"
    return "Yes" if matches[-1].lower() == "yes" else "No"


def starts_with_yes(text: str) -> bool:
    """The legacy ``baseline_eval`` verdict rule, kept for side-by-side diagnostics."""
    return text.strip().lower().startswith("yes")


def contains_yes_substring(text: str) -> bool:
    """The legacy scored-path rule (``"Yes" in response``), kept for side-by-side diagnostics."""
    return "Yes" in text


def single_token_id(tokenizer, word: str) -> int:
    """Return the single token id for ``word``. Fall back to its space-prefixed form.

    Promoted from ``LlamaLogitsLLM._single_token_id`` so both the Llama and generalized SLM
    logit scorers share one implementation. Warns and returns the first sub-token if ``word``
    does not encode to a single token under this tokenizer.
    """
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    ids_space = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids_space) == 1:
        return ids_space[0]
    logging.getLogger(__name__).warning(
        "Token '%s' encodes as %d sub-tokens; using first (id=%d)", word, len(ids), ids[0]
    )
    return ids[0]


if __name__ == "__main__":  # lightweight, GPU-free regression check
    _STRIP_CASES = [
        ("<think>reasoning yes maybe</think>\nNo", "No"),          # closed <think>
        ("<think>reasoning that got cut off yes", ""),            # unterminated <think>
        ("<unused94>thought yes yes</unused95>\nNo", "No"),        # medgemma closed (slash variant)
        ("<unused94>thought ... Final Answer: No.<unused95>No.", "No."),  # real medgemma close token
        ("<unused94>thought cut off yes yes", ""),                # medgemma truncated
        ("reasoning yes yes </think>\n\nNo", "No"),                # deepseek tagless close
        ("Yes", "Yes"),                                           # clean, unaffected
    ]
    for _raw, _expect in _STRIP_CASES:
        _got = strip_think(_raw)
        assert _got == _expect, f"strip_think({_raw!r}) = {_got!r} != {_expect!r}"

    _PARSE_CASES = [
        ("<think>maybe yes</think> No", "No"),
        ("<unused94>thought yes</unused95> Yes", "Yes"),
        ("reasoning about yes </think> No.", "No"),
        ("The answer is \\boxed{Yes}", "Yes"),
        ("I cannot determine this", "AMBIGUOUS"),
        ("Yes", "Yes"),
    ]
    for _raw, _expect in _PARSE_CASES:
        _got = parse_yes_no(_raw)
        assert _got == _expect, f"parse_yes_no({_raw!r}) = {_got!r} != {_expect!r}"

    print(f"verdict.py self-check OK ({len(_STRIP_CASES)} strip + {len(_PARSE_CASES)} parse cases)")
