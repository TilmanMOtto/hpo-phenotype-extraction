"""
Per-model-family loaders for the earlier runs SLM ensemble.

Different models need different loading APIs and chat handling:
- standard causal LMs (Apertus, DeepSeek, II-Medical, MedPsy, Llama-3.1, Phi-4)
  → AutoModelForCausalLM + AutoTokenizer
- MedGemma (gemma3, multimodal) → AutoModelForImageTextToText + AutoProcessor
- Llama3-OpenBioLLM → transformers.pipeline + bf16 + <|eot_id|> terminator

`load_slm()` hides these differences behind a uniform `LoadedSLM.generate(system_prompt,
user_msg, max_new_tokens) -> str`. Verified against all 8 ensemble models in
an earlier exploratory run under transformers 4.57 / torch 2.6.

`LoadedSLM.generate_many()` is the batched entry point (the Free Listing generation run). Only the standard family
has a real batched path (left-padded). Medgemma and openbiollm fall back to a sequential loop,
so callers may always use it.

`LoadedSLM.rank_sequences()` is the earlier constrained-decoding entry point: given one prompt and N
candidate strings, it returns the teacher-forced sequence log-probability of each. All three
families implement it, because the chat/processor construction it needs is the
family-specific part that `score_fn` already had to solve.
"""

import gc
import logging
from typing import Callable, Optional

import torch

from hpo_extraction.models.verdict import single_token_id

# Minimal valid Llama-3 instruct chat template. OpenBioLLM-Llama3-8B ships a tokenizer
# with no chat_template, so apply_chat_template() fails. The model card instructs using
# The Llama-3 instruct template explicitly. Assigned only when the tokenizer lacks one.
LLAMA3_CHAT_TEMPLATE = (
    "{{ '<|begin_of_text|>' }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)

# Generic ChatML fallback for standard-loader models whose tokenizer ships no chat_template
# (e.g. An Apertus base model). Lets the model run. Instruction-following quality depends
# on whether the model was actually instruction-tuned.
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def _get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    return logger if logger is not None else logging.getLogger(__name__)


#: Continuations scored per forward pass. Small on purpose: the logits tensor is
#: ``batch x seq_len x vocab``, and vocab is ~128k, so 8 rows of a ~200-token prompt already
#: materialises ~430 MB in bf16. Raising this trades GPU memory for wall-clock.
DEFAULT_RESCORE_BATCH = 8


def sequence_logprobs(
    model,
    prompt_ids: list,
    cont_ids_list: list,
    *,
    pad_id: int,
    batch_size: int = DEFAULT_RESCORE_BATCH,
) -> list:
    """Teacher-forced ``sum log p(continuation | prompt)`` for each continuation.

    This is the primitive constrained decoding is built on here, and it is *exact*
    rather than a masked greedy decode. Scoring each candidate string as a whole continuation
    sidesteps the reason a naive token trie is subtly wrong: BPE is not prefix-consistent, so a
    trie built from ``encode(term)`` dead-ends when the model reaches the same character prefix by
    a different-but-equivalent token path. Here every candidate is tokenised once, as
    itself, and the comparison between candidates is over complete sequences, which is the
    constrained MAP, not an approximation of it.

    Padding is on the **right**. The generation helpers pad left because ``generate`` must find the
    real prompt end at the last position. Here the continuation sits at known fixed offsets and the
    attention mask hides the pads, so right padding keeps those offsets identical across rows.

    Not yet using a shared-prefix KV cache. Every row re-encodes the same prompt, which is ~95 % of
    the tokens in the batch. That is a pure speedup with no effect on the numbers, so it can be
    added behind this signature if the pilot shows it is needed.
    """
    if not cont_ids_list:
        return []
    device = model.device
    prompt_len = len(prompt_ids)
    out: list[float] = []

    for start in range(0, len(cont_ids_list), batch_size):
        chunk = cont_ids_list[start:start + batch_size]
        rows = [list(prompt_ids) + list(c) for c in chunk]
        width = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
            attention[i, :len(row)] = 1
        input_ids = input_ids.to(device)
        attention = attention.to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention).logits

        for i, cont in enumerate(chunk):
            n = len(cont)
            # logits at position t predict token t+1, so the distribution over cont[j] sits at
            # prompt_len - 1 + j. Sliced per row so the full-vocab log_softmax is never
            # materialised for the whole batch at once.
            window = logits[i, prompt_len - 1:prompt_len - 1 + n, :].float()
            logprobs = torch.log_softmax(window, dim=-1)
            target = torch.tensor(list(cont), dtype=torch.long, device=logprobs.device)
            out.append(float(logprobs.gather(1, target.unsqueeze(1)).sum()))

        del logits
    return out


def _rank_with(tokenizer, model, prompt_text: str, continuations: list, *,
               terminator_id: Optional[int], batch_size: int) -> list:
    """Tokenise, dedupe, score, and map back, the family-independent half of ``rank_fn``."""
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # Encode each candidate once. The terminator counts: without it "Seizure" is a token-prefix
    # of "Seizures" and every short string collects the probability mass of its own extensions,
    # which systematically inflates it. With it, each score answers "emit this, and stop".
    encoded: dict[str, list] = {}
    for text in continuations:
        if text in encoded:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if terminator_id is not None:
            ids = ids + [terminator_id]
        encoded[text] = ids

    unique = list(encoded)
    scores = sequence_logprobs(
        model, prompt_ids, [encoded[t] for t in unique],
        pad_id=pad_id, batch_size=batch_size,
    )
    by_text = dict(zip(unique, scores))
    return [(by_text[t], len(encoded[t])) for t in continuations]


class LoadedSLM:
    """A loaded SLM with a uniform text-in/text-out generate() across model families.

    ``score_fn`` (optional) exposes a single-forward-pass Yes/No decision: it returns the
    ``yes_logit - no_logit`` margin at the first generated position, letting the earlier logit
    study classify without parsing free-form text. It is family-specific (built by each loader
    where the correct chat/processor construction lives), so it lives here, not in a
    generic wrapper.
    """

    def __init__(
        self,
        model_name: str,
        generate_fn: Callable[[str, str, int], str],
        cleanup_fn: Callable[[], None],
        score_fn: Optional[Callable[[str, str], float]] = None,
        generate_many_fn: Optional[Callable[[str, list, int], list]] = None,
        rank_fn: Optional[Callable[[str, str, list, int], list]] = None,
    ) -> None:
        self.model_name = model_name
        self._generate_fn = generate_fn
        self._cleanup_fn = cleanup_fn
        self._score_fn = score_fn
        self._generate_many_fn = generate_many_fn
        self._rank_fn = rank_fn

    @property
    def supports_batching(self) -> bool:
        """True if this family has a real batched path (else generate_many loops)."""
        return self._generate_many_fn is not None

    @property
    def supports_ranking(self) -> bool:
        """True if this family can score candidate continuations (earlier constrained decoding)."""
        return self._rank_fn is not None

    def generate(self, system_prompt: str, user_msg: str, max_new_tokens: int) -> str:
        """Return the model's reply to (system_prompt, user_msg)."""
        return self._generate_fn(system_prompt, user_msg, max_new_tokens)

    def generate_many(
        self, system_prompt: str, user_msgs: list, max_new_tokens: int
    ) -> list:
        """Return one reply per user message, batched where the family supports it.

        Families without a batched path (medgemma, openbiollm) fall back to a sequential
        loop, so callers can always use this entry point regardless of model.
        """
        if not user_msgs:
            return []
        if self._generate_many_fn is None:
            return [self._generate_fn(system_prompt, m, max_new_tokens) for m in user_msgs]
        return self._generate_many_fn(system_prompt, user_msgs, max_new_tokens)

    def score_yes_no(self, system_prompt: str, user_msg: str) -> float:
        """Return the ``yes_logit - no_logit`` margin (positive → Yes) via one forward pass."""
        if self._score_fn is None:
            raise NotImplementedError(
                f"Logit scoring is not implemented for model family '{self.model_name}'."
            )
        return self._score_fn(system_prompt, user_msg)

    def rank_sequences(
        self,
        system_prompt: str,
        user_msg: str,
        continuations: list,
        batch_size: int = DEFAULT_RESCORE_BATCH,
    ) -> list:
        """``[(logprob, n_tokens), ...]`` for each continuation under one prompt.

        The earlier constrained-decoding entry point. ``logprob`` is the summed teacher-forced
        log-probability of emitting that string and stopping; ``n_tokens`` includes the
        terminator and is what the length-normalised and PMI scoring rules divide by.

        Results are positional: the returned list is the same length and order as ``continuations``,
        including duplicates, which are scored once and reused.
        """
        if self._rank_fn is None:
            raise NotImplementedError(
                f"Sequence ranking is not implemented for model family '{self.model_name}'."
            )
        if not continuations:
            return []
        return self._rank_fn(system_prompt, user_msg, continuations, batch_size)

    def unload(self) -> None:
        """Free GPU/CPU memory held by the underlying model."""
        self._cleanup_fn()


def _cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Standard loader: AutoModelForCausalLM + AutoTokenizer
# (Apertus, DeepSeek, II-Medical, MedPsy, Llama-3.1, Phi-4)
# ---------------------------------------------------------------------------

def _load_standard(
    model_name: str, model_path: str, logger: logging.Logger, deterministic: bool = False
) -> LoadedSLM:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("[%s] Loading with AutoModelForCausalLM …", model_name)

    tok_kwargs = {"trust_remote_code": True}
    if "medpsy" in model_name.lower():
        # MedPsy-4B's tokenizer warns about an incorrect Mistral regex pattern
        tok_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", dtype="auto", trust_remote_code=True
    )
    model.eval()
    logger.info(
        "[%s] Loaded — device map: %s",
        model_name,
        model.hf_device_map if hasattr(model, "hf_device_map") else "N/A",
    )

    if tokenizer.chat_template is None:
        logger.info("[%s] tokenizer has no chat_template — using generic ChatML template", model_name)
        tokenizer.chat_template = CHATML_TEMPLATE

    # MedPsy example only uses the user role. Fold the system prompt into the user turn.
    has_system = "medpsy" not in model_name.lower()

    # Batched generation needs left padding (decoder-only: the last position must be the real
    # prompt end for every row) and a pad token. Many instruct models ship neither.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    can_batch = tokenizer.pad_token_id is not None

    def _prompt_text(system_prompt: str, user_msg: str) -> str:
        if has_system:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
        else:
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_msg}"}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _gen_kwargs(max_new_tokens: int) -> dict:
        kwargs = {"max_new_tokens": max_new_tokens}
        if deterministic:
            kwargs["do_sample"] = False  # greedy → reproducible
        else:
            kwargs.update(temperature=0.6, do_sample=True)
        return kwargs

    def generate_fn(system_prompt: str, user_msg: str, max_new_tokens: int) -> str:
        text = _prompt_text(system_prompt, user_msg)
        inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, **_gen_kwargs(max_new_tokens))
        return tokenizer.decode(
            out_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        ).strip()

    def generate_many_fn(system_prompt: str, user_msgs: list, max_new_tokens: int) -> list:
        """Left-padded batched generation. Under greedy decoding this is equivalent to the
        sequential path up to padding/attention fp noise, verify before trusting a batched run.
        """
        texts = [_prompt_text(system_prompt, m) for m in user_msgs]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        prompt_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, pad_token_id=tokenizer.pad_token_id, **_gen_kwargs(max_new_tokens)
            )
        return [
            tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
            for row in out_ids
        ]

    yes_id = single_token_id(tokenizer, "Yes")
    no_id = single_token_id(tokenizer, "No")

    def score_fn(system_prompt: str, user_msg: str) -> float:
        text = _prompt_text(system_prompt, user_msg)
        inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        return logits[yes_id].item() - logits[no_id].item()

    def rank_fn(system_prompt: str, user_msg: str, continuations: list, batch_size: int) -> list:
        return _rank_with(
            tokenizer, model, _prompt_text(system_prompt, user_msg), continuations,
            terminator_id=tokenizer.eos_token_id, batch_size=batch_size,
        )

    def cleanup_fn() -> None:
        nonlocal model, tokenizer
        del model, tokenizer
        _cuda_cleanup()

    return LoadedSLM(
        model_name,
        generate_fn,
        cleanup_fn,
        score_fn,
        generate_many_fn if can_batch else None,
        rank_fn,
    )


# ---------------------------------------------------------------------------
# MedGemma loader: AutoModelForImageTextToText + AutoProcessor
# ---------------------------------------------------------------------------

def _load_medgemma(
    model_name: str, model_path: str, logger: logging.Logger, deterministic: bool = False
) -> LoadedSLM:
    # MedGemma already decodes greedily (do_sample=False); `deterministic` is accepted
    # for a uniform loader signature and has no additional effect here.
    from transformers import AutoProcessor, AutoModelForImageTextToText

    logger.info("[%s] Loading with AutoModelForImageTextToText …", model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    logger.info("[%s] Loaded.", model_name)

    def generate_fn(system_prompt: str, user_msg: str, max_new_tokens: int) -> str:
        # Text-only input, MedGemma content must be a list of typed parts.
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{system_prompt}\n\n{user_msg}"}],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()

    yes_id = single_token_id(processor.tokenizer, "Yes")
    no_id = single_token_id(processor.tokenizer, "No")

    def score_fn(system_prompt: str, user_msg: str) -> float:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{system_prompt}\n\n{user_msg}"}],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1, :]
        return logits[yes_id].item() - logits[no_id].item()

    def rank_fn(system_prompt: str, user_msg: str, continuations: list, batch_size: int) -> list:
        # The processor owns the chat template, but the ranking helper needs plain token ids, so
        # render to text here and let the wrapped tokenizer encode it, the same split the
        # standard loader gets for free from `_prompt_text`.
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{system_prompt}\n\n{user_msg}"}],
            }
        ]
        prompt_text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return _rank_with(
            processor.tokenizer, model, prompt_text, continuations,
            terminator_id=processor.tokenizer.eos_token_id, batch_size=batch_size,
        )

    def cleanup_fn() -> None:
        nonlocal model, processor
        del model, processor
        _cuda_cleanup()

    return LoadedSLM(model_name, generate_fn, cleanup_fn, score_fn, None, rank_fn)


# ---------------------------------------------------------------------------
# OpenBioLLM loader: transformers.pipeline + bfloat16 + eot_id terminator
# ---------------------------------------------------------------------------

def _load_openbiollm(
    model_name: str, model_path: str, logger: logging.Logger, deterministic: bool = False
) -> LoadedSLM:
    import transformers

    logger.info("[%s] Loading with transformers.pipeline (bfloat16) …", model_name)
    pipe = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
    )
    logger.info("[%s] Loaded.", model_name)

    if pipe.tokenizer.chat_template is None:
        logger.info("[%s] tokenizer has no chat_template — using Llama-3 instruct template", model_name)
        pipe.tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

    eot_id = pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    terminators = [t for t in [pipe.tokenizer.eos_token_id, eot_id] if t is not None]

    def generate_fn(system_prompt: str, user_msg: str, max_new_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        prompt = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": terminators if terminators else None,
        }
        if deterministic:
            gen_kwargs["do_sample"] = False  # greedy → reproducible
        else:
            gen_kwargs.update(do_sample=True, temperature=0.6)
        result = pipe(prompt, **gen_kwargs)
        return result[0]["generated_text"][len(prompt):].strip()

    yes_id = single_token_id(pipe.tokenizer, "Yes")
    no_id = single_token_id(pipe.tokenizer, "No")

    def score_fn(system_prompt: str, user_msg: str) -> float:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        input_ids = pipe.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(pipe.model.device)
        with torch.no_grad():
            logits = pipe.model(input_ids).logits[0, -1, :]
        return logits[yes_id].item() - logits[no_id].item()

    def rank_fn(system_prompt: str, user_msg: str, continuations: list, batch_size: int) -> list:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        prompt_text = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # `<|eot_id|>`, not `eos_token_id`: this model ends an assistant turn with the former,
        # and `generate_fn` already has to pass it as a terminator for the same reason.
        return _rank_with(
            pipe.tokenizer, pipe.model, prompt_text, continuations,
            terminator_id=eot_id if eot_id is not None else pipe.tokenizer.eos_token_id,
            batch_size=batch_size,
        )

    def cleanup_fn() -> None:
        nonlocal pipe
        pipe_model = pipe.model
        pipe_tok = pipe.tokenizer
        del pipe
        del pipe_model, pipe_tok
        _cuda_cleanup()

    return LoadedSLM(model_name, generate_fn, cleanup_fn, score_fn, None, rank_fn)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_LOADER_FN = {
    "medgemma": _load_medgemma,
    "openbiollm": _load_openbiollm,
}


def load_slm(
    model_name: str,
    model_path: str,
    logger: Optional[logging.Logger] = None,
    deterministic: bool = False,
) -> LoadedSLM:
    """
    Load an ensemble SLM using the correct API for its family.

    Dispatch is keyed on `model_name` (the ensemble key, e.g. "medgemma", "openbiollm",
    "apertus", "deepseek", "medpsy", …), not on the model path.

    When `deterministic` is True, generation is greedy (`do_sample=False`, no
    temperature) so outputs are reproducible, used by the earlier baseline runs.
    Default False preserves the earlier ensemble's sampled decoding.
    """
    log = _get_logger(logger)
    fn = _LOADER_FN.get(model_name, _load_standard)
    return fn(model_name, model_path, log, deterministic)
