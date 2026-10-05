"""LLaMA model loading and inference (HuggingFace / BitsAndBytes 8-bit)."""

import time

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig, LlamaForCausalLM

from hpo_extraction.models.verdict import single_token_id


def load_llama(llama_dir: str, load_in_4bit: bool = False):
    """
    Load LLaMA with BitsAndBytes quantization.

    Args:
        llama_dir: Path to the pretrained LLaMA model directory.
        load_in_4bit: If True, use NF4 4-bit quantization (~35 GB for 70B) instead
            of 8-bit (~70 GB). Use for models that don't fit in GPU VRAM at 8-bit.

    Placement is `sequential` on multi-GPU and `auto` on one card. See the comment at the
    `device_map` assignment for why `auto` cannot be used with bitsandbytes across several GPUs.

    Returns:
        (tokenizer, model) tuple.
    """
    start = time.time()
    if load_in_4bit:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        config = BitsAndBytesConfig(load_in_8bit=True)
    # Use 95% of each GPU's VRAM (accelerate defaults to 90%, which can be too
    # conservative for large models that barely fit across multiple GPUs).
    n_gpus = torch.cuda.device_count()
    max_memory = {
        i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.95 / 1024**3)}GiB"
        for i in range(n_gpus)
    }
    # `sequential`, not `auto`, as soon as there is more than one card, a correctness choice,
    # not a preference. Under `auto`, transformers routes through accelerate's
    # `get_balanced_memory`, which caps every GPU but the last at
    # `model_size // n_gpus + 1.25 * largest_layer`, a budget sized to fit the model with no margin.
    # The bitsandbytes quantizer then multiplies that budget by 0.90
    # (`Bnb8BitHfQuantizer.adjust_max_memory`, "need more space for buffers"), so the plan no
    # longer fits its own model, `infer_auto_device_map` spills the tail to CPU, and the same
    # quantizer rejects the result with "Some modules are dispatched on the CPU or the disk".
    #
    # That message names a shortage that does not exist: a 70B at 8-bit is ~66 GiB and four 24 GB
    # cards still hold ~82 GiB after the haircut. Adding GPUs does not help either, because the
    # balanced cap shrinks with `n_gpus` in lockstep. `sequential` skips `get_balanced_memory`
    # and honours the `max_memory` computed above.
    #
    # For a single GPU the two are identical, `get_balanced_memory` returns `max_memory`
    # untouched when `num_devices == 1` and the caller supplied it, which every caller here does,
    # so this changes nothing for the ~60 single-GPU experiments.
    #
    # Both halves of this were paid for: the RAG-HPO reproduction with the published code on 4x RTX 4090 (jobs 9473960/1, 2026-08-08) and
    # The AutoPCR 70B baseline on 4x RTX 4090 (jobs 9585773/4, 2026-09-12) each died here after ~3 minutes, and
    # The RAG-HPO reproduction with the published code was then worked around by requesting eight cards it did not need.
    device_map = "auto" if n_gpus <= 1 else "sequential"
    model = LlamaForCausalLM.from_pretrained(
        llama_dir,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=config,
        use_cache=False,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(llama_dir)
    tokenizer.pad_token = tokenizer.eos_token
    print("Time to load the model: ", time.time() - start)
    return tokenizer, model


def complete_chat(model, tokenizer, messages: list[dict], **kwargs) -> str:
    """
    Generate a chat completion from a list of messages.

    Args:
        model: Loaded LLaMA model.
        tokenizer: Associated tokenizer.
        messages: List of {role, content} dicts.
        **kwargs: Passed to model.generate (e.g., max_new_tokens). ``temperature`` is consumed
            here rather than forwarded: a positive value switches decoding to sampling at that
            temperature (top_p=1.0, the OpenAI/Groq default), which is what reproducing an
            API-hosted baseline needs. Absent, None or 0 keeps the greedy default every other
            experiment relies on.

    Returns:
        Decoded response string (input tokens excluded, special tokens stripped).
    """
    temperature = kwargs.pop("temperature", None)
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    ).to(model.device)
    num_input_tokens = len(inputs["input_ids"][0])
    if temperature:
        generation_config = GenerationConfig(do_sample=True, temperature=float(temperature),
                                             top_p=1.0)
    else:
        generation_config = GenerationConfig(do_sample=False)
    model.eval()
    with torch.no_grad():
        return tokenizer.decode(
            model.generate(**inputs, **kwargs, generation_config=generation_config)[0][num_input_tokens:],
            skip_special_tokens=True,
        )


def complete_chat_single_turn(model, tokenizer, system_prompt: str, user: str, **kwargs) -> str:
    """
    Single-turn chat: system prompt + one user message.

    Args:
        model: Loaded LLaMA model.
        tokenizer: Associated tokenizer.
        system_prompt: System-level context/instructions.
        user: User message.
        **kwargs: Passed to model.generate.

    Returns:
        Model response string.
    """
    return complete_chat(
        model,
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        **kwargs,
    )


class LlamaLLM:
    """
    LLMProtocol-compatible wrapper around a loaded LLaMA model.
    Instantiate with a pre-loaded (tokenizer, model) pair.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, system_prompt: str, **kwargs) -> str:
        """Generate a reply to *prompt* under *system_prompt*. Extra keyword arguments go to ``generate``."""
        return complete_chat_single_turn(self.model, self.tokenizer, system_prompt, prompt, **kwargs)


class LlamaLogitsLLM:
    """
    LLMProtocol-compatible wrapper that classifies Yes/No via a single forward
    pass instead of autoregressive generation.

    After each generate() call, self.last_margin holds (yes_logit - no_logit). Positive means Yes, magnitude indicates confidence.

    ``last_detail`` additionally exposes the raw scalars behind that margin. The margin
    alone only gives P(Yes | {Yes, No}); ``logsumexp_all`` is needed for the *absolute*
    P(Yes) = exp(logit_yes - logsumexp_all), and ``top1_*`` reveals the case where the
    model would rather emit a third token entirely (the earlier broken-decoder failure,
    which a margin cannot distinguish from a confident verdict).
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.last_margin: float = 0.0
        self.last_detail: dict[str, float] = {}
        self._yes_id = single_token_id(self.tokenizer, "Yes")
        self._no_id = single_token_id(self.tokenizer, "No")

    def _record(self, logits) -> str:
        """Populate last_margin / last_detail from a [vocab_size] logit row."""
        yes_logit = logits[self._yes_id].item()
        no_logit = logits[self._no_id].item()
        # float32 for the reduction: an 8-bit model's logits come back in float16, where
        # logsumexp over ~128k vocab entries can overflow to inf.
        logsumexp_all = torch.logsumexp(logits.float(), dim=-1).item()
        top1_logit, top1_id = torch.max(logits, dim=-1)
        self.last_margin = yes_logit - no_logit
        self.last_detail = {
            "logit_yes": yes_logit,
            "logit_no": no_logit,
            "logsumexp_all": logsumexp_all,
            "top1_token_id": int(top1_id.item()),
            "top1_logit": float(top1_logit.item()),
        }
        return "Yes" if self.last_margin > 0 else "No"

    def generate(self, prompt: str, system_prompt: str, **kwargs) -> str:
        """Return 'Yes' or 'No' via logit comparison. Sets last_margin and last_detail."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids)

        return self._record(outputs.logits[0, -1, :])

    def generate_batch(self, prompts: list[str], system_prompt: str, **kwargs) -> list[dict]:
        """Score several *independent* prompts in one forward pass.

        Semantically identical to calling ``generate`` in a loop, each prompt is its own
        Yes/No question and gets its own logit pair. This is **not** an earlier exploratory run-style prompt
        batching, which concatenated sentences into one prompt and produced a single joint
        margin.

        Left padding is essential: with right padding, position -1 would read a pad token
        for every sequence shorter than the longest one.

        Returns one detail dict per prompt, in input order, each with the keys of
        ``last_detail`` plus ``margin`` and ``verdict``.
        """
        if not prompts:
            return []

        encodings = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p},
                ],
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        max_len = max(len(e) for e in encodings)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids = torch.full((len(encodings), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encodings), max_len), dtype=torch.long)
        for i, ids in enumerate(encodings):
            input_ids[i, max_len - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, max_len - len(ids):] = 1
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        results = []
        for i in range(len(encodings)):
            verdict = self._record(outputs.logits[i, -1, :])
            results.append({**self.last_detail, "margin": self.last_margin, "verdict": verdict})
        return results
