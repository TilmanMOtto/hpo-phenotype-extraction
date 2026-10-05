"""A multi-GPU bitsandbytes load must not be planned with ``device_map="auto"``.

Regression test for a failure that cost two experiments three minutes each and one of them a
permanent 2x over-allocation.

``device_map="auto"`` sends transformers through accelerate's ``get_balanced_memory``, which caps
every GPU but the last at ``model_size // n_gpus + 1.25 * largest_layer``, a budget sized to fit
the model with no margin. ``Bnb8BitHfQuantizer.adjust_max_memory`` then multiplies that budget by 0.90
("need more space for buffers"), so the plan no longer fits its own model,
``infer_auto_device_map`` spills the tail onto CPU, and the same quantizer rejects the result:

    ValueError: Some modules are dispatched on the CPU or the disk. Make sure you have enough
    GPU RAM to fit the quantized model.

The message names a shortage that is not there. A 70B at 8-bit is ~66 GiB. Four 24 GB cards hold
~82 GiB *after* the haircut. Nor does adding cards help, the balanced cap shrinks with ``n_gpus``
in lockstep, which is why ``raghpo_reproduction_published_code`` only stopped failing at eight GPUs and would have run on
four. Casualties: ``raghpo_reproduction_published_code`` jobs 9473960/9473961 (2026-08-08) and ``baseline_autopcr_70b`` jobs
9585773/9585774 (2026-09-12).

``sequential`` skips ``get_balanced_memory`` and honours the ``max_memory`` the loader computes.
On one GPU the two are identical, because ``get_balanced_memory`` returns ``max_memory`` untouched
when ``num_devices == 1`` and the caller supplied it, so the single-GPU path is asserted to be
unchanged rather than merely assumed to be.

``torch`` is not installed in the test environment (and a GPU is not available where it is), so
``torch`` and ``transformers`` are stubbed and the loader is inspected through the kwargs it hands
to ``from_pretrained``.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest

MODULE = "hpo_extraction.models.llama"

# RTX 4090, the card the RAG-HPO reproduction with the published code / the AutoPCR 70B baseline actually run on. Only its magnitude counts here: the
# loader formats ``total_memory`` into a GiB string, so it has to be a real int.
RTX_4090_BYTES = 25_757_220_864


def _load_llama_kwargs(n_gpus: int) -> dict:
    """Import the loader against stubbed torch/transformers and return the load kwargs."""
    torch_stub = MagicMock()
    torch_stub.cuda.device_count.return_value = n_gpus
    torch_stub.cuda.get_device_properties.return_value.total_memory = RTX_4090_BYTES

    transformers_stub = MagicMock()

    saved = {k: sys.modules.get(k) for k in ("torch", "transformers", MODULE)}
    sys.modules["torch"] = torch_stub
    sys.modules["transformers"] = transformers_stub
    sys.modules.pop(MODULE, None)
    try:
        llama = importlib.import_module(MODULE)
        llama.load_llama("/nonexistent/llama-dir")
        call = transformers_stub.LlamaForCausalLM.from_pretrained.call_args
        assert call is not None, "the loader never called from_pretrained"
        return call.kwargs
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class TestDeviceMap:
    @pytest.mark.parametrize("n_gpus", [2, 4, 8])
    def test_multi_gpu_is_not_auto(self, n_gpus: int):
        """`auto` + bitsandbytes + >1 GPU is the bug. Anything else is a deliberate choice."""
        assert _load_llama_kwargs(n_gpus)["device_map"] != "auto"

    @pytest.mark.parametrize("n_gpus", [2, 4, 8])
    def test_multi_gpu_is_sequential(self, n_gpus: int):
        assert _load_llama_kwargs(n_gpus)["device_map"] == "sequential"

    def test_single_gpu_is_unchanged(self):
        """One GPU keeps `auto`: identical behaviour, and ~60 experiments depend on this path."""
        assert _load_llama_kwargs(1)["device_map"] == "auto"

    @pytest.mark.parametrize("n_gpus", [1, 4])
    def test_max_memory_is_still_supplied_for_every_card(self, n_gpus: int):
        """`sequential` only helps because an explicit per-GPU budget reaches `get_max_memory`."""
        max_memory = _load_llama_kwargs(n_gpus)["max_memory"]
        assert sorted(max_memory) == list(range(n_gpus))
        # 0.95 * 23.99 GiB, floored by the loader's int() -> 22 GiB.
        assert set(max_memory.values()) == {"22GiB"}
