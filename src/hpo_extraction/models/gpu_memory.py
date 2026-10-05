"""Peak GPU memory of ONE process, summed over every card it can see.

``torch.cuda.max_memory_allocated()`` with no argument reads the *current* device only. A model
loaded with ``device_map="auto"`` across four RTX 4090s leaves roughly a quarter of its weights on
device 0, so that call reports a 70B linker as an ~18 GB system -- beside RAG-HPO 70B's ~70 GB
measured on one card, for the same model at the same precision. Summing over the visible
devices gives the footprint the model actually needed, which is the comparable quantity.

This is the sum within one process. Separate jobs (tree shards, jurors of an array) are separate
processes on separate cards, and the comparison's cost collector takes the MAXIMUM over those -- adding
them would describe hardware no single run used.
"""

from __future__ import annotations


def _torch(torch=None):
    if torch is not None:
        return torch
    try:
        import torch as _t
    except ImportError:  # pragma: no cover - CPU-only envs
        return None
    return _t


def reset_peak_all_devices(torch=None) -> None:
    """Reset the peak counter on every visible card, not just the current one."""
    t = _torch(torch)
    if t is None or not t.cuda.is_available():
        return
    for device in range(t.cuda.device_count()):
        t.cuda.reset_peak_memory_stats(device)


def peak_gpu_bytes_all_devices(torch=None) -> int:
    """Sum of ``max_memory_allocated`` over every visible card; 0 without CUDA."""
    t = _torch(torch)
    if t is None or not t.cuda.is_available():
        return 0
    return int(sum(t.cuda.max_memory_allocated(device)
                   for device in range(t.cuda.device_count())))
