from __future__ import annotations

import time
from contextlib import contextmanager

import torch


@contextmanager
def timed(bucket: dict[str, float], key: str):
    start = time.perf_counter()
    yield
    bucket[key] = bucket.get(key, 0.0) + time.perf_counter() - start


def gpu_memory_gb(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"gpu_allocated_gb": 0.0, "gpu_reserved_gb": 0.0}
    return {"gpu_allocated_gb": torch.cuda.memory_allocated(device) / 1024 ** 3, "gpu_reserved_gb": torch.cuda.memory_reserved(device) / 1024 ** 3}
