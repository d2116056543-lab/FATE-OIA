from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class METERRuntimeProfile:
    batch_size: int
    gradient_accumulation_steps: int
    num_workers: int = 4
    prefetch_factor: int = 2
    reserved_gb: float = 0.0
    samples_per_sec: float = 0.0

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


def effective_cuda_memory_gb(
    *,
    allocator_reserved_gb: float,
    physical_used_gb: float,
    physical_total_gb: float,
) -> float:
    """Return a conservative usable memory reading across CUDA/WDDM.

    On Windows WDDM, PyTorch allocator accounting can include shared virtual
    memory and exceed the physical device capacity. In that specific case the
    device-level used-memory reading is the only physically meaningful value.
    Otherwise retain the conservative maximum of allocator and device usage.
    """
    allocator = float(allocator_reserved_gb)
    physical_used = float(physical_used_gb)
    physical_total = float(physical_total_gb)
    if allocator > physical_total + 0.25:
        return physical_used
    return max(allocator, physical_used)


def choose_meter_profile(profiles: Iterable[METERRuntimeProfile], hard_max_reserved_gb: float = 45.0, preferred_gb: float = 42.0) -> METERRuntimeProfile:
    valid = [profile for profile in profiles if profile.reserved_gb < hard_max_reserved_gb]
    if not valid:
        raise RuntimeError("No METER runtime profile satisfies the memory hard limit")
    preferred = [profile for profile in valid if profile.reserved_gb <= preferred_gb]
    pool = preferred or valid
    return max(pool, key=lambda profile: (profile.samples_per_sec, -profile.reserved_gb))
