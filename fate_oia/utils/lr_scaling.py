from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LRScalingResult:
    num_gpus: int
    per_gpu_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    reference_effective_batch: int
    base_lr_at_reference_batch: float
    lr_actual: float
    loss_divided_by_accumulation: bool = True


def effective_batch_size(
    per_gpu_batch_size: int,
    gradient_accumulation_steps: int,
    num_gpus: int = 1,
) -> int:
    return int(max(1, num_gpus) * max(1, per_gpu_batch_size) * max(1, gradient_accumulation_steps))


def scale_lr_linear(
    base_lr_at_reference_batch: float,
    effective_batch: int,
    reference_effective_batch: int,
    max_lr: float | None = None,
) -> float:
    if reference_effective_batch <= 0:
        raise ValueError("reference_effective_batch must be positive")
    lr = float(base_lr_at_reference_batch) * float(effective_batch) / float(reference_effective_batch)
    if max_lr is not None:
        lr = min(lr, float(max_lr))
    return lr


def compute_lr_scaling(
    per_gpu_batch_size: int,
    gradient_accumulation_steps: int,
    reference_effective_batch: int,
    base_lr_at_reference_batch: float,
    *,
    num_gpus: int = 1,
    auto_scale_lr: bool = True,
    current_lr: float | None = None,
    max_lr: float | None = None,
) -> LRScalingResult:
    eff = effective_batch_size(per_gpu_batch_size, gradient_accumulation_steps, num_gpus)
    lr_actual = (
        scale_lr_linear(base_lr_at_reference_batch, eff, reference_effective_batch, max_lr)
        if auto_scale_lr
        else float(current_lr if current_lr is not None else base_lr_at_reference_batch)
    )
    return LRScalingResult(
        num_gpus=int(max(1, num_gpus)),
        per_gpu_batch_size=int(max(1, per_gpu_batch_size)),
        gradient_accumulation_steps=int(max(1, gradient_accumulation_steps)),
        effective_batch_size=int(eff),
        reference_effective_batch=int(reference_effective_batch),
        base_lr_at_reference_batch=float(base_lr_at_reference_batch),
        lr_actual=float(lr_actual),
    )
