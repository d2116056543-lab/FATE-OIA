"""Typed point/curve/region evidence splatting at native patch resolution."""

from __future__ import annotations

from typing import Sequence

import torch


def typed_evidence_splat(
    sampling_coordinates: torch.Tensor,
    sampled_features: torch.Tensor,
    sample_attention: torch.Tensor,
    factor_types: Sequence[str],
    *,
    output_hw: tuple[int, int] = (45, 80),
    coarse_hw: tuple[int, int] = (12, 20),
    eta_by_type: dict[str, float] | None = None,
    max_splat_samples: int | None = 24,
) -> dict[str, torch.Tensor]:
    """Splat typed samples into a fine map and retain coarse comparison maps.

    Coordinates are normalized to ``[-1,1]`` in ``(x,y)`` order.  The kernel
    width is type dependent: points are compact, curves are anisotropic-ish
    through a narrow width, and regions use a wider support.  The operation
    is differentiable with respect to coordinates and sample attention.
    """
    if sampling_coordinates.ndim != 6 or sampling_coordinates.shape[-1] != 2:
        raise ValueError("sampling_coordinates must be [B,F,A,H,S,2]")
    if sampled_features.ndim != 6 or sampled_features.shape[:5] != sampling_coordinates.shape[:5]:
        raise ValueError("sampled_features must align with sampling_coordinates")
    if sample_attention.shape != sampling_coordinates.shape[:-1]:
        raise ValueError("sample_attention must be [B,F,A,H,S]")
    if len(factor_types) != sampling_coordinates.shape[1]:
        raise ValueError("factor_types must match factor dimension")
    if max_splat_samples is not None and (type(max_splat_samples) is not int or max_splat_samples <= 0):
        raise ValueError("max_splat_samples must be a positive integer or None")
    if not output_hw[0] > 0 or not output_hw[1] > 0 or not coarse_hw[0] > 0 or not coarse_hw[1] > 0:
        raise ValueError("splat output sizes must be positive")
    eta = eta_by_type or {"point": 0.70, "object": 0.70, "curve": 0.80, "region": 0.60}
    b, f, a, h, s, d = sampled_features.shape
    fine_y, fine_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, output_hw[0], device=sampled_features.device, dtype=sampled_features.dtype),
        torch.linspace(-1.0, 1.0, output_hw[1], device=sampled_features.device, dtype=sampled_features.dtype),
        indexing="ij",
    )
    coarse_y, coarse_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, coarse_hw[0], device=sampled_features.device, dtype=sampled_features.dtype),
        torch.linspace(-1.0, 1.0, coarse_hw[1], device=sampled_features.device, dtype=sampled_features.dtype),
        indexing="ij",
    )
    fine_mask = sampled_features.new_zeros(b, f, *output_hw)
    coarse_mask = sampled_features.new_zeros(b, f, *coarse_hw)
    fine_feature = sampled_features.new_zeros(b, f, d)
    eps = sampled_features.new_tensor(1e-6)
    flat_coords = sampling_coordinates.reshape(b, f, a * h * s, 2)
    flat_attention = sample_attention.reshape(b, f, a * h * s).clamp_min(0.0)
    flat_features = sampled_features.reshape(b, f, a * h * s, d)
    for factor_index, factor_type in enumerate(factor_types):
        width = {"point": 0.045, "object": 0.055, "curve": 0.030, "region": 0.120}.get(factor_type, 0.060)
        centers = flat_coords[:, factor_index]
        weights = flat_attention[:, factor_index]
        values = flat_features[:, factor_index]
        if max_splat_samples is not None and centers.shape[1] > max_splat_samples:
            # Typed attention is already sparse, but retaining only its top
            # support keeps fine transport bounded at high resolution. This
            # is a real sample selection, not a coarse-mask interpolation.
            _, indices = weights.topk(max_splat_samples, dim=-1)
            centers = centers.gather(1, indices.unsqueeze(-1).expand(-1, -1, 2))
            weights = weights.gather(1, indices)
            values = values.gather(1, indices.unsqueeze(-1).expand(-1, -1, d))
        distance_fine = (fine_x[None, None] - centers[..., 0, None, None]) ** 2 + (
            fine_y[None, None] - centers[..., 1, None, None]
        ) ** 2
        kernel_fine = torch.exp(-distance_fine / (2.0 * width**2))
        norm = weights.sum(-1).clamp_min(eps)
        map_value = (kernel_fine * weights[..., None, None]).sum(1) / norm[:, None, None]
        fine_mask[:, factor_index] = map_value * float(eta.get(factor_type, 0.7))
        distance_coarse = (coarse_x[None, None] - centers[..., 0, None, None]) ** 2 + (
            coarse_y[None, None] - centers[..., 1, None, None]
        ) ** 2
        coarse_kernel = torch.exp(-distance_coarse / (2.0 * (width * 1.8) ** 2))
        coarse_mask[:, factor_index] = (coarse_kernel * weights[..., None, None]).sum(1) / norm[:, None, None]
        fine_feature[:, factor_index] = (values * weights[..., None]).sum(1) / norm[:, None]
    fine_mask = fine_mask / fine_mask.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    coarse_mask = coarse_mask / coarse_mask.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return {"fine_mask": fine_mask, "coarse_mask": coarse_mask, "fine_features": fine_feature}
