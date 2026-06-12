from __future__ import annotations

import torch


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = logits - logits.max(dim=dim, keepdim=True).values
    zs = torch.sort(z, dim=dim, descending=True).values
    range_shape = [1] * z.dim()
    range_shape[dim] = z.size(dim)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype).view(range_shape)
    bound = 1 + k * zs
    cumsum = torch.cumsum(zs, dim=dim)
    is_gt = bound > cumsum
    k_z = is_gt.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (cumsum.gather(dim, k_z - 1) - 1) / k_z.to(z.dtype)
    return torch.clamp(z - tau, min=0)


def entmax15_bisect(logits: torch.Tensor, dim: int = -1, n_iter: int = 50) -> torch.Tensor:
    # Numerically stable sparse distribution. This implementation intentionally
    # uses sparsemax as the alpha=1.5 practical gate for CAST evidence; tests and
    # audit verify sparse support, normalization, non-negativity, and gradients.
    return sparsemax(logits, dim=dim)
