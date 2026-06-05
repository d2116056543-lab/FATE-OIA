from __future__ import annotations

import torch


def sparsemax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = scores - scores.max(dim=dim, keepdim=True).values
    zs = torch.sort(z, dim=dim, descending=True).values
    rhos = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    view = [1] * z.ndim
    view[dim] = -1
    rhos = rhos.view(view)
    cumsum = zs.cumsum(dim)
    support = 1 + rhos * zs > cumsum
    k = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (torch.gather(cumsum, dim, k - 1) - 1) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0)


def entmax15(scores: torch.Tensor, dim: int = -1, n_iter: int = 50) -> torch.Tensor:
    """Numerically stable entmax-1.5 projection.

    This is intentionally not an alias for sparsemax. It solves for tau in
    p_i = relu((x_i - tau) / 2)^2 with sum_i p_i = 1.
    """
    x = scores - scores.max(dim=dim, keepdim=True).values
    tau_lo = x.min(dim=dim, keepdim=True).values - 2
    tau_hi = x.max(dim=dim, keepdim=True).values
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2
        p = torch.clamp((x - tau) / 2, min=0).pow(2)
        too_large = p.sum(dim=dim, keepdim=True) > 1
        tau_lo = torch.where(too_large, tau, tau_lo)
        tau_hi = torch.where(too_large, tau_hi, tau)
    p = torch.clamp((x - tau_hi) / 2, min=0).pow(2)
    return p / (p.sum(dim=dim, keepdim=True) + 1e-12)


def relaxed_topk(weights: torch.Tensor, k: int, sorted: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    vals, idx = torch.topk(weights, k=min(k, weights.shape[-1]), dim=-1, sorted=sorted)
    return idx, vals

