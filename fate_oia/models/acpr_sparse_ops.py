from __future__ import annotations

import torch


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = logits - logits.max(dim=dim, keepdim=True).values
    zs = torch.sort(z, dim=dim, descending=True).values
    cssv = zs.cumsum(dim) - 1
    rhos = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.ndim
    shape[dim] = -1
    rhos = rhos.view(shape)
    support = (rhos * zs) > cssv
    k = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = cssv.gather(dim, k.long() - 1) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0.0)


def entmax15_bisect(logits: torch.Tensor, dim: int = -1, n_iter: int = 32) -> torch.Tensor:
    # Stable bisection implementation for alpha=1.5 entmax.
    x = logits - logits.max(dim=dim, keepdim=True).values
    tau_lo = x.min(dim=dim, keepdim=True).values - 1.0
    tau_hi = x.max(dim=dim, keepdim=True).values
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2.0
        p = torch.clamp((x - tau) / 2.0, min=0.0).pow(2)
        s = p.sum(dim=dim, keepdim=True)
        tau_lo = torch.where(s > 1.0, tau, tau_lo)
        tau_hi = torch.where(s <= 1.0, tau, tau_hi)
    p = torch.clamp((x - tau_hi) / 2.0, min=0.0).pow(2)
    return p / p.sum(dim=dim, keepdim=True).clamp_min(1e-12)
