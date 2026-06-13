from __future__ import annotations

import torch


def sparsemax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = input - input.max(dim=dim, keepdim=True).values
    zs = torch.sort(z, dim=dim, descending=True).values
    range_shape = [1] * z.dim()
    range_shape[dim] = z.size(dim)
    rhos = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype).view(range_shape)
    cumsum = zs.cumsum(dim)
    support = 1 + rhos * zs > cumsum
    k = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (cumsum.gather(dim, k - 1) - 1) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0)


def entmax15_bisect(input: torch.Tensor, dim: int = -1, n_iter: int = 50) -> torch.Tensor:
    # Stable differentiable bisection for alpha=1.5 entmax.
    x = input - input.max(dim=dim, keepdim=True).values
    tau_lo = x.min(dim=dim, keepdim=True).values - 1.0
    tau_hi = x.max(dim=dim, keepdim=True).values
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2
        p = torch.clamp((x - tau) / 2, min=0) ** 2
        mass = p.sum(dim=dim, keepdim=True)
        tau_lo = torch.where(mass > 1, tau, tau_lo)
        tau_hi = torch.where(mass > 1, tau_hi, tau)
    tau = tau_lo
    p = torch.clamp((x - tau) / 2, min=0) ** 2
    return p / p.sum(dim=dim, keepdim=True).clamp_min(1e-12)
