from __future__ import annotations

import torch


def sparsemax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = scores - scores.max(dim=dim, keepdim=True).values
    zs = torch.sort(z, dim=dim, descending=True).values
    shape = [1] * z.dim()
    shape[dim] = z.size(dim)
    r = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype).view(shape)
    cumsum = zs.cumsum(dim)
    support = 1 + r * zs > cumsum
    k = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (cumsum.gather(dim, k.long() - 1) - 1) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0)


def entmax15(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return sparsemax(scores, dim=dim)


def relaxed_topk(weights: torch.Tensor, k: int, straight_through: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    vals, idx = torch.topk(weights, k=k, dim=-1)
    hard = torch.zeros_like(weights).scatter_(-1, idx, vals)
    selected = hard + weights - weights.detach() if straight_through else hard
    return idx, torch.gather(selected, -1, idx)
