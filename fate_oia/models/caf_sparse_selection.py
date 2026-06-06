from __future__ import annotations

import torch


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = logits.transpose(dim, -1)
    orig = z.shape
    z = z.reshape(-1, z.shape[-1])
    z = z - z.max(dim=1, keepdim=True).values
    zs = torch.sort(z, dim=1, descending=True).values
    cssv = zs.cumsum(dim=1) - 1
    k = torch.arange(1, z.shape[1] + 1, device=z.device, dtype=z.dtype).view(1, -1)
    cond = zs - cssv / k > 0
    k_z = cond.sum(dim=1, keepdim=True).clamp_min(1)
    tau = cssv.gather(1, k_z.long() - 1) / k_z
    p = torch.clamp(z - tau, min=0.0)
    return p.reshape(orig).transpose(dim, -1)
