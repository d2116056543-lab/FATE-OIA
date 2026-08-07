from __future__ import annotations

import torch
from torch import Tensor


def entmax15(logits: Tensor, dim: int = -1) -> Tensor:
    """Exact alpha=1.5 entmax with FP32 internal arithmetic."""
    original_dtype = logits.dtype
    x = logits.float() / 2.0
    x = x - x.amax(dim=dim, keepdim=True)
    xsrt, _ = torch.sort(x, dim=dim, descending=True)
    rho_shape = [1] * x.ndim
    rho_shape[dim] = x.shape[dim]
    rho = torch.arange(1, x.shape[dim] + 1, device=x.device, dtype=x.dtype).view(rho_shape)
    mean = xsrt.cumsum(dim) / rho
    mean_sq = xsrt.square().cumsum(dim) / rho
    ss = rho * (mean_sq - mean.square())
    delta = (1.0 - ss) / rho
    # A strictly positive floor avoids the undefined sqrt derivative at a
    # support boundary while preserving the exact sparse projection.
    tau = mean - torch.sqrt(delta.clamp_min(1e-12))
    support = tau <= xsrt
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau_star = tau.gather(dim, support_size - 1)
    output = (x - tau_star).clamp_min(0.0).square()
    output = output / output.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(output.dtype).tiny)
    return output.to(original_dtype)
