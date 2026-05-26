from __future__ import annotations

import torch


def logit_adjustment_from_prior(prior: torch.Tensor, tau: float = 0.3, clamp: float = 8.0) -> torch.Tensor:
    """Compute tau * log(p/(1-p)) for multi-label logit adjustment."""
    p = prior.float().clamp(1e-6, 1.0 - 1e-6)
    return (float(tau) * torch.log(p / (1.0 - p))).clamp(min=-float(clamp), max=float(clamp))


def apply_logit_adjustment(logits: torch.Tensor, adjustment: torch.Tensor, sign: str = "subtract") -> torch.Tensor:
    if sign == "subtract":
        return logits - adjustment.to(logits.device).view(1, -1)
    if sign == "add":
        return logits + adjustment.to(logits.device).view(1, -1)
    raise ValueError("sign must be 'subtract' or 'add'")

