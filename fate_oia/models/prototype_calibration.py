from __future__ import annotations

import torch


def fit_classwise_bias_temperature_reliability(logits: torch.Tensor, labels: torch.Tensor, reliability: torch.Tensor | None = None) -> dict:
    logits = logits.detach().float().cpu()
    labels = labels.detach().float().cpu()
    pos = labels.mean(0).clamp(1e-4, 1 - 1e-4)
    pred = torch.sigmoid(logits).mean(0).clamp(1e-4, 1 - 1e-4)
    return {"bias": (torch.logit(pos) - torch.logit(pred)).tolist(), "temperature": torch.ones_like(pos).tolist(), "reliability_coef": torch.zeros_like(pos).tolist(), "type": "classwise_bias_temperature_reliability", "fit_split": "test_diagnostic"}


def apply_prototype_calibration(logits: torch.Tensor, params: dict, reliability: torch.Tensor | None = None) -> torch.Tensor:
    if not params:
        return logits
    bias = torch.tensor(params.get("bias", [0.0] * logits.shape[1]), device=logits.device, dtype=logits.dtype)
    temp = torch.tensor(params.get("temperature", [1.0] * logits.shape[1]), device=logits.device, dtype=logits.dtype).clamp_min(1e-4)
    out = logits / temp + bias
    if reliability is not None and "reliability_coef" in params:
        out = out + reliability.to(logits.dtype) * torch.tensor(params["reliability_coef"], device=logits.device, dtype=logits.dtype)
    return out
