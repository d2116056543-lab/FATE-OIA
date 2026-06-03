from __future__ import annotations

import torch


def apply_reason_temperature_bias(reason_logits: torch.Tensor, temperature: torch.Tensor | float, bias: torch.Tensor | float) -> torch.Tensor:
    temp = torch.as_tensor(temperature, dtype=reason_logits.dtype, device=reason_logits.device).clamp_min(1e-4)
    b = torch.as_tensor(bias, dtype=reason_logits.dtype, device=reason_logits.device)
    return reason_logits / temp + b


def warning_calibration_not_ranking() -> dict[str, str | bool]:
    return {
        "calibration_only_warning": True,
        "message": "Bias/temperature calibration can improve thresholded F1 without improving AP/ranking; PSR reports Exp_mAP separately.",
    }
