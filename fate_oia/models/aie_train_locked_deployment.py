from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn


class AIETrainLockedDeployment(nn.Module):
    """Checkpointed train-only deployment boundary for an AIE model."""

    def __init__(
        self,
        model: nn.Module,
        action_scales: Sequence[float],
        threshold_prob: Sequence[float],
        *,
        action_dim: int = 4,
        reason_dim: int = 21,
        reason_action_scale: float = 0.0,
        reason_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.reason_action_scale = float(reason_action_scale)
        self.reason_scale = float(reason_scale)
        scales = torch.as_tensor(action_scales, dtype=torch.float32).view(-1)
        thresholds = torch.as_tensor(threshold_prob, dtype=torch.float32).view(-1)
        if scales.numel() != self.action_dim:
            raise ValueError(f"action_scales must have {self.action_dim} entries")
        if thresholds.numel() != self.action_dim + self.reason_dim:
            raise ValueError(f"threshold_prob must have {self.action_dim + self.reason_dim} entries")
        if not bool(torch.isfinite(scales).all()) or not bool(torch.isfinite(thresholds).all()):
            raise ValueError("deployment parameters must be finite")
        if bool(((thresholds <= 0.0) | (thresholds >= 1.0)).any()):
            raise ValueError("threshold probabilities must lie strictly between zero and one")
        self.register_buffer("action_scales", scales)
        self.register_buffer("threshold_prob", thresholds)

    def forward(self, images: Tensor, *, reason_scale: Optional[float] = None) -> dict[str, Any]:
        effective_reason_scale = self.reason_scale if reason_scale is None else float(reason_scale)
        field = self.model.encode_images(images)
        output = self.model.decode_from_field(
            field,
            action_scale=self.action_scales,
            reason_scale=effective_reason_scale,
            reason_action_scale=self.reason_action_scale,
        )
        final_logits = torch.cat(
            (output["action_logits_final"], output["reason_logits_final"]), dim=-1
        )
        threshold_logit = torch.logit(self.threshold_prob).to(final_logits)
        deploy_logits = final_logits - threshold_logit.view(1, -1)
        return {
            **output,
            "action_logits_deploy": deploy_logits[:, : self.action_dim],
            "reason_logits_deploy": deploy_logits[:, self.action_dim :],
            "deployment_action_scales": self.action_scales,
            "deployment_threshold_prob": self.threshold_prob,
            "deployment_threshold_logit": threshold_logit,
            "deployment_reason_action_scale": torch.as_tensor(
                self.reason_action_scale, device=final_logits.device, dtype=final_logits.dtype
            ),
            "deployment_reason_scale": torch.as_tensor(
                effective_reason_scale, device=final_logits.device, dtype=final_logits.dtype
            ),
        }

    def deployment_state(self) -> Mapping[str, Tensor]:
        return {
            "action_scales": self.action_scales.detach().cpu(),
            "threshold_prob": self.threshold_prob.detach().cpu(),
            "reason_action_scale": torch.tensor(self.reason_action_scale),
            "reason_scale": torch.tensor(self.reason_scale),
        }
