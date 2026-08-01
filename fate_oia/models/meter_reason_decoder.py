from __future__ import annotations

import torch
from torch import Tensor, nn

from .meter_semantic_action import heca_credit_ramp


class METERPrivateReasonDecoder(nn.Module):
    """CalAlign global reason anchor plus detached, groundable correction."""

    def __init__(
        self,
        dim: int = 384,
        reason_dim: int = 21,
        action_dim: int = 4,
        max_correction: float = 0.50,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.reason_dim = int(reason_dim)
        self.global_delta_head = nn.Linear(dim, 1)
        nn.init.zeros_(self.global_delta_head.weight)
        nn.init.zeros_(self.global_delta_head.bias)
        # A zero evidence residual preserves the calibrated global reason
        # anchor exactly; evidence gradients first update this vector.
        self.correction_vector = nn.Parameter(torch.zeros(reason_dim, dim))
        self.correction_kappa_raw = nn.Parameter(
            torch.full((reason_dim,), -2.2521685)
        )
        self.max_correction = float(max_correction)

    def initialize_from_foundation(self, foundation: nn.Module) -> None:
        # The complete foundation predictor remains the explicit anchor. HECA
        # initializes only a zero residual, so every q/k/v/attention/norm/head
        # behavior is preserved rather than partially copied into a new decoder.
        if not hasattr(foundation, "trunk"):
            raise ValueError("HECA reason initialization requires a CalAlign trunk")

    def forward(
        self,
        *,
        reason_logits_calalign: Tensor,
        reason_nodes: Tensor,
        factor_measurement_token: Tensor,
        factor_reliability: Tensor,
        factor_groundable_mask: Tensor,
        progress: float = 1.0,
        **_: Tensor,
    ) -> dict[str, Tensor]:
        global_delta = self.global_delta_head(reason_nodes).squeeze(-1)
        global_logits = reason_logits_calalign + global_delta
        evidence = factor_measurement_token.detach()
        reliability = factor_reliability.detach().clamp(0.0, 1.0)
        groundable = factor_groundable_mask.to(evidence).view(1, -1)
        raw = torch.einsum("brd,rd->br", evidence, self.correction_vector)
        kappa = torch.nn.functional.softplus(self.correction_kappa_raw).clamp(
            0.02, self.max_correction
        )
        correction = (
            groundable
            * reliability
            * kappa.view(1, -1)
            * torch.tanh(raw / kappa.view(1, -1).clamp_min(1e-6))
        )
        ramp = heca_credit_ramp(progress)
        final = global_logits + ramp * correction
        global_rms = global_logits.detach().float().square().mean(0).sqrt()
        correction_rms = correction.detach().float().square().mean(0).sqrt()
        return {
            "reason_global_tokens": reason_nodes,
            "reason_view_embedding": reason_nodes,
            "reason_logits_global": global_logits,
            "reason_evidence_delta": correction,
            "reason_logits_final": final,
            "reason_correction_kappa": kappa,
            "reason_groundable_mask": groundable.squeeze(0),
            "reason_correction_rms_ratio": correction_rms
            / global_rms.clamp_min(1e-6),
        }
