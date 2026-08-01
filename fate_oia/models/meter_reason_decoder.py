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
        max_global_delta: float = 0.05,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.reason_dim = int(reason_dim)
        self.global_delta_head = nn.Linear(dim, 1)
        # The gate below, rather than this projection, establishes the exact
        # zero-effect initialization. A live projection lets the adapter-only
        # input receive a useful first-step gradient.
        nn.init.xavier_uniform_(self.global_delta_head.weight)
        nn.init.zeros_(self.global_delta_head.bias)
        self.global_delta_gate_raw = nn.Parameter(torch.zeros(()))
        self.global_delta_startup_gradient_scale = 0.10
        # A zero evidence residual preserves the calibrated global reason
        # anchor exactly; evidence gradients first update this vector.
        self.correction_vector = nn.Parameter(torch.zeros(reason_dim, dim))
        self.correction_kappa_raw = nn.Parameter(
            torch.full((reason_dim,), -2.2521685)
        )
        self.max_correction = float(max_correction)
        if float(max_global_delta) <= 0.0:
            raise ValueError("HECA max_global_delta must be positive")
        # CalAlign owns static label calibration. The private global reader is
        # therefore permitted to resolve only small, sample-specific ranking
        # errors; an unrestricted residual becomes a second threshold head.
        self.max_global_delta = float(max_global_delta)
        self.evidence_mean_momentum = 0.95
        self.register_buffer(
            "running_evidence_mean", torch.zeros(reason_dim), persistent=True
        )
        self.register_buffer(
            "evidence_mean_updates", torch.zeros((), dtype=torch.long), persistent=True
        )

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
        update_running_stats: bool = False,
        **_: Tensor,
    ) -> dict[str, Tensor]:
        # The CalAlign predictor is a fixed reason anchor. ``reason_nodes``
        # is supplied through the adapter-only route, so this head can train
        # the shared/private reason adapters without rewriting the foundation.
        raw_global_delta = self.global_delta_head(reason_nodes).squeeze(-1)
        gate = 0.20 * torch.tanh(self.global_delta_gate_raw)
        # The subtract-detach term is exactly zero in the forward pass at
        # startup, yet gives the zero-initialized head and adapter-only input
        # a bounded first-step gradient. The learned gate alone determines
        # every deployed logit delta.
        unbounded_global_delta = (
            gate * raw_global_delta
            + (1.0 - gate)
            * self.global_delta_startup_gradient_scale
            * (raw_global_delta - raw_global_delta.detach())
        )
        global_delta = self.max_global_delta * torch.tanh(
            unbounded_global_delta / self.max_global_delta
        )
        global_logits = reason_logits_calalign.detach() + global_delta
        evidence = factor_measurement_token.detach()
        reliability = factor_reliability.detach().clamp(0.0, 1.0)
        groundable = factor_groundable_mask.to(evidence).view(1, -1)
        raw = torch.einsum("brd,rd->br", evidence, self.correction_vector)
        if self.training and update_running_stats:
            with torch.no_grad():
                batch_mean = raw.detach().float().mean(0).to(self.running_evidence_mean)
                if int(self.evidence_mean_updates) == 0:
                    self.running_evidence_mean.copy_(batch_mean)
                else:
                    self.running_evidence_mean.mul_(self.evidence_mean_momentum).add_(
                        batch_mean * (1.0 - self.evidence_mean_momentum)
                    )
                self.evidence_mean_updates.add_(1)
        # CalAlign owns static label-level calibration. The private residual
        # therefore carries only sample-relative factor evidence, which can
        # improve ranking instead of learning a duplicate threshold shift.
        centered_raw = raw - self.running_evidence_mean.detach().to(raw)
        kappa = torch.nn.functional.softplus(self.correction_kappa_raw).clamp(
            0.02, self.max_correction
        )
        correction = (
            groundable
            * reliability
            * kappa.view(1, -1)
            * torch.tanh(centered_raw / kappa.view(1, -1).clamp_min(1e-6))
        )
        ramp = heca_credit_ramp(progress)
        final = global_logits + ramp * correction
        global_rms = global_logits.detach().float().square().mean(0).sqrt()
        correction_rms = correction.detach().float().square().mean(0).sqrt()
        return {
            "reason_global_tokens": reason_nodes,
            "reason_view_embedding": reason_nodes,
            "reason_logits_global": global_logits,
            "reason_global_delta": global_delta,
            "reason_global_delta_cap": reason_logits_calalign.new_tensor(
                self.max_global_delta
            ),
            "reason_evidence_delta": correction,
            "reason_evidence_centered": centered_raw,
            "reason_evidence_running_mean": self.running_evidence_mean.detach(),
            "reason_logits_final": final,
            "reason_correction_kappa": kappa,
            "reason_groundable_mask": groundable.squeeze(0),
            "reason_correction_rms_ratio": correction_rms
            / global_rms.clamp_min(1e-6),
        }
