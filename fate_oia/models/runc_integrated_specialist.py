from __future__ import annotations

import math

import torch
from torch import nn

from fate_oia.models.action_set_head import ActionSetHead
from fate_oia.models.evidence_auxiliary import EvidenceAuxiliary
from fate_oia.models.reason_visual_specialist import ReasonVisualSpecialist


class RunCIntegratedSpecialist(nn.Module):
    """Additive specialist wrapper around the strict Run C FATE-OIA head."""

    def __init__(
        self,
        base_fate_head: nn.Module,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        pattern_matrix: torch.Tensor | None = None,
        alpha_max: float = 0.25,
        alpha_init: float = 0.05,
        enable_reason_specialist: bool = True,
        enable_action_set_head: bool = True,
        enable_evidence_aux: bool = False,
    ) -> None:
        super().__init__()
        self.base_fate_head = base_fate_head
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.enable_reason_specialist = bool(enable_reason_specialist)
        self.enable_action_set_head = bool(enable_action_set_head)
        self.reason_specialist = ReasonVisualSpecialist(dim=dim, reason_dim=reason_dim) if enable_reason_specialist else None
        self.reason_bias = nn.Parameter(torch.zeros(reason_dim))
        self.action_set_head = ActionSetHead(dim=dim, reason_dim=reason_dim, action_dim=action_dim, pattern_matrix=pattern_matrix) if enable_action_set_head else None
        self.alpha_max = float(alpha_max)
        ratio = min(0.99, max(1e-4, float(alpha_init) / max(self.alpha_max, 1e-6)))
        self.action_alpha_raw = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32))
        self.evidence_aux = EvidenceAuxiliary(dim=dim, reason_dim=reason_dim) if enable_evidence_aux else None

    def action_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.action_alpha_raw) * self.alpha_max

    def forward(self, tokens: torch.Tensor, evidence: dict | None = None) -> dict[str, torch.Tensor | dict]:
        base = self.base_fate_head(tokens)
        if "label_tokens" not in base:
            raise KeyError("Run C base output must include label_tokens")
        base_action = base["action_logits"]
        base_reason = base["reason_logits"]
        evidence_tokens = None if evidence is None else evidence.get("tokens")
        if self.reason_specialist is not None:
            reason_delta, reason_diag = self.reason_specialist(
                visual_tokens=tokens,
                label_tokens=base["label_tokens"],
                base_reason_logits=base_reason,
                attention=base.get("attention"),
                evidence_tokens=evidence_tokens,
            )
        else:
            reason_delta = torch.zeros_like(base_reason)
            reason_diag = {"reason_delta_abs_mean": 0.0, "reason_delta_abs_max": 0.0}
        final_reason = base_reason + reason_delta - self.reason_bias.view(1, -1)
        if self.action_set_head is not None:
            pattern_logits, pattern_action_logits = self.action_set_head(base_action, final_reason, base["label_tokens"])
            alpha = self.action_alpha().to(dtype=base_action.dtype, device=base_action.device)
            final_action = (1.0 - alpha) * base_action + alpha * pattern_action_logits
        else:
            pattern_logits = base_action.new_zeros(base_action.shape[0], 1)
            pattern_action_logits = base_action
            alpha = base_action.new_zeros(())
            final_action = base_action
        evidence_reason_logits = None
        evidence_diag = {"evidence_count": 0, "evidence_available": 0}
        if self.evidence_aux is not None:
            evidence_reason_logits, evidence_diag = self.evidence_aux(evidence_tokens)
        diagnostics = {
            **reason_diag,
            **evidence_diag,
            "action_set_alpha": float(alpha.detach().item()) if alpha.ndim == 0 else float(alpha.detach().mean().item()),
            "reason_bias_abs_mean": float(self.reason_bias.detach().abs().mean().item()),
        }
        return {
            **base,
            "final_action_logits": final_action,
            "final_reason_logits": final_reason,
            "base_action_logits": base_action,
            "base_reason_logits": base_reason,
            "action_logits": final_action,
            "reason_logits": final_reason,
            "reason_delta_logits": reason_delta,
            "pattern_logits": pattern_logits,
            "pattern_action_logits": pattern_action_logits,
            "action_set_alpha": alpha,
            "reason_bias": self.reason_bias,
            "evidence_reason_logits": evidence_reason_logits,
            "diagnostics": diagnostics,
        }
