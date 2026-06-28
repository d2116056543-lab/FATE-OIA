from __future__ import annotations

import torch
from torch import nn

from .pmcal_reason_formula_bank import PMCalReasonFormulaBank


def _soft_or(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() <= 0:
        return torch.zeros(x.shape[0], mask.shape[0], device=x.device, dtype=x.dtype)
    xm = x.unsqueeze(1) * mask.to(x.device, x.dtype).unsqueeze(0)
    return 1.0 - torch.prod(1.0 - xm.clamp(0, 1) + (1.0 - mask.to(x.device, x.dtype)).unsqueeze(0), dim=-1)


class PMCalReasonFormulaHead(nn.Module):
    def __init__(self, formula_bank: PMCalReasonFormulaBank, cap: float = 0.20, gate_max: float = 0.35) -> None:
        super().__init__()
        self.formula_bank = formula_bank
        self.cap = float(cap)
        self.gate_max = float(gate_max)
        self.register_buffer("positive_matrix", formula_bank.positive_matrix)
        self.register_buffer("contradiction_matrix", formula_bank.contradiction_matrix)
        self.bias = nn.Parameter(torch.zeros(21))
        self.w_pos = nn.Parameter(torch.ones(21))
        self.w_neg = nn.Parameter(torch.ones(21))
        self.gate_param = nn.Parameter(torch.full((21,), -1.5))

    def forward(self, q_pred: torch.Tensor, rho_pred: torch.Tensor) -> dict[str, torch.Tensor]:
        support_score = _soft_or(q_pred * rho_pred, self.positive_matrix)
        contra_score = _soft_or(q_pred * rho_pred, self.contradiction_matrix)
        eps = 1e-5
        pos_logit = torch.logit(support_score.clamp(eps, 1 - eps))
        neg_logit = torch.logit(contra_score.clamp(eps, 1 - eps))
        formula_logits = self.w_pos.view(1, -1) * pos_logit - self.w_neg.view(1, -1) * neg_logit + self.bias.view(1, -1)
        formula_logits = formula_logits.clamp(-self.cap, self.cap)
        formula_confidence = (support_score + contra_score).clamp(0, 1)
        gate = torch.sigmoid(self.gate_param).view(1, -1).clamp(max=self.gate_max) * formula_confidence
        return {
            "reason_formula_logits": formula_logits,
            "reason_formula_gate": gate,
            "support_score": support_score,
            "contra_score": contra_score,
            "formula_confidence": formula_confidence,
        }
