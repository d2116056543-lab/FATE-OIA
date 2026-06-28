from __future__ import annotations

import torch

from .pmcal_reason_formula_bank import PMCalReasonFormulaBank


class PMCalPUReasonState:
    def __init__(
        self,
        formula_bank: PMCalReasonFormulaBank,
        tail_indices: list[int] | None = None,
        support_threshold: float = 0.25,
        contradiction_threshold: float = 0.55,
        reliability_threshold: float = 0.55,
    ) -> None:
        self.formula_bank = formula_bank
        self.tail_indices = tail_indices or [12, 9, 5, 14, 6, 11, 10, 13]
        self.support_threshold = float(support_threshold)
        self.contradiction_threshold = float(contradiction_threshold)
        self.reliability_threshold = float(reliability_threshold)

    def build(self, reason_labels: torch.Tensor, q_pred: torch.Tensor, rho_pred: torch.Tensor) -> dict[str, torch.Tensor]:
        pos_mat = self.formula_bank.positive_matrix.to(q_pred.device, q_pred.dtype)
        neg_mat = self.formula_bank.contradiction_matrix.to(q_pred.device, q_pred.dtype)
        support = torch.einsum("bm,rm->br", q_pred * rho_pred, pos_mat) / pos_mat.sum(-1).clamp_min(1.0).view(1, -1)
        contra = torch.einsum("bm,rm->br", q_pred * rho_pred, neg_mat) / neg_mat.sum(-1).clamp_min(1.0).view(1, -1)
        reliability = (support + contra).clamp(0, 1)
        positive = reason_labels.float() > 0.5
        reliable_negative = (~positive) & (support < self.support_threshold) & (contra > self.contradiction_threshold) & (reliability > self.reliability_threshold)
        unknown = (~positive) & (~reliable_negative)
        pu_state_id = torch.zeros_like(reason_labels, dtype=torch.long)
        pu_state_id[positive] = 1
        pu_state_id[reliable_negative] = -1
        return {
            "positive_mask": positive.float(),
            "unknown_mask": unknown.float(),
            "reliable_negative_mask": reliable_negative.float(),
            "reason_reliability": reliability,
            "support_score": support,
            "contra_score": contra,
            "pu_state_id": pu_state_id,
        }
