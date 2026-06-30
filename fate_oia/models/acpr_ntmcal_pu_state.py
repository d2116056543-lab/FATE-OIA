from __future__ import annotations

import torch


class NativeTextPUReasonState:
    def __init__(self, support_matrix: torch.Tensor, contra_matrix: torch.Tensor, soft_negative_start_epoch: int = 3, hard_negative_start_epoch: int = 7, support_threshold: float = 0.30, contra_threshold: float = 0.55, rho_threshold: float = 0.50) -> None:
        self.support_matrix = support_matrix.float()
        self.contra_matrix = contra_matrix.float()
        self.soft_negative_start_epoch = int(soft_negative_start_epoch)
        self.hard_negative_start_epoch = int(hard_negative_start_epoch)
        self.support_threshold = float(support_threshold)
        self.contra_threshold = float(contra_threshold)
        self.rho_threshold = float(rho_threshold)

    def __call__(self, reason_labels: torch.Tensor | None, q_pred: torch.Tensor, rho_pred: torch.Tensor, epoch: int) -> dict[str, torch.Tensor | dict]:
        device = q_pred.device
        support_mat = self.support_matrix.to(device)
        contra_mat = self.contra_matrix.to(device)
        support_den = support_mat.sum(-1).clamp_min(1.0)
        contra_den = contra_mat.sum(-1).clamp_min(1.0)
        support_score = q_pred @ support_mat.t() / support_den.view(1, -1)
        contra_score = q_pred @ contra_mat.t() / contra_den.view(1, -1)
        reason_rho = rho_pred @ ((support_mat + contra_mat).clamp(0, 1)).t() / ((support_mat + contra_mat).clamp(0, 1).sum(-1).clamp_min(1.0)).view(1, -1)
        b = q_pred.shape[0]
        if reason_labels is None:
            pos = torch.zeros(b, 21, device=device)
        else:
            pos = reason_labels.float()
        soft_neg = torch.zeros_like(pos)
        hard_neg = torch.zeros_like(pos)
        if epoch >= self.soft_negative_start_epoch:
            soft_neg = ((1 - pos) * contra_score * reason_rho).clamp(0, 1)
        if epoch >= self.hard_negative_start_epoch:
            hard_neg = ((1 - pos) * (support_score < self.support_threshold).float() * (contra_score > self.contra_threshold).float() * (reason_rho > self.rho_threshold).float())
        unknown = ((1 - pos) * (1 - hard_neg)).clamp(0, 1)
        return {"positive_mask": pos, "soft_negative_weight": soft_neg, "hard_negative_mask": hard_neg, "unknown_mask": unknown, "support_score": support_score, "contra_score": contra_score, "reason_rho": reason_rho, "reason_reliability": reason_rho, "stats": {"positive_count": int(pos.sum().detach().cpu()), "soft_negative_weight_sum": float(soft_neg.sum().detach().cpu()), "hard_negative_count": int(hard_neg.sum().detach().cpu()), "unknown_count": int(unknown.sum().detach().cpu())}}
