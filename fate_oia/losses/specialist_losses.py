from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fate_oia.losses.asymmetric_loss import AsymmetricLossMultiLabel

TAIL_LABELS = [12, 9, 5, 14, 6, 11, 10, 13]


def reason_asl_loss(logits: torch.Tensor, labels: torch.Tensor, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05) -> torch.Tensor:
    return AsymmetricLossMultiLabel(gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)(logits, labels)


def hard_reason_ranking_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float = 0.5, hard_k: int = 5, tail_label_weight: float = 2.0, tail_labels: list[int] | None = None) -> torch.Tensor:
    losses = []
    tail = set(TAIL_LABELS if tail_labels is None else tail_labels)
    for row_logits, row_labels in zip(logits, labels):
        pos = torch.nonzero(row_labels > 0.5, as_tuple=False).flatten()
        neg = torch.nonzero(row_labels <= 0.5, as_tuple=False).flatten()
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        k = min(int(hard_k), int(neg.numel()))
        hard_neg = neg[torch.topk(row_logits[neg], k=k).indices]
        diff = margin - row_logits[pos].view(-1, 1) + row_logits[hard_neg].view(1, -1)
        label_loss = F.relu(diff)
        weights = torch.tensor([tail_label_weight if int(p.item()) in tail else 1.0 for p in pos], device=logits.device, dtype=logits.dtype).view(-1, 1)
        losses.append((label_loss * weights).mean())
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def sigmoid_f1_loss(logits: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    tp = (probs * labels).sum(0)
    fp = (probs * (1.0 - labels)).sum(0)
    fn = ((1.0 - probs) * labels).sum(0)
    f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    return 1.0 - f1.mean()


def non_tail_distillation_loss(final_reason: torch.Tensor, base_reason: torch.Tensor, tail_labels: list[int] | None = None) -> torch.Tensor:
    tail = set(TAIL_LABELS if tail_labels is None else tail_labels)
    mask = torch.ones(final_reason.shape[1], device=final_reason.device, dtype=torch.bool)
    for idx in tail:
        if 0 <= idx < mask.numel():
            mask[idx] = False
    if not bool(mask.any()):
        return final_reason.new_zeros(())
    return F.mse_loss(final_reason[:, mask], base_reason.detach()[:, mask])


def delta_l2_loss(delta: torch.Tensor) -> torch.Tensor:
    return delta.pow(2).mean()


def action_preserve_loss(final_action: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(final_action, base_action.detach())


def evidence_distill_loss(final_reason: torch.Tensor, evidence_reason: torch.Tensor | None) -> torch.Tensor:
    if evidence_reason is None:
        return final_reason.new_zeros(())
    return F.mse_loss(final_reason, evidence_reason.detach())
