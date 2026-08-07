from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def partial_asl(logits: Tensor, target: Tensor, negative_weight: Tensor | None = None,
                gamma_pos: float = 0.0, gamma_neg: float = 4.0) -> Tensor:
    prob = torch.sigmoid(logits)
    pos = -target * (1.0 - prob).pow(gamma_pos) * prob.clamp_min(1e-8).log()
    neg = -(1.0 - target) * prob.pow(gamma_neg) * (1.0 - prob).clamp_min(1e-8).log()
    if negative_weight is not None:
        neg = neg * negative_weight
    return (pos + neg).mean()


def soft_f1(logits: Tensor, target: Tensor, negative_weight: Tensor | None = None) -> Tensor:
    prob = torch.sigmoid(logits)
    tp = (prob * target).sum(0)
    fp_term = prob * (1.0 - target)
    if negative_weight is not None:
        fp_term = fp_term * negative_weight
    fp = fp_term.sum(0)
    fn = ((1.0 - prob) * target).sum(0)
    return 1.0 - ((2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)).mean()


def atom_overlap_ceiling_loss(atom_map: Tensor, contribution: Tensor, ceiling=0.65, threshold=1e-3) -> Tensor:
    norm = F.normalize(atom_map, dim=-1)
    cosine = torch.einsum("bakn,bajn->bakj", norm, norm)
    active = (contribution.abs() > threshold)
    pair = active[..., :, None] & active[..., None, :]
    eye = torch.eye(atom_map.shape[2], device=atom_map.device, dtype=torch.bool)[None, None]
    mask = pair & ~eye
    penalty = F.relu(cosine - ceiling).square()
    return penalty[mask].mean() if mask.any() else atom_map.sum() * 0.0


def ecpo_loss(final_pos: Tensor, final_neg: Tensor, primary_pos: Tensor, primary_neg: Tensor,
              weight: Tensor, beta: float = 2.0) -> Tensor:
    improvement = (final_pos - final_neg) - (primary_pos - primary_neg).detach()
    return -(weight * F.logsigmoid(beta * improvement)).sum() / weight.sum().clamp_min(1.0)


def naming_preference_loss(quality: Tensor, positive_mask: Tensor, margin: float = 0.05) -> Tensor:
    positive = quality.masked_fill(~positive_mask, float("-inf")).amax(-1)
    negative = quality.masked_fill(positive_mask, float("-inf")).amax(-1)
    valid = positive_mask.any(-1) & (~positive_mask).any(-1)
    return F.relu(margin - positive[valid] + negative[valid]).mean() if valid.any() else quality.sum() * 0.0


def evidence_constraints(output: dict[str, Tensor], certificate: dict[str, Tensor] | None,
                         ecpo_gain: Tensor | None, action_target: Tensor) -> tuple[dict[str, Tensor], dict[str, bool]]:
    zero = output["action_delta"].sum() * 0.0
    if certificate is None or not certificate["valid_mask"].any():
        constraints = {"effect": zero, "necessity": zero,
                       "action_budget": output["action_delta"].square().mean() - 0.02}
        availability = {"effect": False, "necessity": False, "action_budget": False, "reason_budget": False}
    else:
        valid = certificate["valid_mask"]
        cert = certificate["certificate"][valid]
        reliability = certificate["reliability"][valid]
        rows = torch.arange(output["bounded_contribution"].shape[0], device=valid.device)
        action_ids = certificate["action_id"]
        target_sign = 2.0 * action_target[rows, action_ids] - 1.0
        contribution = (
            target_sign
            * output["bounded_contribution"][rows, action_ids, certificate["atom_id"]]
        )[valid]
        constraints = {
            "effect": (reliability * F.huber_loss(contribution, cert, reduction="none")).mean() - 0.05,
            "necessity": 0.05 - (reliability * cert).mean(),
            "action_budget": output["action_delta"].square().mean() - 1.25 * F.relu(cert).square().mean() - 0.02,
        }
        availability = {"effect": True, "necessity": True, "action_budget": True, "reason_budget": False}
    if ecpo_gain is not None and ecpo_gain.numel():
        constraints["reason_budget"] = output["reason_delta"].square().mean() - F.relu(ecpo_gain).square().mean() - 0.02
        availability["reason_budget"] = True
    return constraints, availability
