from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits


def pairwise_rank_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    rows = []
    for b in range(logits.shape[0]):
        pos = logits[b][labels[b] > 0.5]
        neg = logits[b][labels[b] <= 0.5]
        if pos.numel() and neg.numel():
            rows.append(F.relu(margin - (pos[:, None] - neg[None, :])).mean())
    return torch.stack(rows).mean() if rows else logits.new_zeros(())


def prototype_diversity_loss(prototypes: torch.Tensor) -> torch.Tensor:
    p = F.normalize(prototypes, dim=-1)
    sim = torch.einsum("rkd,rld->rkl", p, p)
    eye = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool)
    return sim.masked_select(~eye[None]).pow(2).mean()


def compute_trace_loss(args: Any, out: dict[str, Any], labels: torch.Tensor, epoch: int) -> tuple[torch.Tensor, dict[str, float]]:
    ad = int(getattr(args, "action_dim", 4))
    reason_gt = labels[:, ad:]
    logits = torch.cat([out["action_logits"], out["reason_logits"]], dim=1)
    base_asl = asymmetric_loss_with_logits(logits, labels, gamma_pos=float(getattr(args, "asl_gamma_pos", 0.0)), gamma_neg=float(getattr(args, "asl_gamma_neg", 4.0)), clip=float(getattr(args, "asl_clip", 0.05)))
    evidence_asl = asymmetric_loss_with_logits(out["transport"]["evidence_reason_logits"], reason_gt, gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
    evidence_rank = pairwise_rank_loss(out["transport"]["evidence_reason_logits"], reason_gt)
    distill = F.mse_loss(torch.sigmoid(out["transport"]["evidence_reason_logits"]), torch.sigmoid(out["base_reason_logits"].detach()))
    action_preserve = F.mse_loss(out["action_logits"], out["base_action_logits"].detach())
    proto_div = prototype_diversity_loss(out["transport_module"].prototypes) if "transport_module" in out else out["reason_logits"].new_zeros(())
    entropy = out["transport"]["transport_entropy"].mean()
    cf_loss = out["reason_logits"].new_zeros(())
    if out.get("cf") and epoch >= int(getattr(args, "counterfactual_start_epoch", 5)):
        valid = out["cf"].get("cf_valid_mask")
        if torch.is_tensor(valid) and valid.any():
            cf_loss = F.relu(float(getattr(args, "cf_margin", 0.05)) - (out["cf"]["reason_logits_factual"] - out["cf"]["reason_logits_target_deleted"])).masked_select(valid).mean()
    loss = base_asl + 0.22 * evidence_asl + 0.045 * evidence_rank + 0.015 * distill + 0.04 * action_preserve + 0.004 * proto_div + 0.002 * entropy + float(getattr(args, "loss_counterfactual_direct", 0.025)) * cf_loss
    return loss, {"base_asl_loss": float(base_asl.detach().cpu()), "evidence_reason_asl_loss": float(evidence_asl.detach().cpu()), "evidence_reason_rank_loss": float(evidence_rank.detach().cpu()), "evidence_base_distill_loss": float(distill.detach().cpu()), "action_preserve_loss": float(action_preserve.detach().cpu()), "prototype_diversity_loss": float(proto_div.detach().cpu()), "transport_entropy_loss": float(entropy.detach().cpu()), "counterfactual_direct_loss": float(cf_loss.detach().cpu()), "total_loss": float(loss.detach().cpu())}
