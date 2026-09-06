from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fate_oia.utils.coev_contracts import CoEVTargets


def asymmetric_loss(logits: Tensor, target: Tensor, gamma_neg: float = 4.0, clip: float = .05) -> Tensor:
    p = logits.sigmoid(); pos = target * torch.log(p.clamp_min(1e-8))
    neg_p = (1 - p + clip).clamp(max=1.0)
    neg = (1 - target) * torch.log(neg_p.clamp_min(1e-8)) * p.pow(gamma_neg)
    return -(pos + neg).mean()


def binary_gce(logits: Tensor, target: Tensor, q: float = .7) -> Tensor:
    p = logits.sigmoid(); py = target * p + (1 - target) * (1 - p)
    return ((1 - py.clamp_min(1e-8).pow(q)) / q).mean()


def grounding_loss(pred: Tensor, value: Tensor, known: Tensor, weight: Tensor) -> Tensor:
    terms = []
    for c in range(pred.shape[2]):
        mask = known[:, c]
        if not mask.any():
            continue
        pc, yc, wc = pred[:, -1, c][mask], value[:, c][mask], weight[:, c][mask]
        bce = F.binary_cross_entropy(pc, yc, reduction="none") * wc
        pos, neg = yc > .5, yc <= .5
        if pos.any() and neg.any():
            terms.append(.5 * bce[pos].mean() + .5 * bce[neg].mean())
        else:
            terms.append(bce.mean())
    return pred.sum() * 0 if not terms else torch.stack(terms).mean()


class CoEVLoss(nn.Module):
    def forward(self, out: dict[str, Tensor], target: CoEVTargets) -> dict[str, Tensor]:
        with torch.autocast(device_type=out["logits"].device.type, enabled=False):
            a, r = target.action.float(), target.reason.float()
            full_a, full_r = out["action_logits"].float(), out["reason_logits"].float()
            visual = out["visual_logits"].float(); evidence = out["evidence_logits"].float()
            def la(x: Tensor) -> Tensor: return asymmetric_loss(x, a)
            def lr(x: Tensor) -> Tensor: return .8 * asymmetric_loss(x, r) + .2 * binary_gce(x, r)
            evidence_only = out["bias_logits"].float() + evidence
            action = la(full_a) + .25 * la(visual[:, :4]) + .25 * la(evidence_only[:, :4])
            reason = lr(full_r) + .25 * lr(visual[:, 4:]) + .25 * lr(evidence_only[:, 4:])
            ground = grounding_loss(out["predicate_maps_undetached"].float(), target.grounding_value,
                                    target.grounding_known, target.grounding_weight)
            match = out.get("match_loss", full_a.sum() * 0).float()
            total = action + reason + .20 * ground + .05 * match
        return {"total": total, "action_task": action, "reason_task": reason,
                "ground": ground, "match": match}
