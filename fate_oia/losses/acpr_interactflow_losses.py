from __future__ import annotations

import torch
import torch.nn.functional as F

from fate_oia.acpr_interactflow.nnpu_calalign import nnpu_binary_loss


def action_soft_ce_loss(logits: torch.Tensor, soft_targets: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    loss = -(soft_targets * F.log_softmax(logits, dim=-1)).sum(-1)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def action_soft_kl_loss(logits: torch.Tensor, soft_targets: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    probs_log = F.log_softmax(logits, dim=-1)
    loss = (soft_targets.clamp_min(1e-9) * (soft_targets.clamp_min(1e-9).log() - probs_log)).sum(-1)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def action_majority_ce_loss(logits: torch.Tensor, majority: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.cross_entropy(logits, majority.long(), reduction="none")
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def exp29_masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def exp29_positive_unlabeled_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, positive_prior: float = 0.08) -> torch.Tensor:
    return nnpu_binary_loss(logits, targets.float(), mask.float(), positive_prior=positive_prior)


def predicate_bag_loss(predicate_logits: torch.Tensor, weak_targets: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
    if weak_targets is None:
        return predicate_logits.new_zeros(())
    loss = F.binary_cross_entropy_with_logits(predicate_logits, weak_targets.float(), reduction="none")
    if mask is not None:
        loss = loss * mask.float()
        return loss.sum() / mask.float().sum().clamp_min(1.0)
    return loss.mean()


def flow_sparsity_loss(flow_edges: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(flow_edges, dim=-1)
    entropy = -(probs.clamp_min(1e-9).log() * probs).sum(-1)
    return entropy.mean()


def ledger_identity_loss(identity_error: torch.Tensor) -> torch.Tensor:
    return identity_error.float()


def non_degradation_hinge_loss(final_logits: torch.Tensor, base_logits: torch.Tensor, target: torch.Tensor, margin: float = 0.01) -> torch.Tensor:
    final_ce = F.cross_entropy(final_logits, target.long(), reduction="none")
    base_ce = F.cross_entropy(base_logits, target.long(), reduction="none")
    return F.relu(final_ce - base_ce + margin).mean()


def contribution_alignment_js_loss(contribution_terms: dict[str, torch.Tensor]) -> torch.Tensor:
    components = [v.softmax(-1) for k, v in contribution_terms.items() if k in {"visual", "motion", "predicate", "flow"}]
    if len(components) < 2:
        return next(iter(contribution_terms.values())).new_zeros(())
    mean = torch.stack(components, 0).mean(0).clamp_min(1e-8)
    js = torch.stack([(p.clamp_min(1e-8) * (p.clamp_min(1e-8).log() - mean.log())).sum(-1) for p in components], 0).mean()
    return js


def lag_entropy_loss(lag_weights: torch.Tensor) -> torch.Tensor:
    return (-(lag_weights.clamp_min(1e-9).log() * lag_weights).sum(-1)).mean()


def intervention_margin_loss(selected_drop: torch.Tensor | None = None, random_drop: torch.Tensor | None = None, margin: float = 0.01) -> torch.Tensor:
    if selected_drop is None or random_drop is None:
        return torch.tensor(0.0)
    return F.relu(margin - (selected_drop - random_drop)).mean()


DEFAULT_INTERACTFLOW_LOSS_WEIGHTS = {
    "action_final_soft_kl": 1.00,
    "action_global_soft_kl": 0.50,
    "ledger_residual_soft_kl": 0.20,
    "non_degradation_hinge": 0.10,
    "exp29_masked_asl": 0.30,
    "predicate_nnpu": 0.10,
    "interaction_state_semantic": 0.05,
    "response_lag_consistency": 0.03,
    "contribution_alignment_js": 0.05,
    "temporal_consistency": 0.02,
    "gate_entropy": 0.002,
    "group_sparsity": 0.002,
}


def compute_interactflow_losses(output, batch, weights: dict[str, float] | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    w = {**DEFAULT_INTERACTFLOW_LOSS_WEIGHTS, **(weights or {})}
    terms: dict[str, torch.Tensor] = {}
    terms["action_final_soft_kl"] = action_soft_kl_loss(output.action_logits, batch.action_soft, batch.paper_effective_weight)
    terms["action_global_soft_kl"] = action_soft_kl_loss(output.ledger.global_logits, batch.action_soft, batch.paper_effective_weight)
    residual_logits = output.ledger.flow_delta_logits + output.ledger.calibration_delta
    terms["ledger_residual_soft_kl"] = action_soft_kl_loss(output.action_logits - residual_logits.detach(), batch.action_soft, batch.paper_effective_weight)
    terms["exp29_masked_asl"] = exp29_masked_bce_loss(output.exp29_logits, batch.exp29, batch.exp29_mask)
    terms["predicate_nnpu"] = exp29_positive_unlabeled_loss(output.exp29_logits, batch.exp29, batch.exp29_mask)
    terms["interaction_state_semantic"] = output.action_logits.new_zeros(())
    terms["group_sparsity"] = flow_sparsity_loss(output.flow.flow_edges)
    terms["ledger_identity"] = ledger_identity_loss(output.ledger.identity_error)
    base_logits = output.ledger.visual_logits + 0.35 * output.ledger.motion_logits + 0.25 * output.ledger.predicate_logits
    terms["non_degradation_hinge"] = non_degradation_hinge_loss(output.action_logits, base_logits, batch.action_majority)
    terms["contribution_alignment_js"] = contribution_alignment_js_loss(output.ledger.contribution_terms)
    terms["response_lag_consistency"] = lag_entropy_loss(output.flow.lag_weights)
    terms["temporal_consistency"] = output.action_logits.new_zeros(())
    terms["gate_entropy"] = lag_entropy_loss(output.ledger.gate)
    total = sum(float(w.get(k, 0.0)) * v for k, v in terms.items())
    terms["total_loss"] = total
    return total, terms
