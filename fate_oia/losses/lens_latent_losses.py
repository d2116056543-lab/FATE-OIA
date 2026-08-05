from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def conflict_discounted_responsibility(state_prob: torch.Tensor, emission_prob: torch.Tensor, observed_reason: torch.Tensor, action_state_logits: torch.Tensor, action_targets: torch.Tensor, lambda_action: float) -> dict[str, torch.Tensor]:
    annotation = torch.where(observed_reason.unsqueeze(-1) > 0.5, emission_prob.unsqueeze(0), 1.0 - emission_prob.unsqueeze(0))
    annotation = annotation / annotation.sum(-1, keepdim=True).clamp_min(1e-8)
    visual = state_prob
    kl_vr = (visual.clamp_min(1e-8) * (visual.clamp_min(1e-8).log() - annotation.clamp_min(1e-8).log())).sum(-1)
    kl_rv = (annotation.clamp_min(1e-8) * (annotation.clamp_min(1e-8).log() - visual.clamp_min(1e-8).log())).sum(-1)
    mixture = 0.5 * (visual + annotation)
    js = 0.5 * ((visual * (visual.clamp_min(1e-8).log() - mixture.clamp_min(1e-8).log())).sum(-1) + (annotation * (annotation.clamp_min(1e-8).log() - mixture.clamp_min(1e-8).log())).sum(-1)) / math.log(2.0)
    # action logits are [B,R,3,4]
    loss_by_state = F.binary_cross_entropy_with_logits(action_state_logits, action_targets[:, None, None, :].expand_as(action_state_logits), reduction="none").mean(-1)
    action = torch.softmax(-lambda_action * loss_by_state, dim=-1)
    discounted_annotation = (1 - js).unsqueeze(-1) * annotation + js.unsqueeze(-1) * torch.tensor([0.0, 0.0, 1.0], device=state_prob.device, dtype=state_prob.dtype)
    gamma = visual * discounted_annotation * action
    gamma = (gamma / gamma.sum(-1, keepdim=True).clamp_min(1e-8)).detach()
    return {"gamma": gamma, "conflict": js, "annotation_likelihood": annotation, "action_state_utility": action, "kl_visual_annotation": kl_vr + kl_rv}


def state_loss(state_prob: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return -(gamma * state_prob.clamp_min(1e-8).log()).sum(-1).mean()


def emission_loss(emission_prob: torch.Tensor, observed_reason: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    likelihood = torch.where(observed_reason.unsqueeze(-1) > 0.5, emission_prob.unsqueeze(0), 1 - emission_prob.unsqueeze(0))
    return -(gamma * likelihood.clamp_min(1e-8).log()).sum(-1).mean()


def emission_prior_loss(emission_prob: torch.Tensor) -> torch.Tensor:
    identity = emission_prob.new_tensor([0.0, 0.5, 1.0])
    return F.mse_loss(emission_prob, identity.expand_as(emission_prob))
