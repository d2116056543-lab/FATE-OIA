from __future__ import annotations

import torch
from torch import Tensor, nn


# Visual state tensors use [positive, counter, unknown]. Annotation emission
# tensors use [counter, unknown, positive] so their values stay ordered.
STATE_TO_EMISSION = (1, 2, 0)
EMISSION_TO_STATE = (2, 0, 1)


def state_to_emission_order(state_prob: Tensor) -> Tensor:
    return state_prob[..., list(STATE_TO_EMISSION)]


def emission_to_state_order(value: Tensor) -> Tensor:
    return value[..., list(EMISSION_TO_STATE)]


class LENSAnnotationEmission(nn.Module):
    def __init__(self, reason_dim: int = 21, group_ids: Tensor | None = None, label_delta_scale: float = 0.25) -> None:
        super().__init__()
        if group_ids is None:
            group_ids = torch.tensor([0] * 7 + [1] * 7 + [2] * 7, dtype=torch.long)
        self.register_buffer("group_ids", group_ids.long(), persistent=True)
        self.label_delta_scale = label_delta_scale
        self.group_base = nn.Parameter(torch.zeros(3, 3))
        self.label_delta = nn.Parameter(torch.zeros(reason_dim, 3))

    def emission_probabilities(self) -> Tensor:
        raw = self.group_base[self.group_ids] + self.label_delta_scale * torch.tanh(self.label_delta)
        counter = raw[:, 0]
        unknown = counter + torch.nn.functional.softplus(raw[:, 1])
        positive = unknown + torch.nn.functional.softplus(raw[:, 2])
        return torch.sigmoid(torch.stack([counter, unknown, positive], dim=-1))

    def initialize_from_frequency(self, frequency: Tensor) -> None:
        freq = frequency.detach().float().clamp(1e-4, 1 - 1e-4)
        minus = (0.25 * freq).clamp(0.005, 0.08)
        unknown = torch.maximum(freq, minus + 0.05).clamp(max=0.60)
        positive = (0.90 + 0.08 * (1.0 - freq)).clamp(0.90, 0.995)
        logits = torch.logit(torch.stack([minus, unknown, positive], -1))
        def inverse_softplus(value: Tensor) -> Tensor:
            return torch.log(torch.expm1(value.clamp_min(1e-6)))
        target_raw=torch.stack([logits[:,0],inverse_softplus(logits[:,1]-logits[:,0]),inverse_softplus(logits[:,2]-logits[:,1])],-1)
        with torch.no_grad():
            group_mean = torch.stack([target_raw[self.group_ids == group].mean(0) for group in range(3)])
            self.group_base.copy_(group_mean)
            centered=(target_raw-group_mean[self.group_ids])/self.label_delta_scale
            self.label_delta.copy_(torch.atanh(centered.clamp(-0.999,0.999)))

    def forward(self, state_prob: Tensor, source_reason: Tensor, *, progress: float) -> dict[str, Tensor]:
        alpha = float(max(0.0, min(1.0, progress)))
        learned = self.emission_probabilities()
        identity = learned.new_tensor([0.0, 0.5, 1.0]).expand_as(learned)
        effective = (1.0 - alpha) * identity + alpha * learned
        state_for_emission = state_to_emission_order(state_prob)
        latent_prob = torch.einsum("brs,rs->br", state_for_emission, effective).clamp(1e-6, 1 - 1e-6)
        latent_logits = torch.logit(latent_prob)
        formal = source_reason if alpha == 0.0 else (1.0 - alpha) * source_reason + alpha * latent_logits
        return {
            "emission_prob": effective,
            "emission_prob_learned": learned,
            "state_prob_emission_order": state_for_emission,
            "emission_order_margin": torch.stack([effective[:, 1] - effective[:, 0], effective[:, 2] - effective[:, 1]], dim=-1),
            "emission_order_margin_1": effective[:, 1] - effective[:, 0],
            "emission_order_margin_2": effective[:, 2] - effective[:, 1],
            "reason_prob_latent": latent_prob,
            "reason_logits_latent": latent_logits,
            "reason_logits_formal": formal,
        }
