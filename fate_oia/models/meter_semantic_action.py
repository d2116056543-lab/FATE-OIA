from __future__ import annotations

import torch
from torch import Tensor, nn

from .acpr_sparse_ops import entmax15_bisect


def heca_credit_ramp(progress: float) -> float:
    """Warm up for 5% of updates, then ramp credit through update 20%."""
    return float(min(max((float(progress) - 0.05) / 0.15, 0.0), 1.0))


class StateConditionedActionCredit(nn.Module):
    """All-action soft allocation with state-conditioned signed factor values."""

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        factor_dim: int = 21,
        rank: int = 16,
        max_states: int = 3,
        correction_fraction: float = 0.20,
        max_action_delta: float = 1.0,
        rms_momentum: float = 0.95,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.factor_dim = int(factor_dim)
        self.rank = int(rank)
        self.max_states = int(max_states)
        self.register_buffer(
            "correction_fraction",
            torch.tensor(float(correction_fraction)),
            persistent=True,
        )
        self.max_action_delta = float(max_action_delta)
        self.rms_momentum = float(rms_momentum)
        self.action_query = nn.Linear(dim, dim, bias=False)
        self.factor_key = nn.Linear(dim, dim, bias=False)
        self.learned_action_factor_bias = nn.Parameter(
            torch.zeros(action_dim, factor_dim)
        )
        # V_r keeps each factor's state value semantically distinct, while
        # U_a turns the current sample's detached action token into a reader.
        # The state effect starts at zero, preserving the visual action anchor
        # while still giving the effect tensor a first-step gradient.
        self.factor_value_weight = nn.Parameter(torch.empty(factor_dim, rank, dim))
        self.factor_value_bias = nn.Parameter(torch.zeros(factor_dim, rank))
        self.action_value_weight = nn.Parameter(torch.empty(action_dim, rank, dim))
        self.action_value_bias = nn.Parameter(torch.zeros(action_dim, rank))
        self.state_effect_embedding = nn.Parameter(
            torch.zeros(factor_dim, max_states, rank)
        )
        nn.init.xavier_uniform_(self.factor_value_weight)
        nn.init.xavier_uniform_(self.action_value_weight)
        self.register_buffer(
            "running_visual_rms", torch.ones(action_dim), persistent=True
        )
        self.register_buffer(
            "running_updates", torch.zeros((), dtype=torch.long), persistent=True
        )

    def _ownership(self, value: Tensor, reference: Tensor) -> Tensor:
        owner = value.to(reference).reshape(-1)
        if owner.shape != (self.factor_dim,):
            raise ValueError("HECA factor_action_ownership must be scalar [F]")
        return owner.clamp(0.0, 1.0)

    def forward(
        self,
        action_logits_visual: Tensor,
        action_nodes: Tensor,
        factor_action_bridge_token: Tensor,
        factor_state_prob_credit: Tensor,
        factor_reliability: Tensor,
        factor_action_ownership: Tensor,
        *,
        progress: float = 1.0,
        update_running_stats: bool = False,
        diagnostic_schema_target: int | None = None,
    ) -> dict[str, Tensor]:
        if factor_action_bridge_token.shape[1:] != (
            self.factor_dim,
            action_nodes.shape[-1],
        ):
            raise ValueError("Expected HECA factor bridge [B,F,D]")
        if factor_state_prob_credit.shape[1:] != (
            self.factor_dim,
            self.max_states,
        ):
            raise ValueError("Expected HECA state probabilities [B,F,S]")
        owner = self._ownership(factor_action_ownership, action_logits_visual)
        query = self.action_query(action_nodes.detach())
        key = self.factor_key(factor_action_bridge_token)
        allocation_score = (
            torch.einsum("bad,bfd->baf", query, key)
            / query.shape[-1] ** 0.5
            + self.learned_action_factor_bias
        )
        if diagnostic_schema_target is not None:
            target = int(diagnostic_schema_target)
            if not 0 <= target < self.action_dim:
                raise ValueError("HECA schema diagnostic target is out of range")
            allocation_score = allocation_score.clone()
            allocation_score[:, target] = torch.roll(
                allocation_score[:, target], shifts=1, dims=-1
            )
        # Only latent/non-owned factors are excluded. Every other factor can
        # learn a signed effect for every action.
        allocation_score = allocation_score.masked_fill(
            owner.view(1, 1, -1).eq(0), -1e9
        )
        factor_weight = entmax15_bisect(allocation_score, dim=-1)

        factor_value_embedding = torch.einsum(
            "bfd,frd->bfr", factor_action_bridge_token, self.factor_value_weight
        ) + self.factor_value_bias.unsqueeze(0)
        action_value_query = torch.einsum(
            "bad,ard->bar", action_nodes.detach(), self.action_value_weight
        ) + self.action_value_bias.unsqueeze(0)
        state_modulated = (
            factor_value_embedding.unsqueeze(2)
            * self.state_effect_embedding.unsqueeze(0)
        )
        raw_state_values = torch.einsum(
            "bar,bfsr->bafs", action_value_query, state_modulated
        )
        weighted_state_values = (
            raw_state_values * factor_state_prob_credit.unsqueeze(1)
        )
        factor_value = weighted_state_values.sum(-1)
        contribution = (
            owner.view(1, 1, -1)
            * factor_reliability.detach().clamp(0.0, 1.0).unsqueeze(1)
            * factor_weight
            * factor_value
        )
        credit_sum = contribution.sum(-1)

        visual_rms = (
            action_logits_visual.detach().float().square().mean(0).sqrt()
        )
        if self.training and update_running_stats:
            with torch.no_grad():
                if int(self.running_updates) == 0:
                    self.running_visual_rms.copy_(visual_rms)
                else:
                    self.running_visual_rms.mul_(self.rms_momentum).add_(
                        visual_rms * (1.0 - self.rms_momentum)
                    )
                self.running_updates.add_(1)
        kappa = (
            self.correction_fraction.to(self.running_visual_rms)
            * self.running_visual_rms
        ).clamp(0.10, min(1.0, self.max_action_delta)).to(action_logits_visual)
        unramped_delta = kappa * torch.tanh(
            credit_sum / kappa.clamp_min(1e-6)
        )
        ramp = heca_credit_ramp(progress)
        evidence_delta = ramp * unramped_delta
        final = action_logits_visual + evidence_delta
        return {
            "action_logits_visual": action_logits_visual,
            "action_logits_final": final,
            "action_factor_state_values": weighted_state_values,
            "action_factor_values": factor_value,
            "action_factor_weights": factor_weight,
            "action_factor_contribution": contribution,
            "action_factor_contributions": contribution,
            "action_credit_sum": credit_sum,
            "action_evidence_delta_unramped": unramped_delta,
            "action_evidence_delta": evidence_delta,
            "action_correction_kappa": kappa.view(1, -1),
            "action_credit_ramp": action_logits_visual.new_tensor(ramp),
            "action_correction_rms_ratio": evidence_delta.detach().float().square().mean(0).sqrt()
            / visual_rms.clamp_min(1e-6),
        }
