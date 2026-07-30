from __future__ import annotations

import torch
from torch import Tensor, nn

from .acpr_sparse_ops import entmax15_bisect


class FactorSpecificActionTransport(nn.Module):
    """Exact factor-index-specific additive action evidence transport."""

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        factor_dim: int = 21,
        rank: int = 16,
        rms_momentum: float = 0.95,
        reliability_floor: float = 0.10,
        exploration_mass: float = 0.05,
        correction_fraction: float = 0.20,
        max_visual_rms: float = 5.0,
        max_action_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.factor_dim = int(factor_dim)
        self.rank = int(rank)
        self.rms_momentum = float(rms_momentum)
        self.reliability_floor = float(reliability_floor)
        self.exploration_mass = float(exploration_mass)
        self.correction_fraction = float(correction_fraction)
        self.max_visual_rms = float(max_visual_rms)
        self.max_action_delta = float(max_action_delta)
        if not 0.0 <= self.reliability_floor <= 1.0:
            raise ValueError("reliability_floor must be in [0, 1]")
        if not 0.0 <= self.exploration_mass < 1.0:
            raise ValueError("exploration_mass must be in [0, 1)")
        if self.correction_fraction <= 0.0:
            raise ValueError("correction_fraction must be positive")
        if self.max_visual_rms <= 0.0:
            raise ValueError("max_visual_rms must be positive")
        if self.max_action_delta <= 0.0:
            raise ValueError("max_action_delta must be positive")
        self.action_query = nn.Linear(dim, dim)
        self.factor_key = nn.Parameter(torch.randn(factor_dim, dim, dim) * 0.01)
        self.type_bias = nn.Parameter(torch.zeros(action_dim, factor_dim))
        # This is deliberately action-by-factor, rather than a factor-global gate.
        # It starts neutral and is learned only through the action objective.
        self.action_factor_compatibility = nn.Parameter(
            torch.zeros(action_dim, factor_dim)
        )
        self.null_bias = nn.Parameter(torch.zeros(action_dim))
        self.factor_down = nn.Parameter(torch.randn(factor_dim, rank, dim) * 0.02)
        self.factor_up = nn.Parameter(torch.randn(factor_dim, dim, rank) * 0.02)
        self.register_buffer("running_visual_rms", torch.ones(action_dim), persistent=True)
        self.register_buffer("running_updates", torch.zeros((), dtype=torch.long), persistent=True)

    @staticmethod
    def _ramp(progress: float) -> float:
        return float(min(max(progress / 0.10, 0.0), 1.0))

    def forward(
        self,
        action_logits_visual: Tensor,
        action_nodes: Tensor,
        factor_typed_token: Tensor,
        factor_reliability: Tensor,
        factor_action_ownership: Tensor,
        *,
        factor_value_token: Tensor | None = None,
        factor_source: Tensor | None = None,
        progress: float = 1.0,
        update_running_stats: bool = False,
    ) -> dict[str, Tensor]:
        if factor_source is None:
            factor_source = torch.ones_like(factor_reliability)
        source = factor_source.to(factor_reliability).clamp(0.0, 1.0)
        reliability = factor_reliability.clamp(0.0, 1.0)
        # Observability answers whether evidence exists. Reliability only
        # modulates its strength; it must not make the route structurally
        # unreachable while the state classifier is still high-entropy.
        effective_reliability = self.reliability_floor + (
            1.0 - self.reliability_floor
        ) * reliability
        source_reliability = source * effective_reliability
        if factor_value_token is None:
            factor_value_token = factor_typed_token
        # Transport is an additive action-only reader. Its auxiliary objectives
        # must not rewrite the baseline visual action representation.
        query = self.action_query(action_nodes.detach())
        factor_key = torch.einsum(
            "brd,rde->bre", factor_typed_token, self.factor_key
        )
        score = (
            torch.einsum("bad,brd->bar", query, factor_key)
            + self.type_bias
            + self.action_factor_compatibility
        )
        owner = factor_action_ownership.to(score)
        if owner.ndim == 1:
            owner = owner.unsqueeze(0).expand(self.action_dim, -1)
        if owner.shape != (self.action_dim, self.factor_dim):
            raise ValueError("factor_action_ownership must be [F] or [A,F]")
        owner = owner.unsqueeze(0)
        allowed = owner > 0
        source_available = source.gt(0.05).unsqueeze(1)
        allowed = allowed & source_available
        masked_score = score.masked_fill(~allowed, -1e9)
        full_score = torch.cat(
            [
                masked_score,
                self.null_bias.view(1, -1, 1).expand(score.shape[0], -1, -1),
            ],
            dim=-1,
        )
        dense = torch.softmax(full_score, dim=-1)
        sparse = entmax15_bisect(full_score, dim=-1)
        ramp = self._ramp(progress)
        full_weight = dense * (1.0 - ramp) + sparse * ramp
        dense_factor_weight = dense[..., :-1] * allowed.to(score.dtype)
        factor_weight = full_weight[..., :-1] * allowed.to(score.dtype)
        null_weight = full_weight[..., -1]
        # Preserve a small source-aware route floor after entmax. This keeps
        # eligible alternatives trainable without assigning mass to absent or
        # action-incompatible factors.
        route_prior = owner * source_reliability.unsqueeze(1)
        route_prior = route_prior / route_prior.sum(-1, keepdim=True).clamp_min(1e-8)
        factor_mass = factor_weight.sum(-1, keepdim=True)
        exploration = self.exploration_mass * ramp
        factor_weight = (
            (1.0 - exploration) * factor_weight
            + exploration * factor_mass * route_prior
        )
        low = torch.einsum("brd,rkd->brk", factor_value_token, self.factor_down)
        projected = torch.einsum("brk,rdk->brd", low, self.factor_up)
        factor_value = torch.einsum("bad,brd->bar", query, projected)
        raw_contributions = (
            owner
            * source_reliability.unsqueeze(1)
            * factor_weight
            * factor_value
        )
        visual_rms_raw = action_logits_visual.detach().float().square().mean(0).sqrt()
        if self.training and update_running_stats:
            with torch.no_grad():
                # The reference scale is deliberately bounded before the EMA.
                # Otherwise a diverged visual branch can increase transport
                # authority and form a positive feedback loop.
                current = visual_rms_raw.clamp(max=self.max_visual_rms)
                if int(self.running_updates) == 0:
                    self.running_visual_rms.copy_(current)
                else:
                    self.running_visual_rms.mul_(self.rms_momentum).add_(
                        current * (1.0 - self.rms_momentum)
                    )
                self.running_updates.add_(1)
        reference_rms = self.running_visual_rms.clamp(
            min=1e-4, max=self.max_visual_rms
        )
        kappa = (self.correction_fraction * reference_rms).clamp(
            min=1e-4, max=self.max_action_delta
        ).to(action_logits_visual.dtype)
        # The cap follows the actual sparse evidence selected for this sample,
        # rather than the static number of schema-eligible factors.
        sparse_support = allowed.expand_as(factor_weight) & factor_weight.gt(1e-8)
        active_factor_count = sparse_support.sum(-1).clamp_min(1).to(score.dtype)
        per_factor_kappa = (
            kappa.view(1, -1, 1) / active_factor_count.unsqueeze(-1)
        )
        contributions = ramp * per_factor_kappa * torch.tanh(
            raw_contributions / per_factor_kappa.clamp_min(1e-6)
        )
        evidence_delta = contributions.sum(-1)
        final = action_logits_visual + evidence_delta
        deleted = final.unsqueeze(-1) - contributions
        return {
            "action_logits_visual": action_logits_visual,
            "action_evidence_delta": evidence_delta,
            "action_logits_final": final,
            "action_factor_weights": factor_weight,
            "action_factor_dense_weights": dense_factor_weight,
            "action_null_factor_weight": null_weight,
            "action_factor_values": factor_value,
            "action_factor_contributions": contributions,
            "action_logits_factor_deleted": deleted,
            "action_factor_raw_contributions": raw_contributions,
            "action_factor_effective_reliability": effective_reliability,
            "action_factor_source_mask": allowed.expand_as(factor_weight).to(score.dtype),
            "action_effective_support_count": active_factor_count,
            "action_per_factor_kappa": per_factor_kappa,
            "action_correction_kappa": kappa,
            "action_correction_reference_rms": reference_rms,
            "action_visual_rms_raw": visual_rms_raw,
            "action_correction_rms_ratio": evidence_delta.detach().float().square().mean(0).sqrt()
            / action_logits_visual.detach().float().square().mean(0).sqrt().clamp_min(1e-6),
        }


METERSemanticActionPeer = FactorSpecificActionTransport
