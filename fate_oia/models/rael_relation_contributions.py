"""P9 unary evidence contributions and P10 owner-isolated pairwise relations.

P10 exposes pair deltas and reconstruction interfaces but does not alter any
formal final logit until the later P17 integration stage owns that decision.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PUBLIC_SLOT_COUNT = 20
DEFAULT_GAMMA_CAP = 0.25


def _as_last_dimension_alpha(alpha: float | Tensor, scores: Tensor) -> Tensor:
    """Broadcast alpha to score dimensions before its final class axis."""
    alpha_tensor = torch.as_tensor(alpha, device=scores.device, dtype=scores.dtype)
    if alpha_tensor.ndim == scores.ndim and alpha_tensor.shape[-1] == 1:
        alpha_tensor = alpha_tensor.squeeze(-1)
    try:
        return torch.broadcast_to(alpha_tensor, scores.shape[:-1]).unsqueeze(-1)
    except RuntimeError as error:
        raise ValueError(
            "alpha must broadcast to the input dimensions preceding the final class dimension"
        ) from error


def _solve_entmax_probabilities(
    centered_scores: Tensor,
    alpha: Tensor,
    n_iter: int,
) -> Tensor:
    """Solve alpha-entmax in a numerically stable, row-max centered coordinate system."""
    beta = alpha - 1.0
    lower = -beta.reciprocal()
    upper = torch.zeros_like(lower)
    for _ in range(n_iter):
        tau = (lower + upper) * 0.5
        probabilities = (beta * (centered_scores - tau)).clamp_min(0.0)
        probabilities = probabilities.pow(beta.reciprocal())
        excess_mass = probabilities.sum(dim=-1, keepdim=True) > 1.0
        lower = torch.where(excess_mass, tau, lower)
        upper = torch.where(excess_mass, upper, tau)

    tau = (lower + upper) * 0.5
    probabilities = (beta * (centered_scores - tau)).clamp_min(0.0)
    probabilities = probabilities.pow(beta.reciprocal())
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(centered_scores.dtype).tiny
    )


class _EntmaxBisectFunction(torch.autograd.Function):
    """Forward bisection with an implicit fixed-support alpha-entmax backward pass."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        centered_scores: Tensor,
        alpha: Tensor,
        n_iter: int,
    ) -> Tensor:
        probabilities = _solve_entmax_probabilities(centered_scores, alpha, n_iter)
        ctx.save_for_backward(probabilities, alpha)
        return probabilities

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, None]:
        probabilities, alpha = ctx.saved_tensors
        beta = alpha - 1.0
        active = probabilities > 0.0
        safe_probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
        gppr = torch.where(
            active,
            safe_probabilities.pow(2.0 - alpha),
            torch.zeros_like(probabilities),
        )
        normalizer = gppr.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(probabilities.dtype).tiny
        )

        upstream_mean = (gppr * grad_output).sum(dim=-1, keepdim=True) / normalizer
        score_gradient = gppr * (grad_output - upstream_mean)

        # This is d p / d alpha before the normalization constraint. The
        # following projection enforces sum_i d p_i / d alpha = 0 exactly.
        direct_alpha = torch.where(
            active,
            safe_probabilities
            * (1.0 - beta * safe_probabilities.log())
            / beta.square(),
            torch.zeros_like(probabilities),
        )
        tau_alpha = direct_alpha.sum(dim=-1, keepdim=True) / normalizer
        probability_alpha = direct_alpha - gppr * tau_alpha
        alpha_gradient = (grad_output * probability_alpha).sum(dim=-1, keepdim=True)
        return score_gradient, alpha_gradient, None


def entmax_bisect(
    scores: Tensor,
    *,
    alpha: float | Tensor,
    dim: int = -1,
    n_iter: int = 32,
) -> Tensor:
    """Return alpha-entmax using fp32 solve and an implicit custom backward pass."""
    if scores.ndim < 1:
        raise ValueError("scores must have at least one dimension")
    if dim not in (-1, scores.ndim - 1):
        raise ValueError("entmax_bisect operates only on the last dimension")
    if n_iter < 1:
        raise ValueError("n_iter must be positive")
    if not torch.is_floating_point(scores):
        raise TypeError("scores must be floating point")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")

    output_dtype = scores.dtype
    work_dtype = torch.float32 if scores.dtype in (torch.float16, torch.bfloat16) else scores.dtype
    work_scores = scores.to(dtype=work_dtype)
    work_alpha = _as_last_dimension_alpha(alpha, work_scores)
    if not bool(torch.isfinite(work_alpha).all()):
        raise ValueError("alpha must be finite")
    if not bool((work_alpha > 1.0).all()):
        raise ValueError("alpha must be strictly greater than one")

    centered_scores = work_scores - work_scores.max(dim=-1, keepdim=True).values
    probabilities = _EntmaxBisectFunction.apply(centered_scores, work_alpha, n_iter)
    return probabilities.to(dtype=output_dtype)


class RAELUnaryContribution(nn.Module):
    """Task-to-public-slot unary evidence contribution branch."""

    parameter_owner = "unary_contribution"
    learning_rate = 2e-4

    def __init__(
        self,
        *,
        num_targets: int,
        dim: int = 384,
        attribute_dim: int = 8,
        gamma_cap: float = DEFAULT_GAMMA_CAP,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_targets not in (4, 21):
            raise ValueError("num_targets must be 4 for actions or 21 for reasons")
        if dim < 2 or attribute_dim < 1:
            raise ValueError("dim must be at least two and attribute_dim must be positive")
        if not 0.0 < gamma_cap <= 0.25:
            raise ValueError("gamma_cap must be in (0, 0.25]")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.num_targets = num_targets
        self.dim = dim
        self.attribute_dim = attribute_dim
        self.gamma_cap = float(gamma_cap)
        self.eps = float(eps)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.evidence_proj = nn.Linear(dim, dim, bias=False)
        hidden = max(8, dim // 2)
        self.score_bias = nn.Sequential(
            # Routing may see the target query and slot attributes only. Presence
            # belongs to phi, so it can change contribution content without
            # changing which public slot entmax routes to.
            nn.Linear(dim + attribute_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.phi = nn.Sequential(
            nn.Linear(2 * dim + attribute_dim + 1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.unary_vector = nn.Parameter(torch.empty(num_targets, dim))
        self.null_evidence = nn.Parameter(torch.empty(num_targets, dim))
        self.null_score = nn.Parameter(torch.zeros(num_targets))
        alpha_logit = math.log((1.10 - 1.05) / (1.50 - 1.10))
        self.eta = nn.Parameter(torch.full((num_targets,), alpha_logit))
        self.gamma_unary_raw = nn.Parameter(torch.zeros(num_targets))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.evidence_proj.weight)
        for block in (self.score_bias, self.phi):
            for module in block:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.unary_vector)
        nn.init.normal_(self.null_evidence, mean=0.0, std=0.02)

    def adaptive_alpha(self) -> Tensor:
        return 1.05 + 0.45 * torch.sigmoid(self.eta)

    def bounded_gamma(self) -> Tensor:
        return self.gamma_cap * torch.tanh(self.gamma_unary_raw)

    def owned_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.named_parameters())

    @staticmethod
    def _signed_contribution_diagnostics(
        contribution: Tensor,
        *,
        stage: str,
    ) -> dict[str, Tensor]:
        """Return detached, magnitude-based positive/negative contribution summaries."""
        detached = contribution.detach().float()
        positive = detached.clamp_min(0.0)
        negative = (-detached).clamp_min(0.0)
        diagnostics: dict[str, Tensor] = {}
        for sign, values in (("positive", positive), ("negative", negative)):
            prefix = f"{stage}_{sign}_contribution"
            diagnostics[f"{prefix}_mean"] = values.mean(dim=-1)
            diagnostics[f"{prefix}_rms"] = values.square().mean(dim=-1).sqrt()
            diagnostics[f"{prefix}_mass"] = values.sum(dim=-1)
        return diagnostics

    def _require_inputs(
        self,
        target_tokens: Tensor,
        evidence_tokens: Tensor,
        attributes: Tensor,
        presence: Tensor,
        reliability: Tensor,
    ) -> None:
        if target_tokens.ndim != 3 or target_tokens.shape[1:] != (self.num_targets, self.dim):
            raise ValueError(f"target_tokens must be [B,{self.num_targets},{self.dim}]")
        batch = target_tokens.shape[0]
        if evidence_tokens.shape != (batch, PUBLIC_SLOT_COUNT, self.dim):
            raise ValueError(
                f"evidence_tokens must be [B,20,{self.dim}] for 20 public evidence slots"
            )
        if attributes.shape != (batch, PUBLIC_SLOT_COUNT, self.attribute_dim):
            raise ValueError(f"attributes must be [B,20,{self.attribute_dim}]")
        if presence.shape != (batch, PUBLIC_SLOT_COUNT):
            raise ValueError("presence must be [B,20]")
        if reliability.shape != (batch, PUBLIC_SLOT_COUNT):
            raise ValueError("reliability must be [B,20]")
        for name, value in {
            "target_tokens": target_tokens,
            "evidence_tokens": evidence_tokens,
            "attributes": attributes,
            "presence": presence,
            "reliability": reliability,
        }.items():
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} must be floating point")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        if not bool(((presence >= 0.0) & (presence <= 1.0)).all()):
            raise ValueError("presence must be in [0,1]")

    def forward(
        self,
        *,
        target_tokens: Tensor,
        evidence_tokens: Tensor,
        attributes: Tensor,
        presence: Tensor,
        reliability: Tensor,
    ) -> dict[str, Tensor | Mapping[str, Tensor]]:
        self._require_inputs(target_tokens, evidence_tokens, attributes, presence, reliability)
        dtype = self.query_proj.weight.dtype
        target_tokens = target_tokens.to(dtype=dtype)
        evidence_tokens = evidence_tokens.to(dtype=dtype)
        attributes = attributes.to(dtype=dtype)
        presence = presence.to(dtype=dtype)

        rho = reliability.detach().to(device=target_tokens.device, dtype=dtype).clamp(0.0, 1.0)
        batch = target_tokens.shape[0]
        query = self.query_proj(target_tokens)
        encoded_evidence = self.evidence_proj(evidence_tokens)
        query_expanded = query.unsqueeze(2).expand(-1, -1, PUBLIC_SLOT_COUNT, -1)
        evidence_expanded = encoded_evidence.unsqueeze(1).expand(-1, self.num_targets, -1, -1)
        attribute_expanded = attributes.unsqueeze(1).expand(-1, self.num_targets, -1, -1)
        presence_expanded = presence.unsqueeze(1).unsqueeze(-1).expand(
            -1, self.num_targets, -1, -1
        )

        score_features = torch.cat((query_expanded, attribute_expanded), dim=-1)
        score_bias = self.score_bias(score_features).squeeze(-1)
        dot = (query_expanded * evidence_expanded).sum(dim=-1) / math.sqrt(self.dim)
        score = dot + score_bias + torch.log(rho.unsqueeze(1) + self.eps)
        null_feature = self.null_evidence.view(1, self.num_targets, 1, self.dim).expand(
            batch, -1, 1, -1
        )
        null_score = (
            (query.unsqueeze(2) * null_feature).sum(dim=-1) / math.sqrt(self.dim)
            + self.null_score.view(1, -1, 1)
        )
        all_scores = torch.cat((score, null_score), dim=-1)
        alpha = self.adaptive_alpha().view(1, self.num_targets, 1)
        slot_weights = entmax_bisect(all_scores, alpha=alpha, dim=-1)
        public_weights = slot_weights[..., :PUBLIC_SLOT_COUNT]
        null_mass = slot_weights[..., PUBLIC_SLOT_COUNT]

        phi_features = torch.cat(
            (query_expanded, evidence_expanded, attribute_expanded, presence_expanded), dim=-1
        )
        phi_value = self.phi(phi_features)
        target_value = torch.einsum("bkjd,kd->bkj", phi_value, self.unary_vector)
        unary_raw = public_weights * rho.unsqueeze(1) * target_value
        gamma = self.bounded_gamma()
        unary_postgamma = unary_raw * gamma.view(1, self.num_targets, 1)

        diagnostics = {
            "unary_score_mean": score.detach().mean(dim=-1),
            "slot_weight_entropy": (
                -(slot_weights.detach().float().clamp_min(self.eps)
                  * slot_weights.detach().float().clamp_min(self.eps).log()).sum(dim=-1)
            ),
            "null_mass": null_mass.detach(),
            "alpha": alpha.detach().expand(batch, -1, -1).squeeze(-1),
            "gamma_unary": gamma.detach(),
            "reliability": rho.detach(),
            "unary_raw_rms": unary_raw.detach().float().square().mean(dim=(-1, -2)).sqrt(),
            "unary_postgamma_rms": (
                unary_postgamma.detach().float().square().mean(dim=(-1, -2)).sqrt()
            ),
        }
        diagnostics.update(
            self._signed_contribution_diagnostics(unary_raw, stage="raw")
        )
        diagnostics.update(
            self._signed_contribution_diagnostics(unary_postgamma, stage="postgamma")
        )
        return {
            "unary_contributions_raw": unary_raw,
            "unary_contributions": unary_postgamma,
            "pi": slot_weights,
            "slot_weights": slot_weights,
            "null_mass": null_mass,
            "alpha": alpha.expand(batch, -1, -1).squeeze(-1),
            "gamma_unary": gamma,
            "diagnostics": diagnostics,
        }


class RAELPairwiseContribution(nn.Module):
    """P10 vectorized unordered-pair evidence branch, isolated from formal logits."""

    parameter_owner = "pairwise_contribution"
    learning_rate = 2e-4
    geometry_dim = 10
    hidden_dim = 64

    def __init__(
        self,
        *,
        num_targets: int,
        dim: int = 384,
        gamma_cap: float = DEFAULT_GAMMA_CAP,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_targets not in (4, 21):
            raise ValueError("num_targets must be 4 for actions or 21 for reasons")
        if dim < 2:
            raise ValueError("dim must be at least two")
        if not 0.0 < gamma_cap <= 0.25:
            raise ValueError("gamma_cap must be in (0, 0.25]")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        if self.hidden_dim != 64:
            raise RuntimeError("P10 pairwise hidden dimension must remain exactly 64")
        self.num_targets = num_targets
        self.dim = dim
        self.gamma_cap = float(gamma_cap)
        self.eps = float(eps)
        pair_indices = torch.combinations(torch.arange(PUBLIC_SLOT_COUNT), r=2)
        if pair_indices.shape != (190, 2):
            raise RuntimeError("20 public slots must produce exactly 190 unordered pairs")
        self.register_buffer("pair_indices", pair_indices, persistent=True)
        # D06 names are kept explicit: each is independently audited and owned
        # by this branch rather than borrowed from a global/action/reason path.
        self.Wj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.Wl = nn.Linear(dim, self.hidden_dim, bias=False)
        self.Wr = nn.Linear(self.geometry_dim, self.hidden_dim, bias=False)
        self.Wq = nn.Linear(dim, self.hidden_dim, bias=False)
        # This is the per-target pair output projection w_p,k. Its zero state
        # preserves the formal path and is bootstrapped only by the owner loss.
        self.pair_output = nn.Parameter(torch.zeros(num_targets, self.hidden_dim))
        self.gamma_pair_raw = nn.Parameter(torch.zeros(num_targets))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.Wj,
            self.Wl,
            self.Wr,
            self.Wq,
        ):
            nn.init.xavier_uniform_(module.weight)

    def bounded_gamma(self) -> Tensor:
        return self.gamma_cap * torch.tanh(self.gamma_pair_raw)

    def owned_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.named_parameters())

    def _require_inputs(
        self,
        target_tokens: Tensor,
        evidence_tokens: Tensor,
        slot_masks: Tensor,
        sector_probs: Tensor,
        unary_public_pi: Tensor,
        reliability: Tensor,
    ) -> None:
        if target_tokens.ndim != 3 or target_tokens.shape[1:] != (self.num_targets, self.dim):
            raise ValueError(f"target_tokens must be [B,{self.num_targets},{self.dim}]")
        batch = target_tokens.shape[0]
        if evidence_tokens.shape != (batch, PUBLIC_SLOT_COUNT, self.dim):
            raise ValueError(f"evidence_tokens must be [B,20,{self.dim}]")
        if slot_masks.ndim != 4 or slot_masks.shape[:2] != (batch, PUBLIC_SLOT_COUNT):
            raise ValueError("slot_masks must be [B,20,H,W]")
        if slot_masks.shape[-2] < 1 or slot_masks.shape[-1] < 1:
            raise ValueError("slot_masks need nonempty spatial dimensions")
        if sector_probs.shape != (batch, PUBLIC_SLOT_COUNT, 3):
            raise ValueError("sector_probs must be [B,20,3]")
        if unary_public_pi.shape != (batch, self.num_targets, PUBLIC_SLOT_COUNT):
            raise ValueError(f"unary_public_pi must be [B,{self.num_targets},20]")
        if reliability.shape != (batch, PUBLIC_SLOT_COUNT):
            raise ValueError("reliability must be [B,20]")
        for name, value in {
            "target_tokens": target_tokens,
            "evidence_tokens": evidence_tokens,
            "slot_masks": slot_masks,
            "sector_probs": sector_probs,
            "unary_public_pi": unary_public_pi,
            "reliability": reliability,
        }.items():
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} must be floating point")
            if value.device != target_tokens.device:
                raise ValueError(f"E_P10_DEVICE_{name}")

    def validate_values(
        self,
        *,
        target_tokens: Tensor,
        evidence_tokens: Tensor,
        slot_masks: Tensor,
        sector_probs: Tensor,
        unary_public_pi: Tensor,
        reliability: Tensor,
        tolerance: float = 1e-5,
        max_abs: float = 1e6,
    ) -> None:
        """Explicit audit-only value validation; never call this from formal forward."""
        self._require_inputs(
            target_tokens,
            evidence_tokens,
            slot_masks,
            sector_probs,
            unary_public_pi,
            reliability,
        )
        for name, value in {
            "target_tokens": target_tokens,
            "evidence_tokens": evidence_tokens,
            "slot_masks": slot_masks,
            "sector_probs": sector_probs,
            "unary_public_pi": unary_public_pi,
            "reliability": reliability,
        }.items():
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"E_P10_NONFINITE_{name}")
            if not bool((value.abs() <= max_abs).all()):
                raise ValueError(f"E_P10_MAGNITUDE_{name}")
        if not bool(((unary_public_pi >= 0.0) & (unary_public_pi <= 1.0)).all()):
            raise ValueError("E_P10_PI_RANGE")
        if not bool((unary_public_pi.sum(dim=-1) <= 1.0 + tolerance).all()):
            raise ValueError("E_P10_PI_PUBLIC_MASS")
        if not bool(((reliability >= 0.0) & (reliability <= 1.0)).all()):
            raise ValueError("E_P10_RHO_RANGE")

    def _pair_geometry(self, slot_masks: Tensor, sector_probs: Tensor) -> Tensor:
        """Compute centroid/mass/soft-IoU/sector geometry in fp32 for all 190 pairs."""
        masks = slot_masks.float().clamp(0.0, 1.0)
        batch, _, height, width = masks.shape
        y_coordinates = torch.linspace(-1.0, 1.0, height, device=masks.device).view(
            1, 1, height, 1
        )
        x_coordinates = torch.linspace(-1.0, 1.0, width, device=masks.device).view(
            1, 1, 1, width
        )
        mass = masks.sum(dim=(-2, -1))
        safe_mass = mass.clamp_min(self.eps)
        centroid_x = (masks * x_coordinates).sum(dim=(-2, -1)) / safe_mass
        centroid_y = (masks * y_coordinates).sum(dim=(-2, -1)) / safe_mass
        nonempty = mass > self.eps
        centroid_x = torch.where(nonempty, centroid_x, torch.zeros_like(centroid_x))
        centroid_y = torch.where(nonempty, centroid_y, torch.zeros_like(centroid_y))

        left, right = self.pair_indices.unbind(dim=-1)
        mask_left = masks[:, left]
        mask_right = masks[:, right]
        intersection = (mask_left * mask_right).sum(dim=(-2, -1))
        union = (mask_left + mask_right - mask_left * mask_right).sum(dim=(-2, -1))
        soft_iou = intersection / union.clamp_min(self.eps)
        left_mass = mass[:, left]
        right_mass = mass[:, right]
        geometry = torch.cat(
            (
                (centroid_x[:, left] - centroid_x[:, right]).unsqueeze(-1),
                (centroid_y[:, left] - centroid_y[:, right]).unsqueeze(-1),
                torch.log((left_mass + self.eps) / (right_mass + self.eps)).unsqueeze(-1),
                soft_iou.unsqueeze(-1),
                sector_probs.float()[:, left],
                sector_probs.float()[:, right],
            ),
            dim=-1,
        )
        if geometry.shape != (batch, 190, self.geometry_dim):
            raise RuntimeError("pair geometry must be [B,190,10]")
        return geometry

    @staticmethod
    def _signed_pair_diagnostics(contribution: Tensor, *, stage: str) -> dict[str, Tensor]:
        detached = contribution.detach().float()
        positive = detached.clamp_min(0.0)
        negative = (-detached).clamp_min(0.0)
        prefix = f"pair_{stage}"
        return {
            f"{prefix}_positive_contribution_mean": positive.mean(dim=-1),
            f"{prefix}_positive_contribution_rms": positive.square().mean(dim=-1).sqrt(),
            f"{prefix}_positive_contribution_mass": positive.sum(dim=-1),
            f"{prefix}_negative_contribution_mean": negative.mean(dim=-1),
            f"{prefix}_negative_contribution_rms": negative.square().mean(dim=-1).sqrt(),
            f"{prefix}_negative_contribution_mass": negative.sum(dim=-1),
            f"{prefix}_active_fraction": detached.ne(0.0).float().mean(dim=-1),
            f"{prefix}_active_count": detached.ne(0.0).float().sum(dim=-1),
        }

    def _incident_by_slot(self, pair_values: Tensor) -> Tensor:
        """Accumulate every unordered pair into both of its public evidence slots."""
        if pair_values.ndim != 3 or pair_values.shape[1:] != (self.num_targets, 190):
            raise ValueError(f"pair_values must be [B,{self.num_targets},190]")
        batch = pair_values.shape[0]
        left, right = self.pair_indices.unbind(dim=-1)
        shape = (batch, self.num_targets, 190)
        left_index = left.view(1, 1, 190).expand(shape)
        right_index = right.view(1, 1, 190).expand(shape)
        incident = pair_values.new_zeros(batch, self.num_targets, PUBLIC_SLOT_COUNT)
        incident.scatter_add_(2, left_index, pair_values)
        incident.scatter_add_(2, right_index, pair_values)
        return incident

    def delete_slot_from_pair_sum(
        self,
        pair_sum: Tensor,
        incident_by_slot: Tensor,
        slot_index: int,
    ) -> Tensor:
        """Analytically remove all pairs touching one public slot without rerunning the branch."""
        if not 0 <= slot_index < PUBLIC_SLOT_COUNT:
            raise ValueError("slot_index must be in [0, 19]")
        if pair_sum.ndim != 2 or pair_sum.shape[1] != self.num_targets:
            raise ValueError(f"pair_sum must be [B,{self.num_targets}]")
        if incident_by_slot.shape != (*pair_sum.shape, PUBLIC_SLOT_COUNT):
            raise ValueError(f"incident_by_slot must be [B,{self.num_targets},20]")
        return pair_sum - incident_by_slot[..., slot_index]

    def reconstruct_with_pair(self, global_logits: Tensor, pair_delta: Tensor) -> Tensor:
        """P17-safe fp32 global-plus-pair reconstruction with no hidden final-path side effect."""
        if global_logits.ndim != 2 or global_logits.shape[1] != self.num_targets:
            raise ValueError(f"global_logits must be [B,{self.num_targets}]")
        if pair_delta.shape != global_logits.shape:
            raise ValueError("pair_delta must match global_logits")
        if global_logits.device != pair_delta.device:
            raise ValueError("E_P10_RECONSTRUCTION_DEVICE")
        return global_logits.float() + pair_delta.float()

    def owner_isolated_auxiliary(
        self,
        *,
        global_logits: Tensor,
        target_tokens: Tensor,
        evidence_tokens: Tensor,
        slot_masks: Tensor,
        sector_probs: Tensor,
        unary_public_pi: Tensor,
        reliability: Tensor,
    ) -> dict[str, Tensor | Mapping[str, Tensor]]:
        """Recompute owner-only pair logits for real ASL/EvidenceConditional training.

        Every non-owner source is stopped before this recomputation.  The caller
        applies its real ASL and EvidenceConditional terms to the returned
        logits/delta; this module deliberately does not invent a synthetic target.
        """
        batch = target_tokens.shape[0]
        if global_logits.shape != (batch, self.num_targets):
            raise ValueError(f"global_logits must be [B,{self.num_targets}]")
        if global_logits.device != target_tokens.device:
            raise ValueError("E_P10_DEVICE_global_logits")

        # The owner bootstrap sees only its true external inputs.  Every one is
        # stopped before recomputation so its ASL/EvidenceConditional objective
        # can train only this pairwise owner.
        detached_inputs = {
            "global_logits": global_logits.detach(),
            "target_tokens": target_tokens.detach(),
            "evidence_tokens": evidence_tokens.detach(),
            "slot_masks": slot_masks.detach(),
            "sector_probs": sector_probs.detach(),
            "unary_public_pi": unary_public_pi.detach(),
            "reliability": reliability.detach(),
        }
        isolated = self(
            target_tokens=detached_inputs["target_tokens"],
            evidence_tokens=detached_inputs["evidence_tokens"],
            slot_masks=detached_inputs["slot_masks"],
            sector_probs=detached_inputs["sector_probs"],
            unary_public_pi=detached_inputs["unary_public_pi"],
            reliability=detached_inputs["reliability"],
        )
        pair_sum = isolated["pair_raw_sum"]
        if not isinstance(pair_sum, Tensor):
            raise RuntimeError("owner isolated pair sum must be a tensor")
        auxiliary_logits = self.reconstruct_with_pair(
            detached_inputs["global_logits"],
            pair_sum,
        )
        output: dict[str, Tensor | Mapping[str, Tensor]] = {
            "owner_pair_auxiliary_delta": pair_sum,
            "owner_auxiliary_logits": auxiliary_logits,
            "pair_indices": isolated["pair_indices"],
            "pair_geometry": isolated["pair_geometry"],
            "diagnostics": isolated["diagnostics"],
        }
        if self.num_targets == 4:
            output["action_pair_auxiliary_delta"] = pair_sum
            output["action_auxiliary_logits"] = auxiliary_logits
        else:
            output["reason_pair_auxiliary_delta"] = pair_sum
            output["reason_auxiliary_logits"] = auxiliary_logits
        return output

    def forward(
        self,
        *,
        target_tokens: Tensor,
        evidence_tokens: Tensor,
        slot_masks: Tensor,
        sector_probs: Tensor,
        unary_public_pi: Tensor,
        reliability: Tensor,
    ) -> dict[str, Tensor | Mapping[str, Tensor]]:
        self._require_inputs(
            target_tokens,
            evidence_tokens,
            slot_masks,
            sector_probs,
            unary_public_pi,
            reliability,
        )
        dtype = self.Wq.weight.dtype
        targets = target_tokens.to(dtype=dtype)
        evidence = evidence_tokens.to(dtype=dtype)
        geometry = self._pair_geometry(slot_masks, sector_probs)
        left, right = self.pair_indices.unbind(dim=-1)
        left_evidence = evidence[:, left]
        right_evidence = evidence[:, right]
        hidden = F.gelu(
            self.Wj(left_evidence).unsqueeze(1)
            + self.Wl(right_evidence).unsqueeze(1)
            + self.Wr(geometry.to(dtype=dtype)).unsqueeze(1)
            + self.Wq(targets).unsqueeze(2)
        )
        pair_value = torch.einsum("bkph,kh->bkp", hidden, self.pair_output)
        # Formal pair supervision must refine P9 routing.  Reliability remains a
        # ledger quantity, so only rho is stop-gradient by contract.
        pi = unary_public_pi.to(device=targets.device, dtype=dtype)
        rho = reliability.detach().to(device=targets.device, dtype=dtype).clamp(0.0, 1.0)
        pi_left = pi[..., left]
        pi_right = pi[..., right]
        rho_left = rho[:, left].unsqueeze(1)
        rho_right = rho[:, right].unsqueeze(1)
        pair_weight = pi_left * pi_right * rho_left * rho_right
        pair_raw = pair_weight * pair_value
        gamma = self.bounded_gamma()
        pair_postgamma = pair_raw * gamma.view(1, self.num_targets, 1)
        pair_raw_sum = pair_raw.sum(dim=-1)
        pair_postgamma_sum = pair_postgamma.sum(dim=-1)
        incident_raw = self._incident_by_slot(pair_raw)
        incident_postgamma = self._incident_by_slot(pair_postgamma)
        diagnostics = {
            "pair_weight_mean": pair_weight.detach().float().mean(dim=-1),
            "pair_geometry_rms": geometry.detach().float().square().mean(dim=(-1, -2)).sqrt(),
            "gamma_pair": gamma.detach(),
        }
        diagnostics.update(self._signed_pair_diagnostics(pair_raw, stage="raw"))
        diagnostics.update(self._signed_pair_diagnostics(pair_postgamma, stage="postgamma"))
        return {
            "pair_contributions_raw": pair_raw,
            "pair_contributions": pair_postgamma,
            "pair_raw_sum": pair_raw_sum,
            "pair_postgamma_sum": pair_postgamma_sum,
            "incident_raw_by_slot": incident_raw,
            "incident_postgamma_by_slot": incident_postgamma,
            "pair_indices": self.pair_indices,
            "pair_geometry": geometry,
            "geometry": geometry,
            "gamma_pair": gamma,
            "diagnostics": diagnostics,
        }
