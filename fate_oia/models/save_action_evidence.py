from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


SAVE_ACTION_DIM = 4
SAVE_FACTOR_DIM = 21
SAVE_PATCH_TOKENS = 3600
SAVE_EVIDENCE_RAMP_FRACTION = 0.10
SAVE_EVIDENCE_GAIN_INIT = 0.05


def evidence_ramp(progress: float, fraction: float = SAVE_EVIDENCE_RAMP_FRACTION) -> float:
    """Return the prescribed linear mechanism ramp in the unit interval."""
    fraction = float(fraction)
    if fraction <= 0.0:
        raise ValueError("ramp fraction must be positive")
    return float(min(max(float(progress) / fraction, 0.0), 1.0))


def _normalise_attention(attention: Tensor, *, actions: int, patches: int) -> Tensor:
    if attention.ndim == 2:
        attention = attention.unsqueeze(1).expand(-1, actions, -1)
    if attention.ndim != 3 or attention.shape[-1] != patches:
        raise ValueError(
            f"attention must have shape [B,{actions},{patches}], got {tuple(attention.shape)}"
        )
    if attention.shape[1] == 1:
        attention = attention.expand(-1, actions, -1)
    elif attention.shape[1] >= actions:
        # CalAlign can provide all label rows; SAVE only uses the action rows
        # as a soft prior and never turns the remaining rows into a mask.
        attention = attention[:, :actions]
    else:
        raise ValueError(
            f"attention must have at least {actions} rows, got {tuple(attention.shape)}"
        )
    attention = attention.to(dtype=torch.promote_types(attention.dtype, torch.float32))
    attention = attention.clamp_min(0.0)
    normalizer = attention.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(attention, 1.0 / patches)
    return torch.where(normalizer > 0.0, attention / normalizer.clamp_min(1e-12), uniform)


def build_predicate_soft_prior(
    action_global_attention: Tensor,
    *,
    calalign_action_attention: Tensor | None = None,
    base_attention: Tensor | None = None,
    predicate_map: Tensor | None = None,
    predicate_candidate_weight: Tensor | None = None,
    predicate_reliability: Tensor | None = None,
    predicate_gain: Tensor | None = None,
    unnamed_epsilon: float = 1e-6,
    unnamed_global_weight: float = 1.0,
    unnamed_calalign_weight: float = 1.0,
    predicate_bias_scale: float = 1.0,
) -> dict[str, Tensor]:
    """Build finite named and unnamed soft priors for the detail read.

    ``predicate_candidate_weight`` is an entmax distribution. The final
    candidate may be null, so it has no named-map mass. The unnamed term is
    additive rather than normalized away, which keeps every patch reachable.
    """
    if action_global_attention.ndim != 3:
        raise ValueError("action_global_attention must be [B,A,N]")
    batch, actions, patches = action_global_attention.shape
    global_attention = _normalise_attention(
        action_global_attention, actions=actions, patches=patches
    ).to(action_global_attention)
    if base_attention is not None:
        calalign_action_attention = base_attention
    if calalign_action_attention is None:
        calalign_attention = global_attention
    else:
        calalign_attention = _normalise_attention(
            calalign_action_attention, actions=actions, patches=patches
        ).to(action_global_attention)

    if predicate_map is None:
        factor_dim = SAVE_FACTOR_DIM
        named = action_global_attention.new_zeros(batch, actions, patches)
    else:
        if predicate_map.ndim != 3 or predicate_map.shape[0] != batch:
            raise ValueError("predicate_map must be [B,R,N]")
        factor_dim = int(predicate_map.shape[1])
        if predicate_map.shape[2] != patches:
            raise ValueError("predicate_map patch count must match action attention")
        maps = predicate_map.to(action_global_attention).clamp_min(0.0)
        if predicate_candidate_weight is None:
            weights = maps.new_full((batch, actions, factor_dim), 1.0 / factor_dim)
        else:
            if predicate_candidate_weight.ndim != 3:
                raise ValueError("predicate_candidate_weight must be [B,A,R] or [B,A,R+1]")
            if predicate_candidate_weight.shape[:2] != (batch, actions):
                raise ValueError("predicate candidate batch/action shape mismatch")
            candidate_dim = int(predicate_candidate_weight.shape[-1])
            if candidate_dim not in (factor_dim, factor_dim + 1):
                raise ValueError(
                    "predicate candidate dimension must match predicate_map "
                    "or include one final null candidate"
                )
            candidate_weights = predicate_candidate_weight.to(maps)
            if not torch.isfinite(candidate_weights).all():
                raise ValueError("predicate_candidate_weight must be finite")
            if (candidate_weights < 0).any():
                raise ValueError(
                    "predicate_candidate_weight must be an entmax distribution"
                )
            tolerance = max(
                1e-5,
                2.0 * candidate_dim * torch.finfo(candidate_weights.dtype).eps,
            )
            if not torch.allclose(
                candidate_weights.sum(dim=-1),
                torch.ones_like(candidate_weights[..., 0]),
                atol=tolerance,
                rtol=0.0,
            ):
                raise ValueError(
                    "predicate_candidate_weight must be an entmax distribution"
                )
            weights = candidate_weights[..., :factor_dim]
        if predicate_reliability is None:
            reliability = maps.new_ones(batch, factor_dim)
        else:
            reliability = predicate_reliability.to(maps)
            if reliability.ndim == 1:
                reliability = reliability.view(1, -1).expand(batch, -1)
            if tuple(reliability.shape) != (batch, factor_dim):
                raise ValueError("predicate_reliability must be [B,R] or [R]")
            reliability = reliability.clamp(0.0, 1.0)
        if predicate_gain is None:
            gain = maps.new_ones(batch, actions, factor_dim)
        else:
            gain = predicate_gain.to(maps)
            if gain.ndim == 2:
                gain = gain.unsqueeze(1).expand(-1, actions, -1)
            if tuple(gain.shape) != (batch, actions, factor_dim):
                raise ValueError("predicate_gain must be [B,A,R] or [B,R]")
            gain = gain.clamp_min(0.0)
        named_weight = weights * reliability.unsqueeze(1) * gain
        named = torch.einsum("bar,brn->ban", named_weight, maps)

    epsilon = float(unnamed_epsilon)
    if epsilon <= 0.0:
        raise ValueError("unnamed_epsilon must be positive")
    if float(unnamed_global_weight) < 0.0 or float(unnamed_calalign_weight) < 0.0:
        raise ValueError("unnamed prior weights must be non-negative")
    unnamed = (
        action_global_attention.new_full((batch, actions, patches), epsilon)
        + float(unnamed_global_weight) * global_attention
        + float(unnamed_calalign_weight) * calalign_attention
    )
    prior = named + unnamed
    predicate_bias = float(predicate_bias_scale) * prior.clamp_min(epsilon).log()
    return {
        "predicate_prior_named": named,
        "predicate_prior_unnamed": unnamed,
        "predicate_prior": prior,
        "detail_attention_bias_predicate": predicate_bias,
        "detail_attention_bias_base": (
            0.5 * (global_attention + calalign_attention)
        ).clamp_min(epsilon).log(),
    }


predicate_soft_prior = build_predicate_soft_prior
action_evidence_ramp = evidence_ramp


def _direction_preserving_cap(
    logits: Tensor, base_logits: Tensor, *, ramp: float, cap: float
) -> Tensor:
    if ramp <= 0.0:
        return base_logits
    cap = float(cap)
    if cap <= 0.0:
        raise ValueError("action logit cap must be positive")
    raw_norm = logits.float().norm(dim=-1, keepdim=True)
    scale = (cap / raw_norm.clamp_min(1e-6)).clamp(max=1.0)
    return logits * scale.to(logits.dtype)


class _FoundationFirewalledEvidence(torch.autograd.Function):
    """Reuse the formal forward value while routing auxiliary gradients safely."""

    @staticmethod
    def forward(
        ctx: Any,
        formal_evidence: Tensor,
        action_nodes_base: Tensor,
        global_field: Tensor,
        detail_field: Tensor,
        calalign_action_attention: Tensor | None,
        predicate_map: Tensor | None,
        predicate_candidate_weight: Tensor | None,
        predicate_reliability: Tensor | None,
        predicate_gain: Tensor | None,
        module: "SAVEActionEvidence",
        *parameters: Tensor,
    ) -> Tensor:
        ctx.module = module
        ctx.options = tuple(
            None if value is None else value.detach()
            for value in (
                calalign_action_attention,
                predicate_map,
                predicate_candidate_weight,
                predicate_reliability,
                predicate_gain,
            )
        )
        ctx.save_for_backward(
            action_nodes_base.detach(),
            global_field.detach(),
            detail_field.detach(),
            *parameters,
        )
        return formal_evidence.detach()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        action_nodes_base, global_field, detail_field, *parameters = ctx.saved_tensors
        (
            calalign_action_attention,
            predicate_map,
            predicate_candidate_weight,
            predicate_reliability,
            predicate_gain,
        ) = ctx.options
        with torch.enable_grad():
            global_read = ctx.module._read_global(action_nodes_base, global_field)
            detail_read = ctx.module._read_detail(
                global_read,
                detail_field,
                calalign_action_attention=calalign_action_attention,
                predicate_map=predicate_map,
                predicate_candidate_weight=predicate_candidate_weight,
                predicate_reliability=predicate_reliability,
                predicate_gain=predicate_gain,
            )
            parameter_gradients = torch.autograd.grad(
                detail_read["action_evidence_raw"],
                parameters,
                grad_outputs=grad_output,
                allow_unused=True,
            )
        return (None,) * 10 + parameter_gradients


class SAVEActionEvidence(nn.Module):
    """Global-to-detail signed action evidence over the complete patch field."""

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = SAVE_ACTION_DIM,
        num_heads: int = 8,
        num_patches: int = SAVE_PATCH_TOKENS,
        max_action_delta: float = 1.0,
        rms_momentum: float = 0.95,
        action_logit_cap: float = 20.0,
        unnamed_epsilon: float = 1e-6,
        unnamed_global_weight: float = 1.0,
        unnamed_calalign_weight: float = 1.0,
        predicate_bias_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.num_patches = int(num_patches)
        if self.dim <= 0 or self.action_dim <= 0 or self.num_patches <= 0:
            raise ValueError("dim, action_dim, and num_patches must be positive")
        if self.dim % int(num_heads) != 0:
            raise ValueError("dim must be divisible by num_heads")
        if not 0.0 < float(rms_momentum) < 1.0:
            raise ValueError("rms_momentum must be in (0, 1)")
        if float(max_action_delta) < 0.10:
            raise ValueError("max_action_delta must be at least 0.10")

        self.global_inquiry = nn.MultiheadAttention(
            self.dim,
            int(num_heads),
            dropout=0.0,
            batch_first=True,
        )
        # The first inquiry is the specified residual read: base action nodes
        # remain a direct bypass around the global field.
        self.global_norm = nn.Identity()
        self.detail_query = nn.Linear(self.dim, self.dim, bias=False)
        self.detail_key = nn.Linear(self.dim, self.dim, bias=False)
        self.detail_value = nn.Linear(self.dim, self.dim, bias=False)
        self.detail_output = nn.Linear(self.dim, self.dim, bias=False)
        self.detail_norm = nn.Identity()
        self.patch_action_value = nn.Linear(self.dim, self.dim, bias=False)
        self.patch_value = nn.Linear(self.dim, self.dim, bias=False)
        self.evidence_gain_raw = nn.Parameter(
            torch.full(
                (self.action_dim,),
                math.log(SAVE_EVIDENCE_GAIN_INIT / (1.0 - SAVE_EVIDENCE_GAIN_INIT)),
            )
        )
        self.register_buffer(
            "running_action_rms",
            torch.ones(self.action_dim, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "running_rms_updates",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.rms_momentum = float(rms_momentum)
        self.max_action_delta = float(max_action_delta)
        self.action_logit_cap = float(action_logit_cap)
        self.unnamed_epsilon = float(unnamed_epsilon)
        self.unnamed_global_weight = float(unnamed_global_weight)
        self.unnamed_calalign_weight = float(unnamed_calalign_weight)
        self.predicate_bias_scale = float(predicate_bias_scale)
        if self.unnamed_epsilon <= 0.0:
            raise ValueError("unnamed_epsilon must be positive")
        if self.unnamed_global_weight < 0.0 or self.unnamed_calalign_weight < 0.0:
            raise ValueError("unnamed prior weights must be non-negative")

    def predicate_soft_prior(
        self,
        action_global_attention: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Expose the finite named/unnamed prior as a decoder-level API."""
        return build_predicate_soft_prior(
            action_global_attention,
            unnamed_epsilon=self.unnamed_epsilon,
            unnamed_global_weight=self.unnamed_global_weight,
            unnamed_calalign_weight=self.unnamed_calalign_weight,
            predicate_bias_scale=self.predicate_bias_scale,
            **kwargs,
        )

    def _validate_inputs(
        self,
        action_nodes_base: Tensor,
        global_field: Tensor,
        detail_field: Tensor,
        action_logits_base: Tensor,
    ) -> None:
        if not isinstance(action_nodes_base, Tensor):
            raise TypeError("action_nodes_base must be a tensor")
        if action_nodes_base.ndim != 3:
            raise ValueError("action_nodes_base must be [B,A,D]")
        batch, actions, dim = action_nodes_base.shape
        if (actions, dim) != (self.action_dim, self.dim):
            raise ValueError(
                f"action_nodes_base must be [B,{self.action_dim},{self.dim}]"
            )
        for name, field in (("global_field", global_field), ("detail_field", detail_field)):
            if field.ndim != 3 or tuple(field.shape) != (batch, self.num_patches, self.dim):
                raise ValueError(
                    f"{name} must be [B,{self.num_patches},{self.dim}], got {tuple(field.shape)}"
                )
        if tuple(action_logits_base.shape) != (batch, self.action_dim):
            raise ValueError(
                f"action_logits_base must be [B,{self.action_dim}], got {tuple(action_logits_base.shape)}"
            )

    def _update_rms(self, action_logits_base: Tensor, update_running_stats: bool) -> None:
        if not (self.training and update_running_stats):
            return
        sample_rms = action_logits_base.detach().float().square().mean(0).sqrt()
        with torch.no_grad():
            if int(self.running_rms_updates) == 0:
                self.running_action_rms.copy_(sample_rms.to(self.running_action_rms))
            else:
                self.running_action_rms.mul_(self.rms_momentum).add_(
                    sample_rms.to(self.running_action_rms) * (1.0 - self.rms_momentum)
                )
            self.running_rms_updates.add_(1)

    def _kappa(self, action_logits_base: Tensor) -> Tensor:
        running = self.running_action_rms.float().to(action_logits_base.device)
        return (
            (0.20 * running)
            .clamp(0.10, min(1.00, self.max_action_delta))
            .to(dtype=action_logits_base.dtype)
        )

    def _validate_global_read_inputs(
        self,
        action_nodes_base: Tensor,
        global_field: Tensor,
    ) -> None:
        if not isinstance(action_nodes_base, Tensor):
            raise TypeError("action_nodes_base must be a tensor")
        if action_nodes_base.ndim != 3:
            raise ValueError("action_nodes_base must be [B,A,D]")
        batch, actions, dim = action_nodes_base.shape
        if (actions, dim) != (self.action_dim, self.dim):
            raise ValueError(
                f"action_nodes_base must be [B,{self.action_dim},{self.dim}]"
            )
        if tuple(global_field.shape) != (batch, self.num_patches, self.dim):
            raise ValueError(
                f"global_field must be [B,{self.num_patches},{self.dim}], "
                f"got {tuple(global_field.shape)}"
            )

    def _validate_detail_read_inputs(
        self,
        global_read: Mapping[str, Tensor],
        detail_field: Tensor,
    ) -> None:
        try:
            action_global_token = global_read["action_global_token"]
            global_attention = global_read["action_global_attention"]
            global_bypass = global_read["action_global_bypass"]
        except KeyError as error:
            raise ValueError("global_read is missing a required global output") from error
        if action_global_token.ndim != 3 or tuple(action_global_token.shape[1:]) != (
            self.action_dim,
            self.dim,
        ):
            raise ValueError("global_read action_global_token has an invalid shape")
        batch = int(action_global_token.shape[0])
        if tuple(global_attention.shape) != (batch, self.action_dim, self.num_patches):
            raise ValueError("global_read action_global_attention has an invalid shape")
        if tuple(global_bypass.shape) != (batch, self.action_dim, self.dim):
            raise ValueError("global_read action_global_bypass has an invalid shape")
        if tuple(detail_field.shape) != (batch, self.num_patches, self.dim):
            raise ValueError(
                f"detail_field must be [B,{self.num_patches},{self.dim}], "
                f"got {tuple(detail_field.shape)}"
            )

    def _read_global(
        self,
        action_nodes_base: Tensor,
        global_field: Tensor,
    ) -> dict[str, Tensor]:
        """Run only the global inquiry over the complete global field."""
        global_update, global_attention = self.global_inquiry(
            action_nodes_base,
            global_field,
            global_field,
            need_weights=True,
            average_attn_weights=True,
        )
        return {
            "action_global_token": self.global_norm(action_nodes_base + global_update),
            "action_global_attention": global_attention,
            "action_global_bypass": action_nodes_base,
        }

    def read_global(
        self,
        action_nodes_base: Tensor,
        global_field: Tensor,
    ) -> dict[str, Tensor]:
        """Read the [B,3600,D] global field without invoking detail inquiry."""
        self._validate_global_read_inputs(action_nodes_base, global_field)
        return self._read_global(action_nodes_base, global_field)

    def _read_detail(
        self,
        global_read: Mapping[str, Tensor],
        detail_field: Tensor,
        *,
        calalign_action_attention: Tensor | None,
        predicate_map: Tensor | None,
        predicate_candidate_weight: Tensor | None,
        predicate_reliability: Tensor | None,
        predicate_gain: Tensor | None,
    ) -> dict[str, Tensor]:
        action_global_token = global_read["action_global_token"]
        global_attention = global_read["action_global_attention"]
        prior = build_predicate_soft_prior(
            global_attention,
            calalign_action_attention=calalign_action_attention,
            predicate_map=predicate_map,
            predicate_candidate_weight=predicate_candidate_weight,
            predicate_reliability=predicate_reliability,
            predicate_gain=predicate_gain,
            unnamed_epsilon=self.unnamed_epsilon,
            unnamed_global_weight=self.unnamed_global_weight,
            unnamed_calalign_weight=self.unnamed_calalign_weight,
            predicate_bias_scale=self.predicate_bias_scale,
        )
        base_bias = prior["detail_attention_bias_base"].to(detail_field)
        predicate_bias = prior["detail_attention_bias_predicate"].to(detail_field)
        query = self.detail_query(action_global_token)
        key = self.detail_key(detail_field)
        value = self.detail_value(detail_field)
        detail_scores = (
            torch.einsum("bad,bnd->ban", query, key) / math.sqrt(self.dim)
            + base_bias
            + predicate_bias
        )
        detail_attention = torch.softmax(detail_scores, dim=-1)
        detail_context = torch.einsum("ban,bnd->bad", detail_attention, value)
        action_detail_token = self.detail_norm(
            action_global_token + self.detail_output(detail_context)
        )

        # The signed evidence query is the result of the second inquiry. This
        # keeps the prescribed query/patch dot product while making the detail
        # value/context path causally necessary for the formal residual.
        action_value = self.patch_action_value(action_detail_token)
        patch_value = self.patch_value(detail_field)
        action_patch_value = torch.einsum(
            "bad,bnd->ban", action_value, patch_value
        ) / math.sqrt(self.dim)
        action_patch_contribution = detail_attention * action_patch_value
        action_evidence_raw = action_patch_contribution.sum(dim=-1)
        return {
            **global_read,
            "action_detail_token": action_detail_token,
            "action_detail_attention": detail_attention,
            "action_detail_scores": detail_scores,
            "action_detail_attention_logits": detail_scores,
            "action_patch_value": action_patch_value,
            "action_patch_contribution": action_patch_contribution,
            "action_evidence_raw": action_evidence_raw,
            "action_evidence_sum": action_evidence_raw,
            **prior,
        }

    def read_detail(
        self,
        global_read: Mapping[str, Tensor],
        detail_field: Tensor,
        *,
        calalign_action_attention: Tensor | None = None,
        predicate_map: Tensor | None = None,
        predicate_candidate_weight: Tensor | None = None,
        predicate_reliability: Tensor | None = None,
        predicate_gain: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Read the [B,3600,D] detail field from a completed global inquiry."""
        self._validate_detail_read_inputs(global_read, detail_field)
        return self._read_detail(
            global_read,
            detail_field,
            calalign_action_attention=calalign_action_attention,
            predicate_map=predicate_map,
            predicate_candidate_weight=predicate_candidate_weight,
            predicate_reliability=predicate_reliability,
            predicate_gain=predicate_gain,
        )

    @staticmethod
    def _detach_optional(value: Tensor | None) -> Tensor | None:
        return None if value is None else value.detach()

    def _evidence_parameters(self) -> tuple[Tensor, ...]:
        return (
            self.global_inquiry.in_proj_weight,
            self.global_inquiry.in_proj_bias,
            self.global_inquiry.out_proj.weight,
            self.global_inquiry.out_proj.bias,
            self.detail_query.weight,
            self.detail_key.weight,
            self.detail_value.weight,
            self.detail_output.weight,
            self.patch_action_value.weight,
            self.patch_value.weight,
        )

    def forward(
        self,
        action_nodes_base: Tensor | Mapping[str, Any] | None = None,
        global_field: Tensor | None = None,
        detail_field: Tensor | None = None,
        action_logits_base: Tensor | None = None,
        *,
        progress: float = 1.0,
        calalign_action_attention: Tensor | None = None,
        action_attention_base: Tensor | None = None,
        base_action_attention: Tensor | None = None,
        predicate_map: Tensor | None = None,
        predicate_candidate_weight: Tensor | None = None,
        predicate_reliability: Tensor | None = None,
        predicate_gain: Tensor | None = None,
        update_running_stats: bool = False,
        action_nodes: Tensor | None = None,
        action_logits_calalign: Tensor | None = None,
        action_logits_visual_base: Tensor | None = None,
        calalign_attention: Tensor | None = None,
        label_attention: Tensor | None = None,
        predicate_action_map: Tensor | None = None,
        predicate_action_reliability: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if action_nodes is not None:
            action_nodes_base = action_nodes
        if action_logits_calalign is not None:
            action_logits_base = action_logits_calalign
        if action_logits_visual_base is not None and action_logits_base is None:
            action_logits_base = action_logits_visual_base
        if calalign_attention is not None:
            calalign_action_attention = calalign_attention
        if label_attention is not None and calalign_action_attention is None:
            calalign_action_attention = label_attention
        if predicate_action_map is not None:
            predicate_map = predicate_action_map
        if predicate_action_reliability is not None:
            predicate_reliability = predicate_action_reliability
        if action_nodes_base is None:
            raise ValueError("action_nodes_base or action_nodes is required")
        if isinstance(action_nodes_base, Mapping):
            values = action_nodes_base
            action_nodes_base = values.get("action_nodes_base", values.get("action_nodes"))
            if global_field is None:
                global_field = values["global_field"]
            if detail_field is None:
                detail_field = values["detail_field"]
            if action_logits_base is None:
                action_logits_base = values.get(
                    "action_logits_base", values.get("action_logits_calalign")
                )
            if calalign_action_attention is None:
                calalign_action_attention = values.get(
                    "calalign_action_attention", values.get("label_attention")
                )
            if predicate_map is None:
                predicate_map = values.get("predicate_map")
            if predicate_candidate_weight is None:
                predicate_candidate_weight = values.get("predicate_candidate_weight")
            if predicate_reliability is None:
                predicate_reliability = values.get("predicate_reliability")
        if global_field is None or detail_field is None or action_logits_base is None:
            raise ValueError("global_field, detail_field, and action_logits_base are required")
        self._validate_inputs(action_nodes_base, global_field, detail_field, action_logits_base)
        if action_attention_base is not None:
            calalign_action_attention = action_attention_base
        if base_action_attention is not None:
            calalign_action_attention = base_action_attention

        global_read = self.read_global(action_nodes_base, global_field)
        evidence = self.read_detail(
            global_read,
            detail_field,
            calalign_action_attention=calalign_action_attention,
            predicate_map=predicate_map,
            predicate_candidate_weight=predicate_candidate_weight,
            predicate_reliability=predicate_reliability,
            predicate_gain=predicate_gain,
        )
        self._update_rms(action_logits_base, update_running_stats)
        kappa = self._kappa(action_logits_base)
        action_evidence_bounded = kappa.view(1, -1) * torch.tanh(
            evidence["action_evidence_raw"]
            / kappa.view(1, -1).clamp_min(torch.finfo(action_logits_base.dtype).tiny)
        )
        ramp = evidence_ramp(progress)
        gain = torch.sigmoid(self.evidence_gain_raw).to(action_logits_base).view(1, -1)
        action_evidence_delta = ramp * gain * action_evidence_bounded
        uncapped_final = action_logits_base + action_evidence_delta
        action_logits_final = _direction_preserving_cap(
            uncapped_final,
            action_logits_base,
            ramp=ramp,
            cap=self.action_logit_cap,
        )
        auxiliary_raw = _FoundationFirewalledEvidence.apply(
            evidence["action_evidence_raw"],
            action_nodes_base,
            global_field,
            detail_field,
            calalign_action_attention,
            predicate_map,
            predicate_candidate_weight,
            predicate_reliability,
            predicate_gain,
            self,
            *self._evidence_parameters(),
        )
        auxiliary_bounded = kappa.view(1, -1) * torch.tanh(
            auxiliary_raw
            / kappa.view(1, -1).clamp_min(torch.finfo(action_logits_base.dtype).tiny)
        )
        action_logits_evidence_aux = action_logits_base.detach() + auxiliary_bounded
        output = {
            "action_nodes_base": action_nodes_base,
            "action_logits_base": action_logits_base,
            "action_logits_visual": action_logits_base,
            **evidence,
            "action_evidence_bounded": action_evidence_bounded,
            "action_evidence_delta_unramped": action_evidence_bounded,
            "action_evidence_delta": action_evidence_delta,
            "action_logits_evidence_aux": action_logits_evidence_aux,
            "action_logits_evidence_auxiliary": action_logits_evidence_aux,
            "action_evidence_aux_raw": auxiliary_raw,
            "action_evidence_aux_bounded": auxiliary_bounded,
            "action_logits_final": action_logits_final,
            "action_correction_kappa": kappa.view(1, -1),
            "action_evidence_gain": gain,
            "action_credit_ramp": action_logits_base.new_tensor(ramp),
            "action_logit_uncapped_final": uncapped_final,
        }
        return output

    decode = forward


SAVEActionEvidenceDecoder = SAVEActionEvidence
SaveActionEvidence = SAVEActionEvidence


__all__ = [
    "SAVE_ACTION_DIM",
    "SAVE_FACTOR_DIM",
    "SAVE_PATCH_TOKENS",
    "SAVEActionEvidence",
    "SAVEActionEvidenceDecoder",
    "SaveActionEvidence",
    "action_evidence_ramp",
    "build_predicate_soft_prior",
    "evidence_ramp",
    "predicate_soft_prior",
]
