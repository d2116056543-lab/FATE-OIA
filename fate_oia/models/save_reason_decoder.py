from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SAVE_REASON_RAMP_FRACTION = 0.10
SAVE_CLEAN_LOGIT_CAP_FRACTION = 0.15
SAVE_PRIVATE_KAPPA_FRACTION = 0.35
SAVE_PRIVATE_KAPPA_MIN = 0.20
SAVE_PRIVATE_KAPPA_MAX = 2.00


def _ramp(progress: float, fraction: float = SAVE_REASON_RAMP_FRACTION) -> float:
    if fraction <= 0.0:
        raise ValueError("ramp fraction must be positive")
    return float(min(max(float(progress) / float(fraction), 0.0), 1.0))


def _copy_module_state(target: nn.Module, source: nn.Module) -> None:
    target_state = target.state_dict()
    source_state = source.state_dict()
    compatible = {
        name: value
        for name, value in source_state.items()
        if name in target_state and target_state[name].shape == value.shape
    }
    if compatible:
        target.load_state_dict(compatible, strict=False)


def _validated_module_state(
    target: nn.Module,
    source: object,
    *,
    primitive: str,
) -> dict[str, Tensor]:
    if not isinstance(source, nn.Module):
        raise ValueError(f"foundation is missing {primitive}")
    target_state = target.state_dict()
    source_state = source.state_dict()
    missing = [name for name in target_state if name not in source_state]
    mismatched = [
        name
        for name, target_value in target_state.items()
        if name in source_state and source_state[name].shape != target_value.shape
    ]
    if missing or mismatched:
        detail = ", ".join(missing + mismatched)
        raise ValueError(f"foundation {primitive} is incompatible: {detail}")
    return {name: source_state[name] for name in target_state}


class _ZeroInitLowRankResidual(nn.Module):
    """A zero-effect adapter whose parameters still receive a live gradient."""

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: Tensor) -> Tensor:
        return self.up(F.gelu(self.down(self.norm(value))))


class _IndependentAttention(nn.Module):
    """Compact multi-head attention with physically independent Q/K/V maps."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("dim must be divisible by a positive num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _split(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        bias: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if bias is not None:
            if bias.ndim != 3 or bias.shape[:2] != query.shape[:2] or bias.shape[-1] != key.shape[1]:
                raise ValueError("attention bias must have shape [B,query_length,key_length]")
            scores = scores + bias.unsqueeze(1).to(scores)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(
            query.shape[0], query.shape[1], self.dim
        )
        return self.out_proj(context), attention.mean(dim=1)


def _as_factor_batch(value: Tensor, batch: int, factors: int, *, name: str) -> Tensor:
    if value.ndim == 1 and tuple(value.shape) == (factors,):
        value = value.view(1, factors).expand(batch, -1)
    if tuple(value.shape) != (batch, factors):
        raise ValueError(f"{name} must have shape [B,{factors}] or [{factors}]")
    return value


class SAVECleanReasonRoute(nn.Module):
    """Trainable clean semantic route anchored at the full CalAlign reason logits."""

    def __init__(
        self,
        dim: int = 384,
        reason_dim: int = 21,
        action_dim: int = 4,
        rank: int = 16,
        num_heads: int = 4,
        reason_to_action: nn.Module | None = None,
        startup_gradient_scale: float = 0.10,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.reason_dim = int(reason_dim)
        self.action_dim = int(action_dim)
        self.startup_gradient_scale = float(startup_gradient_scale)
        if self.startup_gradient_scale <= 0.0:
            raise ValueError("startup_gradient_scale must be positive")

        self.predicate_cross_attention = _IndependentAttention(dim, num_heads)
        self.semantic_norm = nn.LayerNorm(dim)
        self.semantic_head = nn.Linear(dim, 1)
        nn.init.xavier_uniform_(self.semantic_head.weight)
        nn.init.zeros_(self.semantic_head.bias)
        self.clean_adapter = _ZeroInitLowRankResidual(dim, int(rank))
        self.condition_projection = nn.Linear(4, dim)
        self.clean_gate_raw = nn.Parameter(torch.zeros(()))
        self.reason_to_action = (
            reason_to_action
            if reason_to_action is not None
            else nn.Linear(reason_dim, action_dim)
        )

    def initialize_from_foundation(self, foundation: nn.Module) -> None:
        trunk = getattr(foundation, "trunk", foundation)
        source = getattr(trunk, "reason_to_action", None)
        if source is None:
            raise ValueError("clean reason initialization requires reason_to_action")
        _copy_module_state(self.reason_to_action, source)

    @staticmethod
    def _optional_factor_input(
        value: Tensor | None,
        *,
        batch: int,
        factors: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if value is None:
            return torch.ones(batch, factors, device=device, dtype=dtype)
        return _as_factor_batch(value.to(device=device, dtype=dtype), batch, factors, name="factor input")

    def forward(
        self,
        *,
        reason_logits_calalign: Tensor,
        reason_nodes: Tensor,
        predicate_token: Tensor | None = None,
        predicate_map: Tensor | None = None,
        predicate_state_prob: Tensor | None = None,
        factor_reliability: Tensor | None = None,
        action_evidence_overlap: Tensor | None = None,
        **_: Tensor,
    ) -> dict[str, Tensor]:
        if reason_logits_calalign.ndim != 2 or reason_logits_calalign.shape[-1] != self.reason_dim:
            raise ValueError("reason_logits_calalign must have shape [B,reason_dim]")
        if reason_nodes.ndim != 3 or tuple(reason_nodes.shape[1:]) != (self.reason_dim, self.dim):
            raise ValueError("reason_nodes must have shape [B,reason_dim,dim]")
        batch = reason_nodes.shape[0]
        if reason_logits_calalign.shape[0] != batch:
            raise ValueError("reason logits and reason nodes must share the batch dimension")

        reliability = self._optional_factor_input(
            factor_reliability,
            batch=batch,
            factors=self.reason_dim,
            device=reason_nodes.device,
            dtype=reason_nodes.dtype,
        ).detach().clamp(0.0, 1.0)
        if action_evidence_overlap is None:
            overlap = torch.zeros_like(reliability)
        else:
            overlap = action_evidence_overlap.to(reason_nodes)
            if overlap.ndim == 3:
                overlap = overlap.mean(dim=1)
            overlap = _as_factor_batch(
                overlap,
                batch,
                self.reason_dim,
                name="action_evidence_overlap",
            ).clamp(0.0, 1.0)

        conditioned = reason_nodes
        predicate_attention = reason_nodes.new_zeros((batch, self.reason_dim, 0))
        if predicate_token is not None:
            if predicate_token.ndim != 3 or predicate_token.shape[0] != batch or predicate_token.shape[-1] != self.dim:
                raise ValueError("predicate_token must have shape [B,P,dim]")
            predicate_delta, predicate_attention = self.predicate_cross_attention(
                reason_nodes,
                predicate_token,
                predicate_token,
            )
            conditioned = conditioned + predicate_delta

        if predicate_state_prob is None:
            state_summary = torch.zeros_like(reliability)
        else:
            state = predicate_state_prob.to(reason_nodes)
            if state.ndim != 3 or state.shape[0] != batch:
                raise ValueError("predicate_state_prob must have shape [B,F,S]")
            state_summary = state.mean(dim=-1)
            if state_summary.shape[1] != self.reason_dim:
                state_summary = state_summary.mean(dim=1, keepdim=True).expand(-1, self.reason_dim)
        if predicate_map is None:
            map_summary = torch.zeros_like(reliability)
        else:
            predicate_maps = predicate_map.to(reason_nodes)
            if predicate_maps.ndim != 3 or predicate_maps.shape[0] != batch:
                raise ValueError("predicate_map must have shape [B,F,N]")
            map_summary = predicate_maps.float().clamp_min(0.0).mean(-1)
            if map_summary.shape[1] != self.reason_dim:
                map_summary = map_summary.mean(1, keepdim=True).expand(-1, self.reason_dim)
            map_summary = map_summary.to(reliability)
        condition = torch.stack((reliability, overlap, state_summary, map_summary), dim=-1)
        conditioned = conditioned + self.condition_projection(condition)
        conditioned = self.semantic_norm(conditioned)

        raw_delta = self.semantic_head(conditioned).squeeze(-1) + self.clean_adapter(conditioned).mean(-1)
        gate = torch.tanh(self.clean_gate_raw)
        # At initialization this is exactly zero, while raw_delta keeps a live
        # startup gradient into the shared reason nodes and adapter parameters.
        zero_effect_delta = gate * raw_delta + (1.0 - gate) * self.startup_gradient_scale * (
            raw_delta - raw_delta.detach()
        )
        base = reason_logits_calalign.to(reason_nodes)
        cap = (
            SAVE_CLEAN_LOGIT_CAP_FRACTION
            * base.detach().float().square().mean(0).sqrt()
        ).to(base)
        clean_delta = cap * torch.tanh(zero_effect_delta / cap.clamp_min(1e-6))
        clean_logits = base + clean_delta
        action_logits = self.reason_to_action(clean_logits)
        return {
            "reason_logits_calalign": base,
            "reason_logits_clean": clean_logits,
            "reason_logits_clean_delta": clean_delta,
            "reason_nodes_clean": conditioned,
            "reason_reliability": reliability,
            "reason_reliability_clean": reliability,
            "action_logits_clean": action_logits,
            "action_reason_logits_clean": action_logits,
            "predicate_attention_clean": predicate_attention,
            "clean_logit_cap": cap,
            "clean_gate": gate.detach().to(clean_logits),
        }


class SAVEPrivateReasonDecoder(nn.Module):
    """High-capacity, action-independent benchmark reason reader."""

    def __init__(
        self,
        dim: int = 384,
        reason_dim: int = 21,
        action_dim: int = 4,
        num_heads: int = 4,
        foundation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.reason_dim = int(reason_dim)
        self.action_dim = int(action_dim)
        self.num_heads = int(num_heads)
        self.reason_queries = nn.Parameter(torch.randn(reason_dim, dim) * 0.02)
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.reason_self_attention = _IndependentAttention(dim, num_heads)
        self.reason_norm = nn.LayerNorm(dim)

        self.global_cross_attention = _IndependentAttention(dim, num_heads)
        self.detail_cross_attention = _IndependentAttention(dim, num_heads)
        self.factor_query = nn.Linear(dim, dim, bias=False)
        self.factor_key = nn.Linear(dim, dim, bias=False)
        self.factor_value = nn.Linear(dim, dim, bias=False)
        self.null_factor = nn.Parameter(torch.zeros(dim))
        self.null_factor_bias = nn.Parameter(torch.zeros(reason_dim))
        self.reread_query = nn.Linear(dim, dim, bias=False)
        self.reread_key = nn.Linear(dim, dim, bias=False)
        self.reread_value = nn.Linear(dim, dim, bias=False)
        self.reread_output = nn.Linear(dim, dim, bias=False)
        self.evidence_bias_raw = nn.Parameter(torch.full((reason_dim,), 0.25))
        self.private_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.register_buffer(
            "clean_logit_rms_ema",
            torch.ones(reason_dim),
            persistent=True,
        )
        self.register_buffer(
            "clean_logit_rms_updates",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.clean_logit_rms_momentum = 0.95
        if foundation is not None:
            self.initialize_from_foundation(foundation)

    def initialize_from_foundation(self, foundation: nn.Module) -> None:
        """Copy current CalAlign reason primitives without sharing parameters."""
        trunk = getattr(foundation, "trunk", foundation)
        label_queries = getattr(trunk, "label_queries", None)
        if not isinstance(label_queries, Tensor):
            raise ValueError("private reason initialization requires CalAlign label_queries")
        source_action_dim = int(getattr(trunk, "action_dim", label_queries.shape[0] - self.reason_dim))
        source_queries = label_queries[source_action_dim : source_action_dim + self.reason_dim]
        if tuple(source_queries.shape) != tuple(self.reason_queries.shape):
            raise ValueError("foundation reason query shape does not match SAVE decoder")

        projection_states: list[tuple[nn.Module, dict[str, Tensor]]] = []
        for target_name, source_name in (
            ("query_projection", "query_proj"),
            ("key_projection", "key_proj"),
            ("value_projection", "value_proj"),
        ):
            source = getattr(trunk, source_name, None)
            target = getattr(self, target_name)
            projection_states.append(
                (
                    target,
                    _validated_module_state(target, source, primitive=source_name),
                )
            )

        source_self_attention = getattr(trunk, "label_self_attn", None)
        if not isinstance(source_self_attention, nn.MultiheadAttention):
            raise ValueError("foundation is missing label_self_attn")
        weight = source_self_attention.in_proj_weight
        bias = source_self_attention.in_proj_bias
        out_weight = source_self_attention.out_proj.weight
        out_bias = source_self_attention.out_proj.bias
        if (
            weight is None
            or tuple(weight.shape) != (3 * self.dim, self.dim)
            or bias is None
            or tuple(bias.shape) != (3 * self.dim,)
            or tuple(out_weight.shape) != (self.dim, self.dim)
            or out_bias is None
            or tuple(out_bias.shape) != (self.dim,)
        ):
            raise ValueError("foundation label_self_attn is incompatible")

        source_reason_norm = getattr(trunk, "reason_norm", None)
        reason_norm_state = _validated_module_state(
            self.reason_norm,
            source_reason_norm,
            primitive="reason_norm",
        )
        source_classifier = getattr(trunk, "logit_head", None)
        classifier_state = _validated_module_state(
            self.classifier,
            source_classifier,
            primitive="logit_head",
        )

        with torch.no_grad():
            self.reason_queries.copy_(source_queries)
            for target, state in projection_states:
                target.load_state_dict(state, strict=True)
            target_attention = self.reason_self_attention
            target_attention.q_proj.weight.copy_(weight[: self.dim])
            target_attention.k_proj.weight.copy_(weight[self.dim : 2 * self.dim])
            target_attention.v_proj.weight.copy_(weight[2 * self.dim :])
            target_attention.q_proj.bias.copy_(bias[: self.dim])
            target_attention.k_proj.bias.copy_(bias[self.dim : 2 * self.dim])
            target_attention.v_proj.bias.copy_(bias[2 * self.dim :])
            target_attention.out_proj.weight.copy_(out_weight)
            target_attention.out_proj.bias.copy_(out_bias)
            self.reason_norm.load_state_dict(reason_norm_state, strict=True)
            self.classifier.load_state_dict(classifier_state, strict=True)

    @staticmethod
    def _resolve(
        value: Tensor | None,
        aliases: Mapping[str, Any],
        names: tuple[str, ...],
    ) -> Tensor | None:
        if value is not None:
            return value
        for name in names:
            candidate = aliases.get(name)
            if isinstance(candidate, Tensor):
                return candidate
        return None

    def _normalize_inputs(
        self,
        *,
        reason_logits_clean: Tensor | None,
        reason_logits_calalign: Tensor | None,
        global_field: Tensor | None,
        detail_field: Tensor | None,
        factor_measurement_token: Tensor | None,
        factor_evidence_map: Tensor | None,
        factor_reliability: Tensor | None,
        aliases: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        clean = self._resolve(
            reason_logits_clean,
            aliases,
            ("clean_logits", "reason_logits_shared", "reason_logits"),
        )
        if clean is None:
            clean = self._resolve(reason_logits_calalign, aliases, ("reason_logits_base",))
        if clean is None or clean.ndim != 2 or clean.shape[-1] != self.reason_dim:
            raise ValueError("reason_logits_clean must have shape [B,reason_dim]")
        global_value = self._resolve(
            global_field,
            aliases,
            ("global_visual_field", "visual_global_field"),
        )
        detail_value = self._resolve(
            detail_field,
            aliases,
            ("detail_visual_field", "visual_detail_field"),
        )
        if global_value is None and detail_value is None:
            raise ValueError("global_field or detail_field is required")
        if global_value is None:
            global_value = detail_value
        if detail_value is None:
            detail_value = global_value
        assert global_value is not None and detail_value is not None
        for name, value in (("global_field", global_value), ("detail_field", detail_value)):
            if value.ndim != 3 or value.shape[0] != clean.shape[0] or value.shape[-1] != self.dim:
                raise ValueError(f"{name} must have shape [B,N,dim]")
        factor_value = self._resolve(
            factor_measurement_token,
            aliases,
            ("predicate_token", "factor_token", "predicate_measurement_token"),
        )
        if factor_value is None:
            factor_value = clean.new_zeros((clean.shape[0], self.reason_dim, self.dim))
        if factor_value.ndim != 3 or tuple(factor_value.shape) != (
            clean.shape[0],
            self.reason_dim,
            self.dim,
        ):
            raise ValueError("factor_measurement_token must have shape [B,reason_dim,dim]")
        map_value = self._resolve(
            factor_evidence_map,
            aliases,
            ("predicate_map", "factor_anchor_map", "evidence_map"),
        )
        if map_value is None:
            map_value = clean.new_zeros((clean.shape[0], self.reason_dim, detail_value.shape[1]))
        if map_value.ndim != 3 or tuple(map_value.shape[:2]) != (clean.shape[0], self.reason_dim):
            raise ValueError("factor_evidence_map must have shape [B,reason_dim,N]")
        if map_value.shape[-1] != detail_value.shape[1]:
            raise ValueError("factor_evidence_map must match detail_field patches")
        if factor_reliability is None:
            reliability = clean.new_ones((clean.shape[0], self.reason_dim))
        else:
            reliability = _as_factor_batch(
                factor_reliability.to(clean),
                clean.shape[0],
                self.reason_dim,
                name="factor_reliability",
            )
        return (
            clean,
            global_value,
            detail_value,
            factor_value,
            map_value,
            reliability,
        )

    def forward(
        self,
        *,
        reason_logits_clean: Tensor | None = None,
        reason_logits_calalign: Tensor | None = None,
        global_field: Tensor | None = None,
        detail_field: Tensor | None = None,
        factor_measurement_token: Tensor | None = None,
        factor_evidence_map: Tensor | None = None,
        factor_reliability: Tensor | None = None,
        progress: float = 1.0,
        update_running_stats: bool = False,
        **aliases: Any,
    ) -> dict[str, Tensor]:
        (
            clean,
            global_value,
            detail_value,
            factor_value,
            map_value,
            reliability_input,
        ) = self._normalize_inputs(
            reason_logits_clean=reason_logits_clean,
            reason_logits_calalign=reason_logits_calalign,
            global_field=global_field,
            detail_field=detail_field,
            factor_measurement_token=factor_measurement_token,
            factor_evidence_map=factor_evidence_map,
            factor_reliability=factor_reliability,
            aliases=aliases,
        )
        # A benchmark/private loss must not rewrite the shared visual core.
        clean_anchor = clean.detach()
        global_read = global_value.detach()
        detail_read = detail_value.detach()
        factor_read = factor_value.detach()
        evidence_read = map_value.detach().float().clamp_min(0.0)
        reliability = reliability_input.detach().clamp(0.0, 1.0)
        batch = clean.shape[0]
        queries = self.reason_queries.unsqueeze(0).expand(batch, -1, -1)
        projected_queries = self.query_projection(queries)
        global_context, global_attention = self.global_cross_attention(
            projected_queries,
            self.key_projection(global_read),
            self.value_projection(global_read),
        )
        detail_context, detail_attention = self.detail_cross_attention(
            projected_queries,
            self.key_projection(detail_read),
            self.value_projection(detail_read),
        )
        visual_queries = projected_queries + global_context + detail_context
        self_context, self_attention = self.reason_self_attention(
            visual_queries,
            visual_queries,
            visual_queries,
        )
        first_read = self.reason_norm(visual_queries + self_context)

        factor_keys = self.factor_key(factor_read)
        null_key = self.factor_key(self.null_factor.view(1, 1, -1)).expand(batch, 1, -1)
        factor_keys = torch.cat((factor_keys, null_key), dim=1)
        factor_values = self.factor_value(factor_read)
        null_value = self.factor_value(self.null_factor.view(1, 1, -1)).expand(batch, 1, -1)
        factor_values = torch.cat((factor_values, null_value), dim=1)
        factor_scores = torch.einsum(
            "brd,bfd->brf",
            self.factor_query(first_read),
            factor_keys,
        ) / (self.dim**0.5)
        factor_scores[..., -1] = factor_scores[..., -1] + self.null_factor_bias.view(1, -1)
        factor_attention = torch.softmax(factor_scores, dim=-1)
        factor_context = torch.einsum("brf,bfd->brd", factor_attention, factor_values)
        composed_map = torch.einsum(
            "brf,bfn->brn", factor_attention[..., : self.reason_dim], evidence_read
        )

        reread_scores = torch.einsum(
            "brd,bnd->brn",
            self.reread_query(first_read),
            self.reread_key(detail_read),
        ) / (self.dim**0.5)
        evidence_bias = F.softplus(self.evidence_bias_raw).view(1, self.reason_dim, 1)
        reread_scores = reread_scores + evidence_bias * torch.log(composed_map.clamp_min(1e-6))
        reread_attention = torch.softmax(reread_scores, dim=-1)
        reread_context = torch.einsum(
            "brn,bnd->brd", reread_attention, self.reread_value(detail_read)
        )
        reread = first_read + self.reread_output(reread_context)
        private_embedding = self.private_norm(reread + factor_context)
        private_delta = self.classifier(private_embedding).squeeze(-1)

        current_rms = clean_anchor.float().square().mean(0).sqrt()
        if self.training and update_running_stats:
            with torch.no_grad():
                if int(self.clean_logit_rms_updates) == 0:
                    self.clean_logit_rms_ema.copy_(current_rms.to(self.clean_logit_rms_ema))
                else:
                    self.clean_logit_rms_ema.mul_(self.clean_logit_rms_momentum).add_(
                        current_rms.to(self.clean_logit_rms_ema)
                        * (1.0 - self.clean_logit_rms_momentum)
                    )
                self.clean_logit_rms_updates.add_(1)
        kappa = (
            SAVE_PRIVATE_KAPPA_FRACTION
            * self.clean_logit_rms_ema.detach().to(private_delta)
        ).clamp(SAVE_PRIVATE_KAPPA_MIN, SAVE_PRIVATE_KAPPA_MAX)
        bounded_delta = kappa.view(1, -1) * torch.tanh(
            private_delta / kappa.view(1, -1).clamp_min(1e-6)
        )
        ramp = _ramp(progress)
        benchmark = clean_anchor + ramp * (1.0 - reliability) * bounded_delta
        private_direct = clean_anchor + bounded_delta
        return {
            "reason_logits_clean": clean_anchor,
            "reason_logits_benchmark": benchmark,
            "reason_logits_bench": benchmark,
            "reason_logits_private_direct": private_direct,
            "reason_logits_private": private_direct,
            "reason_logits_final": benchmark,
            "reason_private_delta": private_delta,
            "reason_private_delta_bounded": bounded_delta,
            "reason_private_kappa": kappa,
            "reason_benchmark_ramp": benchmark.new_tensor(ramp),
            "reason_reliability": reliability,
            "reason_reliability_detached": reliability,
            "reason_embedding_private": private_embedding,
            "private_reason_embedding": private_embedding,
            "reason_global_attention": global_attention,
            "reason_detail_attention": detail_attention,
            "reason_self_attention": self_attention,
            "reason_factor_attention": factor_attention,
            "reason_factor_context": factor_context,
            "reason_composed_evidence_map": composed_map,
            "reason_detail_reread_attention": reread_attention,
            "reason_evidence_bias": evidence_bias.expand(batch, -1, -1),
            "reason_clean_logit_rms_ema": self.clean_logit_rms_ema.detach().to(clean),
        }


class SAVEReasonDecoder(nn.Module):
    """Combined clean/private reason API used by the SAVE model route."""

    def __init__(
        self,
        dim: int = 384,
        reason_dim: int = 21,
        action_dim: int = 4,
        rank: int = 16,
        num_heads: int = 4,
        foundation: nn.Module | None = None,
        reason_to_action: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.clean_reason = SAVECleanReasonRoute(
            dim=dim,
            reason_dim=reason_dim,
            action_dim=action_dim,
            rank=rank,
            num_heads=num_heads,
            reason_to_action=reason_to_action,
        )
        self.private_reason = SAVEPrivateReasonDecoder(
            dim=dim,
            reason_dim=reason_dim,
            action_dim=action_dim,
            num_heads=num_heads,
        )
        if foundation is not None:
            self.initialize_from_foundation(foundation)

    def initialize_from_foundation(self, foundation: nn.Module) -> None:
        self.clean_reason.initialize_from_foundation(foundation)
        self.private_reason.initialize_from_foundation(foundation)

    def forward(
        self,
        *,
        reason_logits_calalign: Tensor,
        reason_nodes: Tensor,
        global_field: Tensor,
        detail_field: Tensor,
        factor_measurement_token: Tensor,
        factor_evidence_map: Tensor,
        factor_reliability: Tensor,
        predicate_token: Tensor | None = None,
        predicate_map: Tensor | None = None,
        predicate_state_prob: Tensor | None = None,
        action_evidence_overlap: Tensor | None = None,
        progress: float = 1.0,
        update_running_stats: bool = False,
    ) -> dict[str, Tensor]:
        clean = self.clean_reason(
            reason_logits_calalign=reason_logits_calalign,
            reason_nodes=reason_nodes,
            predicate_token=predicate_token,
            predicate_map=predicate_map,
            predicate_state_prob=predicate_state_prob,
            factor_reliability=factor_reliability,
            action_evidence_overlap=action_evidence_overlap,
        )
        private = self.private_reason(
            reason_logits_clean=clean["reason_logits_clean"],
            global_field=global_field,
            detail_field=detail_field,
            factor_measurement_token=factor_measurement_token,
            factor_evidence_map=factor_evidence_map,
            factor_reliability=factor_reliability,
            progress=progress,
            update_running_stats=update_running_stats,
        )
        # Keep the clean branch live for its own loss; the private decoder has
        # already consumed detached copies of every shared input.
        return {**private, **clean}


SaveCleanReasonRoute = SAVECleanReasonRoute
SavePrivateReasonDecoder = SAVEPrivateReasonDecoder


__all__ = [
    "SAVE_CLEAN_LOGIT_CAP_FRACTION",
    "SAVE_PRIVATE_KAPPA_FRACTION",
    "SAVE_PRIVATE_KAPPA_MAX",
    "SAVE_PRIVATE_KAPPA_MIN",
    "SAVE_REASON_RAMP_FRACTION",
    "SAVECleanReasonRoute",
    "SAVEPrivateReasonDecoder",
    "SAVEReasonDecoder",
    "SaveCleanReasonRoute",
    "SavePrivateReasonDecoder",
]
