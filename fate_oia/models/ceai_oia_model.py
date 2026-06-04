from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fate_oia.models.ceai_action_set import ActionSetPrototypeHead
from fate_oia.models.ceai_cross_expert_exchange import ControlledCrossExpertExchange
from fate_oia.models.ceai_expert_adapter import ActionExpert, ReasonExpert, SharedExpert, TailExpert
from fate_oia.models.ceai_pair_reliability import PairReliabilityHead
from fate_oia.models.ceai_pair_sparse_attention import TaskGuidedPairSparseAttention, default_reason_to_group, group_reason_tokens
from fate_oia.models.ceai_router import ParetoSafeRouter
from fate_oia.models.ceai_scene_state import SceneStatePrototypeTransformer
from fate_oia.models.fate_oia_model import FATEOIAFeatureModel
from fate_oia.utils.ceai_readiness import default_readiness_state


class CEAIOIAFeatureModel(nn.Module):
    """Controlled Explanation-Action Interaction head over DINO/FATE visual tokens."""

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        scene_proto_count: int = 12,
        implicit_proto_count: int = 12,
        pair_topk: int = 24,
        pair_temperature: float = 0.7,
        expert_layers: int = 2,
        expert_heads: int = 4,
        dropout: float = 0.05,
        action_residual_cap: float = 0.04,
        reason_residual_cap: float = 0.12,
        tail_reason_residual_cap: float = 0.18,
        tail_indices: list[int] | None = None,
        a2r_max_scale: float = 0.75,
        r2a_max_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.tail_indices = list(tail_indices or [5, 6, 9, 10, 11, 12, 13, 14])
        self.base_model = FATEOIAFeatureModel(dim=dim, action_dim=action_dim, reason_dim=reason_dim, use_label_query=True)
        self.scene_state = SceneStatePrototypeTransformer(dim, scene_proto_count, implicit_proto_count, num_heads=expert_heads, dropout=dropout)
        self.shared_expert = SharedExpert(dim=dim, depth=1, heads=expert_heads, dropout=dropout)
        self.action_expert = ActionExpert(dim=dim, action_dim=action_dim, depth=expert_layers, heads=expert_heads, dropout=dropout)
        self.reason_expert = ReasonExpert(dim=dim, reason_dim=reason_dim, depth=expert_layers, heads=expert_heads, dropout=dropout)
        self.tail_expert = TailExpert(dim=dim, reason_dim=reason_dim, tail_indices=self.tail_indices, depth=1, heads=expert_heads, dropout=dropout)
        self.action_set = ActionSetPrototypeHead(dim=dim, action_dim=action_dim)
        self.exchange = ControlledCrossExpertExchange(dim=dim, action_dim=action_dim, reason_dim=reason_dim, heads=expert_heads, a2r_max_scale=a2r_max_scale, r2a_max_scale=r2a_max_scale)
        self.pair_attention = TaskGuidedPairSparseAttention(dim=dim, action_dim=action_dim, reason_group_count=6, topk=pair_topk, temperature=pair_temperature, heads=expert_heads)
        self.pair_reliability = PairReliabilityHead(dim=dim, action_dim=action_dim, reason_dim=reason_dim, reason_group_count=6)
        self.router = ParetoSafeRouter(action_dim=action_dim, reason_dim=reason_dim, action_cap=action_residual_cap, reason_cap=reason_residual_cap, tail_reason_cap=tail_reason_residual_cap, tail_indices=self.tail_indices)

    def shared_parameters_for_pcgrad(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in [self.base_model, self.scene_state, self.shared_expert, self.pair_attention, self.pair_reliability]:
            params.extend([p for p in module.parameters() if p.requires_grad])
        return params

    def _readiness(self, pair_attn: dict[str, Any], rel: dict[str, Any], readiness_state: dict[str, Any] | None) -> dict[str, Any]:
        if readiness_state is not None:
            return readiness_state
        state = default_readiness_state()
        concentration = pair_attn["attention_concentration"].detach().mean()
        q_std = rel["pair_reliability"].detach().std(unbiased=False)
        state["pair_attention_concentration"] = float(concentration.cpu())
        state["q_ar_std"] = float(q_std.cpu())
        return state

    def forward(self, tokens: torch.Tensor, bdd100k_scene_state: dict[str, torch.Tensor] | None = None, labels: dict[str, torch.Tensor] | None = None, readiness_state: dict[str, Any] | None = None) -> dict[str, Any]:
        base = self.base_model(tokens)
        label_tokens = base["label_tokens"]
        action_label_tokens = label_tokens[:, : self.action_dim]
        reason_label_tokens = label_tokens[:, self.action_dim :]
        base_action_logits = base["action_fused_logits"]
        base_reason_logits = base["reason_logits"]
        scene = self.scene_state(tokens)
        context = torch.cat([scene["scene_state_tokens"], scene["implicit_prototypes"]], dim=1)
        shared_context = self.shared_expert(context, context)
        action_pre = self.action_expert(action_label_tokens, shared_context)
        action_set = self.action_set(action_pre["tokens"])
        action_context = torch.cat([shared_context, action_pre["tokens"].detach()], dim=1)
        reason_pre = self.reason_expert(reason_label_tokens, action_context)
        reason_group_tokens = group_reason_tokens(reason_pre["tokens"], default_reason_to_group(self.reason_dim), 6)
        pair = self.pair_attention(action_pre["tokens"], reason_group_tokens, scene["scene_state_tokens"], tokens)
        rel = self.pair_reliability(action_pre["tokens"], reason_pre["tokens"], pair["pair_group_context"], base_action_logits, base_reason_logits)
        readiness = self._readiness(pair, rel, readiness_state)
        exchanged = self.exchange(action_pre["tokens"], reason_pre["tokens"], q_ar=rel["pair_reliability"], readiness=readiness)
        action_post = self.action_expert(exchanged["action_tokens"], shared_context)
        reason_post = self.reason_expert(exchanged["reason_tokens"], torch.cat([shared_context, exchanged["action_tokens"].detach()], dim=1))
        tail = self.tail_expert(reason_post["tokens"], shared_context)
        router = self.router(
            base_action_logits,
            base_reason_logits,
            action_post["logits"],
            reason_post["logits"],
            rel["pair_support"],
            rel["pair_reliability"],
            rel["reason_reliability"],
            action_set_logits=action_set["action_set_logits"],
            tail_delta=tail["tail_delta"],
            readiness=readiness,
        )
        diagnostics = {
            "scene_attention_stats": scene["scene_attention_stats"],
            "pair_attention_stats": pair["stats"],
            "pair_reliability_stats": rel["stats"],
            "cross_expert_exchange_stats": exchanged["stats"],
            "router_stats": router["stats"],
            "action_set_stats": action_set["action_set_stats"],
            "readiness": readiness,
        }
        out: dict[str, Any] = {
            **base,
            "base_action_logits": base_action_logits,
            "base_reason_logits": base_reason_logits,
            "action_visual_logits": base["action_visual_logits"],
            "action_reason_logits": base["action_reason_logits"],
            "reason_to_action_logits": base["reason_to_action_logits"],
            "action_fused_logits": base["action_fused_logits"],
            "reason_logits": base_reason_logits,
            "scene_state_logits": scene["scene_state_logits"],
            "scene_state_tokens": scene["scene_state_tokens"],
            "implicit_prototypes": scene["implicit_prototypes"],
            "action_tokens": action_pre["tokens"],
            "reason_tokens": reason_pre["tokens"],
            "action_tokens_exchanged": exchanged["action_tokens"],
            "reason_tokens_exchanged": exchanged["reason_tokens"],
            "action_specialist_logits": action_post["logits"],
            "reason_specialist_logits": reason_post["logits"],
            "action_set_logits": action_set["action_set_logits"],
            "pair_group_context": pair["pair_group_context"],
            "pair_attention_weights": pair["attention_weights"],
            "pair_attention_indices": pair["attention_indices"],
            "pair_attention_entropy": pair["attention_entropy"],
            "pair_attention_concentration": pair["attention_concentration"],
            "pair_support": rel["pair_support"],
            "pair_reliability": rel["pair_reliability"],
            "reason_reliability": rel["reason_reliability"],
            "final_action_logits": router["final_action_logits"],
            "final_reason_logits": router["final_reason_logits"],
            "action_logits": router["final_action_logits"],
            "router_action_gate": router["router_action_gate"],
            "router_reason_gate": router["router_reason_gate"],
            "action_correction": router["action_correction"],
            "reason_correction": router["reason_correction"],
            "diagnostics": diagnostics,
        }
        return out
