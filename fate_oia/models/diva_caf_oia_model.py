from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fate_oia.models.fate_oia_model import FATEOIAFeatureModel
from fate_oia.models.diva_multilayer_dino import TinyPatchDINOExtractor
from fate_oia.models.diva_dense_adapter import ActionSpecificLayerMixer, DrivingDenseAdapter
from fate_oia.models.diva_visual_actor import DIVAVisualActor
from fate_oia.models.diva_visual_mixture_gate import SupervisedVisualMixtureGate
from fate_oia.models.caf_factor_bank import CAFFactorBank
from fate_oia.models.caf_bilevel_routing import BiLevelFactorRouter
from fate_oia.models.caf_factor_auditor import CriticalFactorAuditor
from fate_oia.models.caf_reason_decoder import MaskedReasonFromFactorTransformer
from fate_oia.models.caf_reason_reliability import ReasonReliabilityGate


class DIVACAFOIAModel(nn.Module):
    """DIVA-CAF wrapper that preserves the FATE-OIA base actor."""

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        dino_extractor: nn.Module | None = None,
        layer_indices: tuple[int, ...] = (3, 6, 9, 12),
        delta_cap: float = 0.08,
        reason_cap: float = 0.25,
        factor_topk: int = 3,
        use_factor_action_residual: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.layer_indices = tuple(layer_indices)
        self.use_factor_action_residual = bool(use_factor_action_residual)
        self.dino_extractor = dino_extractor if dino_extractor is not None else TinyPatchDINOExtractor(dim=dim, layer_indices=layer_indices)
        self.base_actor = FATEOIAFeatureModel(dim=dim, action_dim=action_dim, reason_dim=reason_dim, use_label_query=True)
        self.layer_mixer = ActionSpecificLayerMixer(dim=dim, action_dim=action_dim, layer_indices=layer_indices)
        self.dense_adapter = DrivingDenseAdapter(dim=dim, action_dim=action_dim)
        self.visual_actor = DIVAVisualActor(dim=dim, action_dim=action_dim)
        self.mixture_gate = SupervisedVisualMixtureGate(action_dim=action_dim, delta_cap=delta_cap)
        self.factor_bank = CAFFactorBank(dim=dim, action_dim=action_dim, factors_per_action=3)
        self.factor_router = BiLevelFactorRouter(dim=dim, action_dim=action_dim, factor_topk=factor_topk, group_topk=2)
        self.factor_auditor = CriticalFactorAuditor()
        self.reason_decoder = MaskedReasonFromFactorTransformer(dim=dim, action_dim=action_dim, reason_dim=reason_dim, reason_cap=reason_cap)
        self.reason_reliability = ReasonReliabilityGate(reason_dim=reason_dim)

    def _gather_selected(self, factors: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        b, n, d = factors.shape
        _, a, k = indices.shape
        expanded = factors.unsqueeze(1).expand(b, a, n, d)
        gather_idx = indices.clamp_min(0).clamp_max(max(n - 1, 0)).unsqueeze(-1).expand(b, a, k, d)
        return torch.gather(expanded, 2, gather_idx)

    def forward(
        self,
        images: torch.Tensor,
        labels: dict[str, torch.Tensor] | None = None,
        train_mode: bool | None = None,
        scene_state_proxy: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if train_mode is None:
            train_mode = self.training
        dino = self.dino_extractor(images)
        tokens_by_layer = dino["tokens_by_layer"]
        maps_by_layer = dino["maps_by_layer"]
        final_layer = self.layer_indices[-1]
        final_tokens = tokens_by_layer[final_layer]

        base = self.base_actor(final_tokens)
        z_fate = base.get("action_fused_logits", base["action_logits"])
        base_reason = base["reason_logits"]
        y_action = labels.get("action") if labels is not None and "action" in labels else None

        action_maps, layer_gates = self.layer_mixer(maps_by_layer)
        pyramids = self.dense_adapter(action_maps)
        actor = self.visual_actor(pyramids)
        z_eva = actor["z_eva"]
        gate_out = self.mixture_gate(z_fate, z_eva, actor["evidence_confidence"], y_action=y_action, train_mode=bool(train_mode))
        z_actor = gate_out["z_actor"]

        effective_scene = scene_state_proxy if bool(train_mode) else None
        factor_bank = self.factor_bank(actor["action_evidence_tokens"], maps_by_layer[final_layer], scene_state_proxy=effective_scene, train_mode=bool(train_mode))
        factor_tokens = factor_bank["factor_tokens"]
        factor_groups = factor_bank["factor_groups"]
        action_tokens = actor["action_tokens"]
        route = self.factor_router(action_tokens, factor_tokens, factor_groups)
        selected_factors = self._gather_selected(factor_tokens, route["selected_factor_indices"])

        auditor = self.factor_auditor(
            z_fate=z_fate,
            z_actor=z_actor,
            gate=gate_out["visual_gate"],
            delta=gate_out["bounded_delta"],
            selected_weights=route["selected_factor_weights"],
            y_action=y_action,
        )
        reason = self.reason_decoder(selected_factors, actor["action_evidence_tokens"], base_reason)
        reason_gate = self.reason_reliability(reason["reason_factor_logits"], base_reason)
        final_reason = base_reason + reason_gate * torch.clamp(reason["reason_factor_logits"] - base_reason, -self.reason_decoder.reason_cap, self.reason_decoder.reason_cap)

        return {
            **base,
            "z_fate_action_logits": z_fate,
            "z_eva_action_logits": z_eva,
            "z_actor_action_logits": z_actor,
            "guarded_action_logits": z_actor,
            "base_reason_logits": base_reason,
            "reason_factor_logits": reason["reason_factor_logits"],
            "final_reason_logits": final_reason,
            "visual_gate": gate_out["visual_gate"],
            "gate_target": gate_out["gate_target"],
            "bounded_delta": gate_out["bounded_delta"],
            "action_evidence_tokens": actor["action_evidence_tokens"],
            "evidence_confidence": actor["evidence_confidence"],
            "action_tokens": action_tokens,
            "layer_gates": layer_gates,
            "factor_tokens": factor_tokens,
            "factor_groups": factor_groups,
            "factor_group_scores": route["factor_group_scores"],
            "selected_factor_indices": route["selected_factor_indices"],
            "selected_factor_weights": route["selected_factor_weights"],
            "selected_factors": selected_factors,
            "selected_vs_random_stats": auditor,
            "reason_to_factor_attention": reason["reason_to_factor_attention"],
            "reason_gate": reason_gate,
            "dino_patch_hw": dino.get("patch_hw", (45, 80)),
            "no_test_leakage_assertion": {
                "used_bdd100k_gt_in_test_forward": bool((not train_mode) and scene_state_proxy is not None and effective_scene is not None),
                "scene_state_proxy_accepted_in_train": bool(train_mode and scene_state_proxy is not None),
            },
        }
