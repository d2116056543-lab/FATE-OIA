from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fate_oia.models.fate_oia_model import FATEOIAFeatureModel
from fate_oia.models.diva_multilayer_dino import TinyPatchDINOExtractor
from fate_oia.models.diva_dense_adapter import ActionSpecificLayerMixer, DrivingDenseAdapter
from fate_oia.models.diva_visual_actor import DIVAVisualActor
from fate_oia.models.diva_visual_mixture_gate import SupervisedVisualMixtureGate, branch_safe_guarded_action
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
        self.reason_group_projector = nn.Linear(reason_dim, self.factor_router.num_groups)

    def _mix_action(self, z_fate: torch.Tensor, z_eva: torch.Tensor, visual_gate: torch.Tensor, delta_cap: float) -> torch.Tensor:
        return z_fate + visual_gate * torch.clamp(z_eva - z_fate, -delta_cap, delta_cap)

    def _gather_selected(self, factors: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        b, n, d = factors.shape
        _, a, k = indices.shape
        expanded = factors.unsqueeze(1).expand(b, a, n, d)
        gather_idx = indices.clamp_min(0).clamp_max(max(n - 1, 0)).unsqueeze(-1).expand(b, a, k, d)
        return torch.gather(expanded, 2, gather_idx)

    def _gather_meta(self, meta: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        b, n = meta.shape[:2]
        trailing = meta.shape[2:]
        expanded = meta.unsqueeze(1).expand(b, self.action_dim, n, *trailing)
        gather_idx = indices.clamp_min(0).clamp_max(max(n - 1, 0))
        for _ in trailing:
            gather_idx = gather_idx.unsqueeze(-1)
        gather_idx = gather_idx.expand(b, self.action_dim, indices.shape[-1], *trailing)
        return torch.gather(expanded, 2, gather_idx)

    def _selected_indices_to_evidence_mask(self, indices: torch.Tensor, evidence_per_action: int, randomize: bool = False) -> torch.Tensor:
        b, a, k = indices.shape
        mask = indices.new_zeros((b, self.action_dim, evidence_per_action), dtype=torch.float32)
        actor_factor_count = self.action_dim * evidence_per_action
        chosen = indices
        if randomize:
            chosen = torch.roll(chosen, shifts=1, dims=-1)
        valid = (chosen >= 0) & (chosen < actor_factor_count)
        src_action = (chosen.clamp_min(0) // evidence_per_action).clamp(0, self.action_dim - 1)
        src_evidence = (chosen.clamp_min(0) % evidence_per_action).clamp(0, evidence_per_action - 1)
        for bi in range(b):
            for ai in range(a):
                for ki in range(k):
                    if bool(valid[bi, ai, ki]) and int(src_action[bi, ai, ki]) == ai:
                        mask[bi, ai, int(src_evidence[bi, ai, ki])] = 1.0
        return mask.to(indices.device)

    def _per_action_group_drop(self, per_action_delta: torch.Tensor, selected_indices: torch.Tensor, factor_groups: torch.Tensor) -> torch.Tensor:
        b, a, k = selected_indices.shape
        out = per_action_delta.new_zeros(self.action_dim, self.factor_router.num_groups)
        counts = per_action_delta.new_zeros(self.action_dim, self.factor_router.num_groups)
        groups = torch.gather(
            factor_groups.unsqueeze(1).expand(b, a, factor_groups.shape[1]),
            2,
            selected_indices.clamp_min(0).clamp_max(max(factor_groups.shape[1] - 1, 0)),
        )
        for ai in range(a):
            for gi in range(self.factor_router.num_groups):
                m = (groups[:, ai] == gi).float()
                if float(m.sum()) > 0:
                    out[ai, gi] = (per_action_delta[:, ai].unsqueeze(-1) * m).sum() / m.sum().clamp_min(1.0)
                    counts[ai, gi] = m.sum()
        return out

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
        guarded_action, guarded_stats = branch_safe_guarded_action(z_fate, z_actor, y_action, tolerance=0.0)

        effective_scene = scene_state_proxy if bool(train_mode) else None
        factor_bank = self.factor_bank(actor["action_evidence_tokens"], maps_by_layer[final_layer], scene_state_proxy=effective_scene, train_mode=bool(train_mode))
        factor_tokens = factor_bank["factor_tokens"]
        factor_groups = factor_bank["factor_groups"]
        action_tokens = actor["action_tokens"]
        factor_type_prob = torch.softmax(factor_bank["factor_type_logits"], dim=-1)
        reason_group_prob = torch.softmax(self.reason_group_projector(torch.sigmoid(base_reason)), dim=-1)
        group_idx = factor_groups.clamp(0, self.factor_router.num_groups - 1)
        exp_prior_base = torch.gather(reason_group_prob, 1, group_idx)
        exp_prior = exp_prior_base.unsqueeze(1).expand(-1, self.action_dim, -1)
        type_conf = factor_type_prob.max(dim=-1).values
        exp_reliability = type_conf.unsqueeze(1).expand(-1, self.action_dim, -1)
        route = self.factor_router(
            action_tokens,
            factor_tokens,
            factor_groups,
            action_uncertainty=gate_out["action_uncertainty"],
            exp_prior=exp_prior,
            exp_reliability=exp_reliability,
        )
        selected_factors = self._gather_selected(factor_tokens, route["selected_factor_indices"])
        selected_regions = self._gather_meta(factor_bank["factor_region"], route["selected_factor_indices"])
        selected_sources = self._gather_meta(factor_bank["factor_source_id"].unsqueeze(-1).float(), route["selected_factor_indices"]).squeeze(-1)

        evidence_per_action = actor["action_evidence_tokens"].shape[2]
        selected_mask = self._selected_indices_to_evidence_mask(route["selected_factor_indices"], evidence_per_action, randomize=False)
        random_mask = self._selected_indices_to_evidence_mask(route["selected_factor_indices"], evidence_per_action, randomize=True)
        z_eva_without_selected = self.visual_actor.score_from_action_evidence(actor["action_evidence_tokens"], selected_mask)["z_eva"]
        z_eva_without_random = self.visual_actor.score_from_action_evidence(actor["action_evidence_tokens"], random_mask)["z_eva"]
        z_actor_without_selected = self._mix_action(z_fate, z_eva_without_selected, gate_out["visual_gate"], self.mixture_gate.delta_cap)
        z_actor_without_random = self._mix_action(z_fate, z_eva_without_random, gate_out["visual_gate"], self.mixture_gate.delta_cap)
        per_action_delta = torch.zeros_like(z_actor)
        if y_action is not None:
            import torch.nn.functional as F

            per_action_delta = (
                F.binary_cross_entropy_with_logits(z_actor_without_selected, y_action.float(), reduction="none")
                - F.binary_cross_entropy_with_logits(z_actor, y_action.float(), reduction="none")
            ).detach()
        per_action_group = self._per_action_group_drop(per_action_delta, route["selected_factor_indices"], factor_groups)
        auditor = self.factor_auditor(
            z_actor_full=z_actor,
            z_actor_without_selected=z_actor_without_selected,
            z_actor_without_random=z_actor_without_random,
            y_action=y_action,
            per_action_group=per_action_group,
        )
        reason = self.reason_decoder(selected_factors, actor["action_evidence_tokens"], base_reason)
        reason_gate = self.reason_reliability(
            reason["reason_factor_logits"],
            factor_support=reason["factor_reason_support"],
            base_reason_logits=base_reason,
        ).detach() + 0.0 * base_reason
        final_reason = base_reason + reason_gate * torch.clamp(reason["reason_factor_logits"] - base_reason, -self.reason_decoder.reason_cap, self.reason_decoder.reason_cap)

        return {
            **base,
            "z_fate_action_logits": z_fate,
            "z_eva_action_logits": z_eva,
            "z_eva_without_action_set": actor["z_eva_without_action_set"],
            "z_eva_action_set_delta": actor["z_eva_action_set_delta"],
            "z_actor_action_logits": z_actor,
            "guarded_action_logits": guarded_action,
            "guarded_action_stats": guarded_stats,
            "base_reason_logits": base_reason,
            "reason_factor_logits": reason["reason_factor_logits"],
            "final_reason_logits": final_reason,
            "tail_reason_indices": reason["tail_reason_indices"],
            "visual_gate": gate_out["visual_gate"],
            "gate_target": gate_out["gate_target"],
            "bounded_delta": gate_out["bounded_delta"],
            "action_evidence_tokens": actor["action_evidence_tokens"],
            "evidence_confidence": actor["evidence_confidence"],
            "evidence_sample_points": actor.get("evidence_sample_points"),
            "evidence_scale_usage": actor.get("evidence_scale_usage"),
            "action_set_prototype_scores": actor.get("action_set_prototype_scores"),
            "prototype_score_mean": actor.get("prototype_score_mean"),
            "prototype_residual_norm": actor.get("prototype_residual_norm"),
            "action_tokens": action_tokens,
            "layer_gates": layer_gates,
            "factor_tokens": factor_tokens,
            "factor_groups": factor_groups,
            "factor_region": factor_bank["factor_region"],
            "factor_source_id": factor_bank["factor_source_id"],
            "factor_group_scores": route["factor_group_scores"],
            "factor_type_logits": factor_bank["factor_type_logits"],
            "exp_prior": exp_prior,
            "exp_reliability": exp_reliability,
            "weak_exp_scores": route["weak_exp_scores"],
            "lambda_exp": route["lambda_exp"],
            "faith_ema": route["faith_ema"],
            "help_ema": route["help_ema"],
            "hurt_ema": route["hurt_ema"],
            "selected_factor_indices": route["selected_factor_indices"],
            "selected_factor_weights": route["selected_factor_weights"],
            "selected_factors": selected_factors,
            "selected_factor_meta": {"region": selected_regions, "source_id": selected_sources},
            "selected_vs_random_stats": auditor,
            "z_actor_without_selected": z_actor_without_selected,
            "z_actor_without_random": z_actor_without_random,
            "selected_evidence_mask": selected_mask,
            "random_evidence_mask": random_mask,
            "reason_to_factor_attention": reason["reason_to_factor_attention"],
            "factor_reason_support": reason["factor_reason_support"],
            "reason_gate": reason_gate,
            "dino_patch_hw": dino.get("patch_hw", (45, 80)),
            "dino_layer_stats": dino.get("layer_stats", {}),
            "dino_extractor_type": dino.get("extractor_type", "unknown"),
            "dino_load_info": dino.get("load_info", {}),
            "no_test_leakage_assertion": {
                "used_bdd100k_gt_in_test_forward": bool((not train_mode) and scene_state_proxy is not None and effective_scene is not None),
                "scene_state_proxy_accepted_in_train": bool(train_mode and scene_state_proxy is not None),
            },
        }
