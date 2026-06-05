from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fate_oia.models.egcaf_dense_adapter import EGCafDrivingDenseAdapter
from fate_oia.models.egcaf_dino_multilayer import EGCafDinoMultiLayerExtractor
from fate_oia.models.egcaf_dynamic_selector import ExplanationGuidedDynamicFactorSelector
from fate_oia.models.egcaf_factor_actor import FactorActor
from fate_oia.models.egcaf_factor_bank import DrivingFactorCandidateBank
from fate_oia.models.egcaf_factor_types import FactorBatch, factor_to_json_records
from fate_oia.models.egcaf_reason_decoder import ReasonFromFactorDecoder


class EGCafOIAModel(nn.Module):
    def __init__(
        self,
        action_dim: int = 4,
        reason_dim: int = 21,
        hidden_dim: int = 256,
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        patch_size: int = 8,
        hook_layers: list[int] | None = None,
        lightweight_backbone: bool = False,
        residual_cap: float = 0.03,
        residual_enabled: bool = False,
        sparse_method: str = "entmax15",
    ) -> None:
        super().__init__()
        hook_layers = hook_layers or [3, 6, 9, 12]
        self.extractor = EGCafDinoMultiLayerExtractor(pretrained_weights=pretrained_weights, patch_size=patch_size, hook_layers=hook_layers, lightweight=lightweight_backbone)
        self.adapter = EGCafDrivingDenseAdapter(input_dim=384, hidden_dim=hidden_dim, num_actions=action_dim, layer_names=[f"layer_{i}" for i in hook_layers])
        self.factor_bank = DrivingFactorCandidateBank(hidden_dim=hidden_dim, num_actions=action_dim)
        self.pre_reason = nn.Linear(hidden_dim, reason_dim)
        self.selector = ExplanationGuidedDynamicFactorSelector(hidden_dim=hidden_dim, action_dim=action_dim, reason_dim=reason_dim, sparse_method=sparse_method)
        self.actor = FactorActor(hidden_dim=hidden_dim, action_dim=action_dim, residual_cap=residual_cap, residual_enabled_default=residual_enabled)
        self.reason_decoder = ReasonFromFactorDecoder(hidden_dim=hidden_dim, reason_dim=reason_dim, action_dim=action_dim)
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.residual_enabled = bool(residual_enabled)

    def forward(
        self,
        images: torch.Tensor,
        bdd100k_scene_state: torch.Tensor | None = None,
        return_artifacts: bool = False,
        mode: str = "train",
    ) -> dict[str, Any]:
        ext = self.extractor(images)
        dense = self.adapter(ext["layer_tokens"], ext["grid_hw"])
        bank = self.factor_bank(dense["pyramid"], weak_scene_state=bdd100k_scene_state if mode == "train" else None)
        factors: FactorBatch = bank["factors"]  # type: ignore[assignment]
        preliminary_reason_logits = self.pre_reason(factors.embeddings.mean(1))
        reason_unc = torch.sigmoid(preliminary_reason_logits) * (1 - torch.sigmoid(preliminary_reason_logits))
        sel = self.selector(factors, preliminary_reason_logits, reason_unc, bank["scene_state_logits"])  # type: ignore[arg-type]
        selected: FactorBatch = sel["selected_factors"]  # type: ignore[assignment]
        actor_out = self.actor(selected.embeddings, sel["selected_weights"], residual_enabled=self.residual_enabled)  # type: ignore[arg-type]
        reason_out = self.reason_decoder(selected.embeddings, bank["scene_state_tokens"])  # type: ignore[arg-type]
        b, a, k = sel["selected_indices"].shape  # type: ignore[index]
        m = factors.embeddings.shape[1]
        random_idx = torch.randint(0, m, (b, a, k), device=images.device)
        z_all = self.actor.mode_logits(factors.embeddings, sel["factor_weights"], sel["selected_indices"], mode="all")  # type: ignore[arg-type]
        z_selected = self.actor.mode_logits(factors.embeddings, sel["factor_weights"], sel["selected_indices"], mode="selected")  # type: ignore[arg-type]
        z_without_selected = self.actor.mode_logits(factors.embeddings, sel["factor_weights"], sel["selected_indices"], mode="without-selected")  # type: ignore[arg-type]
        z_without_random = self.actor.mode_logits(factors.embeddings, sel["factor_weights"], sel["selected_indices"], mode="without-random", random_indices=random_idx)  # type: ignore[arg-type]
        guarded = actor_out["action_final_logits"] if self.residual_enabled else actor_out["action_core_logits"]
        out: dict[str, Any] = {
            **actor_out,
            **reason_out,
            "action_logits": guarded,
            "guarded_action_logits": guarded,
            "reason_logits": reason_out["reason_logits"],
            "factor_embeddings": factors.embeddings,
            "factor_region_masks": factors.region_masks,
            "factor_boxes": factors.boxes,
            "factor_type_logits": factors.type_logits,
            "factor_action_ids": factors.action_ids,
            "factor_scores": sel["factor_scores"],
            "factor_weights": sel["factor_weights"],
            "factor_level_exp_prior": sel["factor_level_exp_prior"],
            "selected_indices": sel["selected_indices"],
            "selected_weights": sel["selected_weights"],
            "selected_factor_sources": selected.source_ids,
            "selected_factor_types": selected.type_logits.argmax(-1),
            "selected_factor_action_ids": selected.action_ids,
            "selected_factor_boxes": selected.boxes,
            "z_all": z_all,
            "z_selected_only": z_selected,
            "z_without_selected": z_without_selected,
            "z_without_random": z_without_random,
            "scene_state_logits": bank["scene_state_logits"],
            "scene_state_weak_labels_used": bank["scene_state_weak_labels_used"],
            "lambda_exp": sel["lambda_exp"],
            "selector_entropy": sel["selector_entropy"],
            "factor_bank_stats": bank["factor_bank_stats"],
            "factor_judge_stats": {},
            "grid_hw": ext["grid_hw"],
            "dense_stats": dense["dense_stats"],
            "layer_gates": dense["layer_gates"],
            "residual_enabled": torch.tensor(float(self.residual_enabled), device=images.device),
        }
        if return_artifacts:
            out["visual_factor_records"] = factor_to_json_records(factors, sel["selected_indices"], sel["selected_weights"])  # type: ignore[arg-type]
        return out

