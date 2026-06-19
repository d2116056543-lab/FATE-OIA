from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .acpr_action_combo_aux import ACPRActionComboAux
from .acpr_calibration import ACPRCalibrationHead
from .acpr_dino_field import ACPRDinoFieldExtractor
from .acpr_ego_regions import ACPREgoRegionEncoder
from .acpr_label_trunk import ACPRLabelTrunk
from .acpr_pair_memory import ACPRPairMemory
from .acpr_predicate_action_coupling import ACPRPredicateActionCoupling
from .acpr_predicate_reason import ACPRPredicateReasoner
from .acpr_scene_predicate_head import ACPRScenePredicateHead
from .acpr_threshold_head import ACPRThresholdHead


class ACPROIAModel(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        scene_config: str = "configs/acpr_scene_predicates.yaml",
        grammar_path: str = "configs/acpr_reason_predicate_grammar.yaml",
        use_mock_dino: bool = False,
        threshold_enabled: bool = False,
        threshold_kwargs: dict | None = None,
        pace_enabled: bool = False,
        pace_coupling_strength: float = 1.0,
        pace_max_action_delta: float = 0.20,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.threshold_enabled = bool(threshold_enabled)
        self.pace_enabled = bool(pace_enabled)
        self.dino = ACPRDinoFieldExtractor(selected_layers=selected_layers, pretrained_weights=pretrained_weights, use_mock_dino=use_mock_dino)
        self.ego = ACPREgoRegionEncoder(grid_hw=(45, 80), dim=dim)
        self.predicate_head = ACPRScenePredicateHead(scene_config=scene_config, dim=dim, num_layers=len(selected_layers))
        self.trunk = ACPRLabelTrunk(dim=dim, action_dim=action_dim, reason_dim=reason_dim)
        self.predicate_reason = ACPRPredicateReasoner(dim=dim, reason_dim=reason_dim, num_predicates=self.predicate_head.num_predicates, predicate_names=self.predicate_head.names, grammar_path=grammar_path)
        self.pair_memory = ACPRPairMemory(dim=dim)
        self.reason_pair_proj = nn.Linear(dim, dim)
        self.action_combo_aux = ACPRActionComboAux(dim=dim, action_dim=action_dim)
        self.calibration = ACPRCalibrationHead(num_labels=action_dim + reason_dim)
        self.threshold_head = ACPRThresholdHead(action_dim=action_dim, reason_dim=reason_dim, **(threshold_kwargs or {}))
        self.predicate_action_coupling = ACPRPredicateActionCoupling(action_dim=action_dim, reason_dim=reason_dim, coupling_strength=pace_coupling_strength, max_action_delta=pace_max_action_delta)

    def forward(self, images: torch.Tensor, epoch: int = 0) -> dict[str, torch.Tensor | dict | tuple[int, int] | int]:
        field = self.dino(images)
        patch = field["patch_tokens_by_layer"]
        patch0, ego_features, region_masks, ego_stats = self.ego(patch[:, 0])
        patch = patch.clone()
        patch[:, 0] = patch0
        predicates = self.predicate_head(patch, region_masks=region_masks)
        trunk = self.trunk(patch, predicate_tokens=predicates["predicate_tokens"])
        reason_delta = self.predicate_reason(trunk["label_nodes"][:, self.action_dim :], predicates["predicate_probs"], predicates["predicate_tokens"])
        reason_logits_base = trunk["reason_logits_visual"] + reason_delta["predicate_reason_delta"]
        action_reason_logits_pace = self.trunk.project_reason_logits_to_action(reason_logits_base)
        if self.pace_enabled:
            pace = self.predicate_action_coupling(
                trunk["action_visual_logits"],
                trunk["action_reason_logits_visual"],
                trunk["action_fusion_gate"],
                reason_delta["predicate_reason_delta"],
                self.trunk.reason_to_action.weight,
                self.trunk.reason_to_action.bias,
            )
            action_logits_base = pace["action_logits_pace"]
        else:
            pace = self.predicate_action_coupling(
                trunk["action_visual_logits"],
                trunk["action_reason_logits_visual"],
                trunk["action_fusion_gate"],
                reason_delta["predicate_reason_delta"],
                self.trunk.reason_to_action.weight,
                self.trunk.reason_to_action.bias,
                coupling_strength=0.0,
            )
            action_logits_base = trunk["action_logits_direct"]
        logits_base = torch.cat([action_logits_base, reason_logits_base], dim=-1)
        thresholded = self.threshold_head(action_logits_base, reason_logits_base)
        legacy_calibrated = self.calibration(action_logits_base, reason_logits_base)
        if self.threshold_enabled:
            action_logits_final_raw = thresholded["action_logits_deploy"]
            reason_logits_final_raw = thresholded["reason_logits_deploy"]
            logits_final_raw = thresholded["logits_deploy"]
            action_logits_final_calibrated = thresholded["action_logits_calibrated"]
            reason_logits_final_calibrated = thresholded["reason_logits_calibrated"]
            logits_final_calibrated = thresholded["logits_calibrated"]
            temperature = thresholded["temperature"]
            calibration_bias = torch.zeros_like(temperature)
        else:
            action_logits_final_raw = action_logits_base
            reason_logits_final_raw = reason_logits_base
            logits_final_raw = logits_base
            action_logits_final_calibrated = legacy_calibrated["action_logits_calibrated"]
            reason_logits_final_calibrated = legacy_calibrated["reason_logits_calibrated"]
            logits_final_calibrated = legacy_calibrated["calibrated_logits"]
            temperature = legacy_calibrated["temperature"]
            calibration_bias = legacy_calibrated["calibration_bias"]
        action_set = self.action_combo_aux(trunk["label_nodes"], action_logits_base)
        cardinality_logits = action_set["cardinality_logits"]
        reason_embeddings_for_pair = F.normalize(self.reason_pair_proj(trunk["label_nodes"][:, self.action_dim :]), dim=-1)
        pair_embedding = self.pair_memory(trunk["label_nodes"])
        out = {
            **field,
            **trunk,
            **predicates,
            **reason_delta,
            **action_set,
            **pace,
            "action_reason_logits_pace_projected": action_reason_logits_pace,
            "global_embedding": field["cls_tokens_by_layer"].mean(1),
            "pair_embedding": pair_embedding,
            "reason_embeddings_for_pair": reason_embeddings_for_pair,
            # Backward-compatible raw aliases are base logits. CalAlign's
            # action/reason_logits_final_raw are deploy logits when enabled.
            "action_logits_raw": action_logits_base,
            "reason_logits_raw": reason_logits_base,
            "action_logits_base": action_logits_base,
            "reason_logits_base": reason_logits_base,
            "logits_base_fixed": logits_base,
            "action_logits_deploy": thresholded["action_logits_deploy"],
            "reason_logits_deploy": thresholded["reason_logits_deploy"],
            "logits_deploy": thresholded["logits_deploy"],
            "threshold_logit": thresholded["threshold_logit"],
            "threshold_prob": thresholded["threshold_prob"],
            "action_threshold_prob": thresholded["action_threshold_prob"],
            "reason_threshold_prob": thresholded["reason_threshold_prob"],
            "action_logits_final_raw": action_logits_final_raw,
            "reason_logits_final_raw": reason_logits_final_raw,
            "action_logits_calibrated": action_logits_final_calibrated,
            "reason_logits_calibrated": reason_logits_final_calibrated,
            "logits_final_raw": logits_final_raw,
            "logits_final_calibrated": logits_final_calibrated,
            "action_logits_final_calibrated": action_logits_final_calibrated,
            "reason_logits_final_calibrated": reason_logits_final_calibrated,
            "temperature": temperature,
            "calibration_bias": calibration_bias,
            "cardinality_logits": cardinality_logits,
            "bias_action": calibration_bias[: self.action_dim],
            "bias_reason": calibration_bias[self.action_dim :],
            "temperature_action": temperature[: self.action_dim],
            "temperature_reason": temperature[self.action_dim :],
            "ego_stats": ego_stats,
            "branch_logits": {
                "direct": logits_base,
                "direct_plus_predicate": logits_base,
                "legacy": torch.cat([pace["action_logits_legacy"], reason_logits_base], dim=-1),
                "base_fixed": logits_base,
                "deploy_fixed": logits_final_raw,
                "raw": logits_final_raw,
                "calibrated": logits_final_calibrated,
                "final_raw": logits_final_raw,
                "final_calibrated": logits_final_calibrated,
                "action_visual": trunk["action_visual_logits"],
                "action_reason": pace["action_reason_logits_pace"],
            },
        }
        return out
