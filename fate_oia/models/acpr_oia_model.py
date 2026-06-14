from __future__ import annotations

import torch
from torch import nn

from .acpr_action_combo_aux import ACPRActionComboAux
from .acpr_calibration import ACPRCalibrationHead
from .acpr_dino_field import ACPRDinoFieldExtractor
from .acpr_ego_regions import ACPREgoRegionEncoder
from .acpr_label_trunk import ACPRLabelTrunk
from .acpr_pair_memory import ACPRPairMemory
from .acpr_predicate_reason import ACPRPredicateReasoner
from .acpr_scene_predicate_head import ACPRScenePredicateHead


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
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.dino = ACPRDinoFieldExtractor(selected_layers=selected_layers, pretrained_weights=pretrained_weights, use_mock_dino=use_mock_dino)
        self.ego = ACPREgoRegionEncoder(grid_hw=(45, 80), dim=dim)
        self.predicate_head = ACPRScenePredicateHead(scene_config=scene_config, dim=dim, num_layers=len(selected_layers))
        self.trunk = ACPRLabelTrunk(dim=dim, action_dim=action_dim, reason_dim=reason_dim)
        self.predicate_reason = ACPRPredicateReasoner(dim=dim, reason_dim=reason_dim, num_predicates=self.predicate_head.num_predicates, predicate_names=self.predicate_head.names, grammar_path=grammar_path)
        self.pair_memory = ACPRPairMemory(dim=dim)
        self.action_combo_aux = ACPRActionComboAux(dim=dim, action_dim=action_dim)
        self.calibration = ACPRCalibrationHead(num_labels=action_dim + reason_dim)

    def forward(self, images: torch.Tensor, epoch: int = 0) -> dict[str, torch.Tensor | dict | tuple[int, int] | int]:
        field = self.dino(images)
        patch = field["patch_tokens_by_layer"]
        patch0, ego_features, region_masks, ego_stats = self.ego(patch[:, 0])
        patch = patch.clone()
        patch[:, 0] = patch0
        predicates = self.predicate_head(patch, region_masks=region_masks)
        trunk = self.trunk(patch, predicate_tokens=predicates["predicate_tokens"])
        reason_delta = self.predicate_reason(trunk["label_nodes"][:, self.action_dim :], predicates["predicate_probs"], predicates["predicate_tokens"])
        action_logits_raw = trunk["action_logits_direct"]
        reason_logits_raw = trunk["reason_logits_visual"] + reason_delta["predicate_reason_delta"]
        raw_logits = torch.cat([action_logits_raw, reason_logits_raw], dim=-1)
        calibrated = self.calibration(action_logits_raw, reason_logits_raw)
        action_set = self.action_combo_aux(trunk["label_nodes"], action_logits_raw)
        cardinality_logits = action_set["cardinality_logits"]
        pair_embedding = self.pair_memory(trunk["label_nodes"])
        out = {
            **field,
            **trunk,
            **predicates,
            **reason_delta,
            **action_set,
            "global_embedding": field["cls_tokens_by_layer"].mean(1),
            "pair_embedding": pair_embedding,
            "action_logits_raw": action_logits_raw,
            "reason_logits_raw": reason_logits_raw,
            "action_logits_final_raw": action_logits_raw,
            "reason_logits_final_raw": reason_logits_raw,
            "action_logits_calibrated": calibrated["action_logits_calibrated"],
            "reason_logits_calibrated": calibrated["reason_logits_calibrated"],
            "logits_final_raw": raw_logits,
            "logits_final_calibrated": calibrated["calibrated_logits"],
            "action_logits_final_calibrated": calibrated["action_logits_calibrated"],
            "reason_logits_final_calibrated": calibrated["reason_logits_calibrated"],
            "temperature": calibrated["temperature"],
            "calibration_bias": calibrated["calibration_bias"],
            "cardinality_logits": cardinality_logits,
            "bias_action": calibrated["bias_action"],
            "bias_reason": calibrated["bias_reason"],
            "temperature_action": calibrated["temperature_action"],
            "temperature_reason": calibrated["temperature_reason"],
            "ego_stats": ego_stats,
            "branch_logits": {
                "direct": raw_logits,
                "direct_plus_predicate": raw_logits,
                "raw": raw_logits,
                "calibrated": calibrated["calibrated_logits"],
                "final_raw": raw_logits,
                "final_calibrated": calibrated["calibrated_logits"],
                "action_visual": trunk["action_visual_logits"],
                "action_reason": trunk["action_reason_logits"],
            },
        }
        return out
