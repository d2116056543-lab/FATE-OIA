from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .acpr_dino_field import ACPRDinoFieldExtractor
from .acpr_ego_regions import ACPREgoRegionEncoder
from .acpr_label_trunk import ACPRLabelTrunk
from .acpr_predicate_reason import ACPRPredicateReasoner
from .acpr_scene_predicate_head import ACPRScenePredicateHead


class AIECalAlignFoundation(nn.Module):
    """The exact raw ACPR-CalAlign path, without historical auxiliary heads."""

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        scene_config: str = "configs/aie_scene_predicates.yaml",
        grammar_path: str = "configs/acpr_reason_predicate_grammar.yaml",
        use_mock_dino: bool = False,
        mock_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.dino = ACPRDinoFieldExtractor(
            selected_layers=selected_layers,
            pretrained_weights=pretrained_weights,
            use_mock_dino=use_mock_dino,
            mock_dim=mock_dim or dim,
        )
        self.ego = ACPREgoRegionEncoder(grid_hw=(45, 80), dim=dim)
        self.predicate_head = ACPRScenePredicateHead(scene_config=scene_config, dim=dim, num_layers=len(selected_layers))
        self.trunk = ACPRLabelTrunk(dim=dim, action_dim=action_dim, reason_dim=reason_dim)
        self.predicate_reason = ACPRPredicateReasoner(
            dim=dim,
            reason_dim=reason_dim,
            num_predicates=self.predicate_head.num_predicates,
            predicate_names=self.predicate_head.names,
            grammar_path=grammar_path,
        )

    def encode_images(self, images: Tensor) -> dict[str, Any]:
        return self.dino(images)

    def decode_field(self, field: dict[str, Any]) -> dict[str, Any]:
        raw_patch = field["patch_tokens_by_layer"]
        patch0, ego_features, region_masks, ego_stats = self.ego(raw_patch[:, 0])
        patch = raw_patch.clone()
        patch[:, 0] = patch0
        predicates = self.predicate_head(patch, region_masks=region_masks)
        trunk = self.trunk(patch, predicate_tokens=predicates["predicate_tokens"])
        predicate_reason = self.predicate_reason(
            trunk["label_nodes"][:, self.action_dim :],
            predicates["predicate_probs"],
            predicates["predicate_tokens"],
        )
        action_primary = trunk["action_logits_direct"]
        reason_primary = trunk["reason_logits_visual"] + predicate_reason["predicate_reason_delta"]
        return {
            **field,
            **predicates,
            **trunk,
            **predicate_reason,
            "patch_tokens_by_layer_raw": raw_patch,
            "patch_tokens_by_layer_ego": patch,
            "ego_features": ego_features,
            "ego_region_masks": region_masks,
            "ego_stats": ego_stats,
            "action_nodes_primary": trunk["label_nodes"][:, : self.action_dim],
            "reason_nodes_primary": trunk["label_nodes"][:, self.action_dim :],
            "action_visual_logits_primary": trunk["action_visual_logits"],
            "action_reason_logits_primary": trunk["action_reason_logits"],
            "action_fusion_gate_primary": trunk["action_fusion_gate"],
            "action_logits_primary": action_primary,
            "reason_logits_visual_primary": trunk["reason_logits_visual"],
            "predicate_reason_delta_primary": predicate_reason["predicate_reason_delta"],
            "reason_logits_primary": reason_primary,
        }

    def forward(self, images: Tensor) -> dict[str, Any]:
        return self.decode_field(self.encode_images(images))

    def load_from_acpr_state_dict(
        self,
        source_state_dict: Mapping[str, Tensor],
        strict: bool = True,
    ) -> dict[str, list[str]]:
        prefixes = ("dino.", "ego.", "predicate_head.", "trunk.", "predicate_reason.")
        filtered = {key: value for key, value in source_state_dict.items() if key.startswith(prefixes)}
        result = self.load_state_dict(filtered, strict=strict)
        return {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}


