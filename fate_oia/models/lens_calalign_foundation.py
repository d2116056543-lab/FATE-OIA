from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .acpr_dino_field import ACPRDinoFieldExtractor
from .acpr_ego_regions import ACPREgoRegionEncoder
from .acpr_label_trunk import ACPRLabelTrunk
from .acpr_predicate_reason import ACPRPredicateReasoner
from .acpr_scene_predicate_head import ACPRScenePredicateHead


@dataclass(frozen=True)
class LoadResult:
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]


class LENSCalAlignFoundation(nn.Module):
    """The ACPR core only; legacy pair, combo, threshold and calibration paths are absent."""

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
        self.dino = ACPRDinoFieldExtractor(
            selected_layers=selected_layers,
            pretrained_weights=pretrained_weights,
            use_mock_dino=use_mock_dino,
        )
        self.ego = ACPREgoRegionEncoder(grid_hw=(45, 80), dim=dim)
        self.scene_predicate = ACPRScenePredicateHead(scene_config=scene_config, dim=dim, num_layers=len(selected_layers))
        self.trunk = ACPRLabelTrunk(dim=dim, action_dim=action_dim, reason_dim=reason_dim)
        self.source_predicate_reason = ACPRPredicateReasoner(
            dim=dim,
            reason_dim=reason_dim,
            num_predicates=self.scene_predicate.num_predicates,
            predicate_names=self.scene_predicate.names,
            grammar_path=grammar_path,
        )

    def encode_images(self, images: Tensor) -> dict[str, Any]:
        return self.dino(images)

    def decode_field(self, field: dict[str, Any]) -> dict[str, Tensor | dict[str, Any]]:
        patch = field["patch_tokens_by_layer"]
        ego_patch0, _, region_masks, ego_stats = self.ego(patch[:, 0])
        patch = patch.clone()
        patch[:, 0] = ego_patch0
        predicates = self.scene_predicate(patch, region_masks=region_masks)
        trunk = self.trunk(patch, predicate_tokens=predicates["predicate_tokens"])
        reason_delta = self.source_predicate_reason(
            trunk["label_nodes"][:, self.action_dim :],
            predicates["predicate_probs"],
            predicates["predicate_tokens"],
        )
        reason_source = trunk["reason_logits_visual"] + reason_delta["predicate_reason_delta"]
        return {
            **field,
            **predicates,
            **trunk,
            **reason_delta,
            "patch_tokens_by_layer": patch,
            "action_logits_source": trunk["action_logits_direct"],
            "reason_logits_source": reason_source,
            "action_nodes_source": trunk["label_nodes"][:, : self.action_dim],
            "reason_nodes_source": trunk["label_nodes"][:, self.action_dim :],
            "action_visual_source": trunk["action_visual_logits"],
            "reason_visual_source": trunk["reason_logits_visual"],
            "action_reason_source": trunk["action_reason_logits"],
            "action_fusion_gate_source": trunk["action_fusion_gate"],
            "label_nodes_source": trunk["label_nodes"],
            "label_attention_source": trunk["label_attention"],
            "ego_stats": ego_stats,
        }

    def forward(self, images: Tensor) -> dict[str, Any]:
        return self.decode_field(self.encode_images(images))

    def load_from_acpr_state_dict(self, acpr_state_dict: Mapping[str, Tensor], strict: bool = True) -> LoadResult:
        mapped: dict[str, Tensor] = {}
        prefixes = {
            "dino.": "dino.",
            "ego.": "ego.",
            "predicate_head.": "scene_predicate.",
            "trunk.": "trunk.",
            "predicate_reason.": "source_predicate_reason.",
        }
        for key, value in acpr_state_dict.items():
            for old, new in prefixes.items():
                if key.startswith(old):
                    mapped[new + key[len(old) :]] = value
                    break
        result = self.load_state_dict(mapped, strict=strict)
        return LoadResult(tuple(result.missing_keys), tuple(result.unexpected_keys))
