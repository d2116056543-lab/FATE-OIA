from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from .acpr_dino_field import ACPRDinoFieldExtractor
from .acpr_ego_regions import ACPREgoRegionEncoder
from .acpr_label_trunk import ACPRLabelTrunk
from .acpr_predicate_reason import ACPRPredicateReasoner
from .acpr_scene_predicate_head import ACPRScenePredicateHead


class AIECertCalAlignFoundation(nn.Module):
    """Value-equivalent AIE foundation with a predicate-to-primary gradient firewall."""

    def __init__(self, dim=384, action_dim=4, reason_dim=21, selected_layers=(3, 7, 11),
                 pretrained_weights="ckp/reference/dino_deitsmall8_pretrain.pth",
                 scene_config="configs/aie_scene_predicates.yaml",
                 grammar_path="configs/acpr_reason_predicate_grammar.yaml", use_mock_dino=False, mock_dim=None):
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.dino = ACPRDinoFieldExtractor(selected_layers=selected_layers, pretrained_weights=pretrained_weights,
                                           use_mock_dino=use_mock_dino, mock_dim=mock_dim or dim)
        self.ego = ACPREgoRegionEncoder(grid_hw=(45, 80), dim=dim)
        self.predicate_head = ACPRScenePredicateHead(scene_config=scene_config, dim=dim, num_layers=len(selected_layers))
        self.trunk = ACPRLabelTrunk(dim=dim, action_dim=action_dim, reason_dim=reason_dim)
        self.predicate_reason = ACPRPredicateReasoner(dim=dim, reason_dim=reason_dim,
            num_predicates=self.predicate_head.num_predicates, predicate_names=self.predicate_head.names,
            grammar_path=grammar_path)

    def encode_images(self, images: Tensor) -> dict[str, Any]:
        return self.dino(images)

    def decode_field(self, field: dict[str, Any]) -> dict[str, Any]:
        raw = field["patch_tokens_by_layer"]
        patch0, ego_features, regions, ego_stats = self.ego(raw[:, 0])
        patch = raw.clone()
        patch[:, 0] = patch0
        predicates = self.predicate_head(patch, region_masks=regions)
        predicate_tokens = predicates["predicate_tokens"].detach()
        predicate_probs = predicates["predicate_probs"].detach()
        trunk = self.trunk(patch, predicate_tokens=predicate_tokens)
        pred_reason = self.predicate_reason(trunk["label_nodes"][:, self.action_dim:], predicate_probs, predicate_tokens)
        return {**field, **predicates, **trunk, **pred_reason,
            "patch_tokens_by_layer_raw": raw,
            "patch_tokens_by_layer_ego": patch,
            "ego_features": ego_features,
            "ego_region_masks": regions,
            "ego_stats": ego_stats,
            "predicate_logits_clean": predicates["predicate_logits"],
            "predicate_probs_clean": predicates["predicate_probs"],
            "predicate_attention_clean": predicates["predicate_attention"],
            "action_nodes_primary": trunk["label_nodes"][:, :self.action_dim],
            "reason_nodes_primary": trunk["label_nodes"][:, self.action_dim:],
            "action_visual_logits_primary": trunk["action_visual_logits"],
            "action_reason_logits_primary": trunk["action_reason_logits"],
            "action_logits_primary": trunk["action_logits_direct"],
            "reason_logits_primary": trunk["reason_logits_visual"] + pred_reason["predicate_reason_delta"],
        }

    def forward(self, images: Tensor) -> dict[str, Any]:
        return self.decode_field(self.encode_images(images))

    def load_from_aie_state_dict(self, state: Mapping[str, Tensor], strict: bool = True):
        prefix = "foundation."
        filtered = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()
                    if k.startswith(prefix) or k.split(".", 1)[0] in {"dino", "ego", "predicate_head", "trunk", "predicate_reason"}}
        return self.load_state_dict(filtered, strict=strict)
