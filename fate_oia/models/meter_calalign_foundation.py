from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .acpr_dino_field import ACPRDinoFieldExtractor
from .acpr_ego_regions import ACPREgoRegionEncoder
from .acpr_label_trunk import ACPRLabelTrunk
from .acpr_predicate_reason import ACPRPredicateReasoner
from .acpr_scene_predicate_head import ACPRScenePredicateHead


class METERCalAlignFoundation(nn.Module):
    """Frozen-DINO CalAlign-compatible visual foundation without ACPR side paths."""

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        scene_config: str = "configs/acpr_scene_predicates.yaml",
        grammar_path: str = "configs/acpr_reason_predicate_grammar.yaml",
        action_logit_norm_cap: float = 20.0,
        use_mock_dino: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.selected_layers = tuple(int(layer) for layer in selected_layers)
        self.dino = ACPRDinoFieldExtractor(
            selected_layers=self.selected_layers,
            pretrained_weights=pretrained_weights,
            use_mock_dino=use_mock_dino,
            freeze_backbone=True,
        )
        # Keep the frozen reference backbone in inference mode even if the
        # enclosing model later switches to train().
        self.dino.eval()
        self.ego = ACPREgoRegionEncoder(grid_hw=(45, 80), dim=dim)
        self.predicate_head = ACPRScenePredicateHead(
            scene_config=scene_config,
            dim=dim,
            num_layers=len(selected_layers),
        )
        self.trunk = ACPRLabelTrunk(
            dim=dim,
            action_dim=action_dim,
            reason_dim=reason_dim,
            action_logit_norm_cap=action_logit_norm_cap,
        )
        # SAVE's private reason decoder copies the full CalAlign primitive
        # set.  Keep this identity at the foundation boundary so older
        # CalAlign checkpoints remain loadable without changing V3 outputs.
        self.trunk.reason_norm = nn.LayerNorm(dim)
        self.predicate_reason = ACPRPredicateReasoner(
            dim=dim,
            reason_dim=reason_dim,
            num_predicates=self.predicate_head.num_predicates,
            predicate_names=self.predicate_head.names,
            grammar_path=grammar_path,
        )
        self.ordinary_dino_calls = 0

    def load_acpr_compatible_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        own = self.state_dict()
        compatible = {name: value for name, value in state_dict.items() if name in own and own[name].shape == value.shape}
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        optional_missing = {
            "trunk.reason_norm.weight",
            "trunk.reason_norm.bias",
        }
        missing = [name for name in missing if name not in optional_missing]
        if missing:
            raise RuntimeError(f"Missing CalAlign-compatible foundation keys: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected mapped foundation keys: {unexpected}")

    def train(self, mode: bool = True) -> "METERCalAlignFoundation":
        # ``nn.Module.train`` recursively flips every child back to train
        # mode.  Re-assert the DINO inference contract after that recursion.
        super().train(mode)
        self.dino.eval()
        return self

    def encode_images(self, images: Tensor) -> dict[str, Any]:
        self.ordinary_dino_calls += 1
        return dict(self.dino(images))

    def decode_foundation(self, field: dict[str, Any]) -> dict[str, Any]:
        patch_tokens = field["patch_tokens_by_layer"]
        if not isinstance(patch_tokens, Tensor) or patch_tokens.ndim != 4:
            raise ValueError("METER foundation requires [B,3,3600,D] patch tokens")
        patch0, ego_features, region_masks, ego_stats = self.ego(patch_tokens[:, 0])
        patched_layers = patch_tokens.clone()
        patched_layers[:, 0] = patch0
        predicates = self.predicate_head(patched_layers, region_masks=region_masks)
        trunk = self.trunk(patched_layers, predicate_tokens=predicates["predicate_tokens"])
        reason_delta = self.predicate_reason(
            trunk["label_nodes"][:, self.action_dim :],
            predicates["predicate_probs"],
            predicates["predicate_tokens"],
        )
        action_logits = trunk["action_logits_direct"]
        reason_logits = trunk["reason_logits_visual"] + reason_delta["predicate_reason_delta"]
        return {
            **trunk,
            **predicates,
            **reason_delta,
            "patch_tokens_by_layer": patched_layers,
            "foundation_patch": patched_layers.mean(dim=1),
            "ego_features": ego_features,
            "ego_region_masks": region_masks,
            "ego_stats": ego_stats,
            "action_nodes": trunk["label_nodes"][:, : self.action_dim],
            "factor_base_nodes": trunk["label_nodes"][:, self.action_dim :],
            "action_logits_visual_base": trunk["action_visual_logits"],
            "action_logits_reason_base": trunk["action_reason_logits"],
            "action_logits_calalign": action_logits,
            "reason_logits_calalign": reason_logits,
        }

    def forward(self, images: Tensor) -> dict[str, Any]:
        field = self.encode_images(images)
        decoded = self.decode_foundation(field)
        return {**field, **decoded}
