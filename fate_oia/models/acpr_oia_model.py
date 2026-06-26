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
from .acpr_predicate_reason import ACPRPredicateReasoner
from .acpr_scene_predicate_head import ACPRScenePredicateHead
from .acpr_threshold_head import ACPRThresholdHead
from .acpr_visual_token_adapter import ACPRPredicateAnchoredVisualAdapter, VistaScaleSchedule


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
        vista_enabled: bool = False,
        vista_kwargs: dict | None = None,
        pair_memory_size: int = 8192,
        pair_memory_device: str = "cpu",
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.threshold_enabled = bool(threshold_enabled)
        self.vista_enabled = bool(vista_enabled)
        self.dino = ACPRDinoFieldExtractor(selected_layers=selected_layers, pretrained_weights=pretrained_weights, use_mock_dino=use_mock_dino)
        self.ego = ACPREgoRegionEncoder(grid_hw=(45, 80), dim=dim)
        self.predicate_head = ACPRScenePredicateHead(scene_config=scene_config, dim=dim, num_layers=len(selected_layers))
        self.trunk = ACPRLabelTrunk(dim=dim, action_dim=action_dim, reason_dim=reason_dim)
        self.predicate_reason = ACPRPredicateReasoner(dim=dim, reason_dim=reason_dim, num_predicates=self.predicate_head.num_predicates, predicate_names=self.predicate_head.names, grammar_path=grammar_path)
        self.pair_memory = ACPRPairMemory(
            dim=dim,
            memory_size=int(pair_memory_size),
            memory_device=str(pair_memory_device),
        )
        self.reason_pair_proj = nn.Linear(dim, dim)
        self.action_combo_aux = ACPRActionComboAux(dim=dim, action_dim=action_dim)
        self.calibration = ACPRCalibrationHead(num_labels=action_dim + reason_dim)
        self.threshold_head = ACPRThresholdHead(action_dim=action_dim, reason_dim=reason_dim, **(threshold_kwargs or {}))
        vista_cfg = dict(vista_kwargs or {})
        self.visual_adapter = ACPRPredicateAnchoredVisualAdapter(
            dim=dim,
            rank=int(vista_cfg.get("rank", 48)),
            num_layers=len(selected_layers),
            num_predicates=self.predicate_head.num_predicates,
            gate_floor=float(vista_cfg.get("gate_floor", 0.20)),
            detach_predicate_gate=bool(vista_cfg.get("detach_predicate_gate", True)),
            predicate_names=self.predicate_head.names,
            reliable_predicate_weight=float(vista_cfg.get("reliable_predicate_weight", 1.0)),
            global_predicate_weight=float(vista_cfg.get("global_predicate_weight", 0.3)),
            unreliable_predicate_weight=float(vista_cfg.get("unreliable_predicate_weight", 0.0)),
            anchor_mix_start_epoch=int(vista_cfg.get("anchor_mix_start_epoch", 2)),
            anchor_mix_end_epoch=int(vista_cfg.get("anchor_mix_end_epoch", 5)),
            early_global_gate=bool(vista_cfg.get("early_global_gate", True)),
            base_fraction=float(vista_cfg.get("base_fraction", 0.20)),
            learned_fraction=float(vista_cfg.get("learned_fraction", 0.10)),
            schedule=VistaScaleSchedule(
                early_scale=float(vista_cfg.get("early_scale", 0.05)),
                main_scale=float(vista_cfg.get("main_scale", 0.15)),
                late_scale=float(vista_cfg.get("late_scale", 0.08)),
                main_start_epoch=int(vista_cfg.get("main_start_epoch", 3)),
                late_start_epoch=int(vista_cfg.get("late_start_epoch", 9)),
            ),
        )

    def forward(self, images: torch.Tensor, epoch: int = 0) -> dict[str, torch.Tensor | dict | tuple[int, int] | int]:
        field = self.dino(images)
        patch_raw = field["patch_tokens_by_layer"]
        patch0_raw, _, raw_region_masks, raw_ego_stats = self.ego(patch_raw[:, 0])
        patch_for_gate = patch_raw.clone()
        patch_for_gate[:, 0] = patch0_raw
        with torch.no_grad():
            raw_predicates_for_gate = self.predicate_head(patch_for_gate, region_masks=raw_region_masks)
        if self.vista_enabled:
            patch_adapted, vista_stats = self.visual_adapter(
                patch_raw,
                raw_predicates_for_gate["predicate_probs"],
                raw_predicates_for_gate["predicate_attention"],
                epoch=epoch,
            )
            patch = patch_adapted
        else:
            patch = patch_raw
            zero_gate = torch.zeros(patch.shape[0], patch.shape[2], device=patch.device, dtype=patch.dtype)
            vista_stats = {
                "vista_enabled": False,
                "vista_alpha_per_layer": torch.zeros(patch.shape[1], device=patch.device, dtype=patch.dtype),
                "vista_alpha_abs_mean": torch.zeros((), device=patch.device, dtype=patch.dtype),
                "vista_adapter_delta_norm_per_layer": torch.zeros(patch.shape[1], device=patch.device, dtype=patch.dtype),
                "vista_adapter_delta_norm_mean": torch.zeros((), device=patch.device, dtype=patch.dtype),
                "vista_gate_map": zero_gate,
                "vista_gate_mean": torch.zeros((), device=patch.device, dtype=patch.dtype),
                "vista_gate_max": torch.zeros((), device=patch.device, dtype=patch.dtype),
                "vista_gate_entropy": torch.zeros((), device=patch.device, dtype=patch.dtype),
                "vista_delta_mass_on_high_gate": torch.zeros((), device=patch.device, dtype=patch.dtype),
                "vista_delta_uniformity": torch.zeros((), device=patch.device, dtype=patch.dtype),
                "vista_anchor_mix": 0.0,
                "vista_predicate_importance_prior": torch.zeros(self.predicate_head.num_predicates),
                "vista_predicate_importance": torch.zeros(self.predicate_head.num_predicates),
                "vista_predicate_names": list(self.predicate_head.names),
            }
        patch0, ego_features, region_masks, ego_stats = self.ego(patch[:, 0])
        patch = patch.clone()
        patch[:, 0] = patch0
        predicates = self.predicate_head(patch, region_masks=region_masks)
        trunk = self.trunk(patch, predicate_tokens=predicates["predicate_tokens"])
        reason_delta = self.predicate_reason(trunk["label_nodes"][:, self.action_dim :], predicates["predicate_probs"], predicates["predicate_tokens"])
        action_logits_base = trunk["action_logits_direct"]
        reason_logits_base = trunk["reason_logits_visual"] + reason_delta["predicate_reason_delta"]
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
            "patch_tokens_by_layer_raw": patch_raw,
            "patch_tokens_by_layer_adapted": patch,
            **trunk,
            **predicates,
            **reason_delta,
            **action_set,
            **vista_stats,
            "vista_raw_predicate_probs_for_gate": raw_predicates_for_gate["predicate_probs"],
            "vista_raw_predicate_attention_for_gate": raw_predicates_for_gate["predicate_attention"],
            "vista_final_predicate_probs": predicates["predicate_probs"],
            "vista_patch_tokens_raw_stats": {
                "mean": float(patch_raw.mean().detach().cpu()),
                "std": float(patch_raw.std().detach().cpu()),
            },
            "vista_patch_tokens_adapted_stats": {
                "mean": float(patch.mean().detach().cpu()),
                "std": float(patch.std().detach().cpu()),
            },
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
            "raw_ego_stats": raw_ego_stats,
            "branch_logits": {
                "direct": logits_base,
                "direct_plus_predicate": logits_base,
                "base_fixed": logits_base,
                "deploy_fixed": logits_final_raw,
                "raw": logits_final_raw,
                "calibrated": logits_final_calibrated,
                "final_raw": logits_final_raw,
                "final_calibrated": logits_final_calibrated,
                "action_visual": trunk["action_visual_logits"],
                "action_reason": trunk["action_reason_logits"],
            },
        }
        return out
