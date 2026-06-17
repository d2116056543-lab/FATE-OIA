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
from .acpr_predicate_conditioned_threshold import ACPRPredicateConditionedThreshold
from .acpr_predicate_reason import ACPRPredicateReasoner
from .acpr_scene_predicate_head import ACPRScenePredicateHead
from .acpr_threshold_head import ACPRThresholdHead
from .acpr_triadic_mediator import ACPRTriadicMediator


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
        pmt_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.threshold_enabled = bool(threshold_enabled)
        self.pmt_cfg = dict(pmt_kwargs or {})
        self.pmt_enabled = bool(self.pmt_cfg.get("enabled", False))
        self.dino = ACPRDinoFieldExtractor(
            selected_layers=selected_layers,
            pretrained_weights=pretrained_weights,
            use_mock_dino=use_mock_dino,
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
        self.pair_memory = ACPRPairMemory(dim=dim)
        self.reason_pair_proj = nn.Linear(dim, dim)
        self.action_combo_aux = ACPRActionComboAux(dim=dim, action_dim=action_dim)
        self.calibration = ACPRCalibrationHead(num_labels=action_dim + reason_dim)
        self.triadic_mediator = ACPRTriadicMediator(
            action_dim=action_dim,
            reason_dim=reason_dim,
            num_predicates=self.predicate_head.num_predicates,
            max_action_delta=float(self.pmt_cfg.get("triadic_delta_max", 0.05)),
            grammar_path=grammar_path,
            action_predicate_grammar_path=str(self.pmt_cfg.get("action_predicate_grammar", "configs/acpr_pmt_action_predicate_grammar.yaml")),
        )
        threshold_kwargs = dict(threshold_kwargs or {})
        if self.pmt_enabled and bool(self.pmt_cfg.get("predicate_conditioned_threshold", True)):
            self.threshold_head = ACPRPredicateConditionedThreshold(
                action_dim=action_dim,
                reason_dim=reason_dim,
                num_predicates=self.predicate_head.num_predicates,
                threshold_delta_max=float(self.pmt_cfg.get("threshold_delta_max", 0.10)),
                **threshold_kwargs,
            )
        else:
            self.threshold_head = ACPRThresholdHead(action_dim=action_dim, reason_dim=reason_dim, **threshold_kwargs)

    def _threshold_forward(
        self,
        action_logits_base: torch.Tensor,
        reason_logits_base: torch.Tensor,
        predicate_probs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if isinstance(self.threshold_head, ACPRPredicateConditionedThreshold):
            return self.threshold_head(action_logits_base, reason_logits_base, predicate_probs)
        return self.threshold_head(action_logits_base, reason_logits_base)

    def forward(
        self,
        images: torch.Tensor,
        epoch: int = 0,
        predicate_patch_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict | tuple[int, int] | int]:
        field = self.dino(images)
        patch = field["patch_tokens_by_layer"]
        patch0, ego_features, region_masks, ego_stats = self.ego(patch[:, 0])
        patch = patch.clone()
        patch[:, 0] = patch0
        predicates = self.predicate_head(
            patch,
            region_masks=region_masks,
            predicate_patch_targets=predicate_patch_targets,
        )
        trunk = self.trunk(patch, predicate_tokens=predicates["predicate_tokens"])
        reason_delta = self.predicate_reason(
            trunk["label_nodes"][:, self.action_dim :],
            predicates["predicate_probs"],
            predicates["predicate_tokens"],
        )
        action_reason_logits = trunk["action_reason_logits"]
        triadic: dict[str, torch.Tensor | dict]
        if self.pmt_enabled:
            triadic = self.triadic_mediator(
                action_visual_logits=trunk["action_visual_logits"],
                action_reason_logits=trunk["action_reason_logits"],
                reason_logits=trunk["reason_logits_visual"] + reason_delta["predicate_reason_delta"],
                predicate_probs=predicates["predicate_probs"],
                predicate_tokens=predicates["predicate_tokens"],
            )
            action_reason_logits = triadic["action_reason_logits_triadic"]
        else:
            zero_action = trunk["action_reason_logits"].new_zeros(trunk["action_reason_logits"].shape)
            zero_reason = trunk["reason_logits_visual"].new_zeros(trunk["reason_logits_visual"].shape)
            zero_chain = trunk["action_reason_logits"].new_zeros(trunk["action_reason_logits"].shape[0], self.action_dim, self.reason_dim, self.predicate_head.num_predicates)
            triadic = {
                "action_reason_logits_triadic": trunk["action_reason_logits"],
                "triadic_action_delta": zero_action,
                "triadic_reason_support": zero_reason,
                "triadic_predicate_support": predicates["predicate_probs"].new_zeros(predicates["predicate_probs"].shape),
                "triadic_chain_score": zero_chain,
                "triadic_top_chain_indices": zero_chain.flatten(1).topk(1, dim=1).indices,
                "triadic_stats": {"enabled": False, "delta_abs_mean": 0.0},
            }
        gate = trunk["action_fusion_gate"].clamp(0.0, 1.0)
        action_logits_base = gate * trunk["action_visual_logits"] + (1.0 - gate) * action_reason_logits
        reason_logits_base = trunk["reason_logits_visual"] + reason_delta["predicate_reason_delta"]
        logits_base = torch.cat([action_logits_base, reason_logits_base], dim=-1)
        thresholded = self._threshold_forward(action_logits_base, reason_logits_base, predicates["predicate_probs"])
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
            **triadic,
            "global_embedding": field["cls_tokens_by_layer"].mean(1),
            "pair_embedding": pair_embedding,
            "reason_embeddings_for_pair": reason_embeddings_for_pair,
            "action_reason_logits_pmt": action_reason_logits,
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
            "threshold_delta": thresholded.get("threshold_delta", torch.zeros_like(logits_base)),
            "action_threshold_delta": thresholded.get("action_threshold_delta", torch.zeros_like(action_logits_base)),
            "reason_threshold_delta": thresholded.get("reason_threshold_delta", torch.zeros_like(reason_logits_base)),
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
            "pmt_enabled": self.pmt_enabled,
            "branch_logits": {
                "direct": logits_base,
                "direct_plus_predicate": logits_base,
                "direct_plus_triadic": torch.cat([action_logits_base, reason_logits_base], dim=-1),
                "base_fixed": logits_base,
                "deploy_fixed": logits_final_raw,
                "deploy_fixed_pmt": logits_final_raw,
                "raw": logits_final_raw,
                "calibrated": logits_final_calibrated,
                "final_raw": logits_final_raw,
                "final_calibrated": logits_final_calibrated,
                "action_visual": trunk["action_visual_logits"],
                "action_reason": trunk["action_reason_logits"],
                "action_reason_pmt": action_reason_logits,
            },
        }
        return out
