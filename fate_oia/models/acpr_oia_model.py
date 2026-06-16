from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .acpr_action_combo_aux import ACPRActionComboAux
from .acpr_action_candidates import ACPRActionCandidates
from .acpr_action_predicate_delta import ACPRActionPredicateDelta
from .acpr_action_utility import ACPRActionUtility
from .acpr_calibration import ACPRCalibrationHead
from .acpr_dino_field import ACPRDinoFieldExtractor
from .acpr_ego_regions import ACPREgoRegionEncoder
from .acpr_label_trunk import ACPRLabelTrunk
from .acpr_pair_memory import ACPRPairMemory
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
        actalign_enabled: bool = False,
        actalign_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.threshold_enabled = bool(threshold_enabled)
        self.actalign_enabled = bool(actalign_enabled)
        actalign_kwargs = actalign_kwargs or {}
        self.actalign_mode = str(actalign_kwargs.get("mode", actalign_kwargs.get("stage_mode", "residual")))
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
        self.action_predicate_delta = ACPRActionPredicateDelta(
            dim=dim,
            num_predicates=self.predicate_head.num_predicates,
            action_dim=action_dim,
            hidden_dim=int(actalign_kwargs.get("predicate_delta_hidden", 192)),
            max_delta=float(actalign_kwargs.get("max_pred_delta", 0.05)),
            detach_inputs=bool(actalign_kwargs.get("detach_predicate_delta_inputs", True)),
        )
        self.action_utility = ACPRActionUtility(
            action_dim=action_dim,
            max_r2a_delta=float(actalign_kwargs.get("max_r2a_delta", 0.20)),
            max_pred_delta=float(actalign_kwargs.get("max_pred_delta", 0.05)),
            initial_r2a_gate=float(actalign_kwargs.get("initial_r2a_gate", 0.0)),
            initial_pred_gate=float(actalign_kwargs.get("initial_pred_gate", 0.0)),
        )
        self.action_candidates = ACPRActionCandidates(
            action_dim=action_dim,
            max_pred_delta=float(actalign_kwargs.get("max_pred_delta", 0.05)),
            initial_blend_gamma=float(actalign_kwargs.get("initial_blend_gamma", 0.5)),
        )

    def forward(self, images: torch.Tensor, epoch: int = 0) -> dict[str, torch.Tensor | dict | tuple[int, int] | int]:
        field = self.dino(images)
        patch = field["patch_tokens_by_layer"]
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
        action_logits_fallback = action_logits_final_raw
        if self.actalign_enabled:
            pred_delta = self.action_predicate_delta(
                action_nodes=trunk["label_nodes"][:, : self.action_dim],
                predicate_probs=predicates["predicate_probs"],
            )
            theta_action = thresholded["threshold_logit"][: self.action_dim] if self.threshold_enabled else torch.zeros(self.action_dim, device=images.device, dtype=action_logits_fallback.dtype)
            candidate_out = self.action_candidates(
                action_logits_fallback=action_logits_fallback,
                action_visual_logits=trunk["action_visual_logits"],
                action_reason_logits=trunk["action_reason_logits"],
                theta_action=theta_action,
                predicate_action_delta=pred_delta["predicate_action_delta"],
                probe_mode=self.actalign_mode == "candidate_probe",
            )
            if self.actalign_mode in {"candidate_probe", "candidate_finetune"}:
                utility = {
                    "action_logits_utility": candidate_out["utility_final"],
                    "action_r2a_delta": candidate_out["reason"] - action_logits_fallback,
                    "action_predicate_delta": candidate_out["predicate_delta_clipped"],
                    "r2a_gate": candidate_out["selected_gate"],
                    "pred_gate": candidate_out["selected_gate"],
                    "r2a_delta_abs_mean": (candidate_out["reason"] - action_logits_fallback).abs().mean(),
                    "pred_delta_abs_mean": candidate_out["predicate_delta_clipped"].abs().mean(),
                    "pred_delta_max_abs": candidate_out["predicate_delta_clipped"].abs().max(),
                    "pred_delta_per_action_mean": candidate_out["predicate_delta_clipped"].mean(0),
                }
            else:
                utility = self.action_utility(
                    action_logits_fallback=action_logits_fallback,
                    action_visual_logits=trunk["action_visual_logits"],
                    action_reason_logits=trunk["action_reason_logits"],
                    predicate_action_delta=pred_delta["predicate_action_delta"],
                )
            action_logits_final_raw = utility["action_logits_utility"]
            logits_final_raw = torch.cat([action_logits_final_raw, reason_logits_final_raw], dim=-1)
        else:
            zero_delta = torch.zeros_like(action_logits_fallback)
            pred_delta = {
                "predicate_action_delta_raw": zero_delta,
                "predicate_action_delta": zero_delta,
                "predicate_action_delta_abs_mean": zero_delta.abs().mean(),
                "predicate_action_delta_max_abs": zero_delta.abs().max(),
                "predicate_action_delta_per_action_mean": zero_delta.mean(0),
            }
            utility = self.action_utility(
                action_logits_fallback=action_logits_fallback,
                action_visual_logits=trunk["action_visual_logits"],
                action_reason_logits=trunk["action_reason_logits"],
                predicate_action_delta=zero_delta,
                override_r2a_gate=torch.zeros_like(self.action_utility.r2a_gate),
                override_pred_gate=torch.zeros_like(self.action_utility.pred_gate),
            )
            theta_action = thresholded["threshold_logit"][: self.action_dim] if self.threshold_enabled else torch.zeros(self.action_dim, device=images.device, dtype=action_logits_fallback.dtype)
            candidate_out = self.action_candidates(
                action_logits_fallback=action_logits_fallback,
                action_visual_logits=trunk["action_visual_logits"],
                action_reason_logits=trunk["action_reason_logits"],
                theta_action=theta_action,
                predicate_action_delta=zero_delta,
            )

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
            "action_logits_fallback": action_logits_fallback,
            "action_logits_utility": utility["action_logits_utility"],
            "action_logits_actalign": action_logits_final_raw,
            "action_r2a_delta": utility["action_r2a_delta"],
            "action_predicate_delta": utility["action_predicate_delta"],
            "predicate_action_delta_raw": pred_delta["predicate_action_delta_raw"],
            "predicate_action_delta_clipped": candidate_out["predicate_delta_clipped"],
            "r2a_gate": utility["r2a_gate"],
            "pred_gate": utility["pred_gate"],
            "r2a_delta_abs_mean": utility["r2a_delta_abs_mean"],
            "pred_delta_abs_mean": utility["pred_delta_abs_mean"],
            "pred_delta_max_abs": utility.get("pred_delta_max_abs", pred_delta.get("predicate_action_delta_max_abs")),
            "pred_delta_per_action_mean": utility.get("pred_delta_per_action_mean", pred_delta.get("predicate_action_delta_per_action_mean")),
            "action_utility_enabled": torch.tensor(float(self.actalign_enabled), device=images.device),
            "action_utility_stats": {
                "mode": self.actalign_mode,
                "r2a_gate": utility["r2a_gate"],
                "pred_gate": utility["pred_gate"],
                "r2a_delta_abs_mean": utility["r2a_delta_abs_mean"],
                "pred_delta_abs_mean": utility["pred_delta_abs_mean"],
                "pred_delta_max_abs": utility.get("pred_delta_max_abs", pred_delta.get("predicate_action_delta_max_abs")),
                "pred_delta_per_action_mean": utility.get("pred_delta_per_action_mean", pred_delta.get("predicate_action_delta_per_action_mean")),
                "candidate_selected_id": candidate_out["selected_candidate_id"],
                "candidate_selected_gate": candidate_out["selected_gate"],
                "candidate_blend_gamma": candidate_out["blend_gamma"],
            },
            "action_candidate_logits": {k: candidate_out[k] for k in ["fallback", "visual", "reason", "blend", "predicate", "blend_predicate"]},
            "action_candidate_names": candidate_out["candidate_names"],
            "action_candidate_selected_id": candidate_out["selected_candidate_id"],
            "action_candidate_selected_gate": candidate_out["selected_gate"],
            "action_candidate_blend_gamma": candidate_out["blend_gamma"],
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
                "base_fixed": logits_base,
                "fallback": torch.cat([action_logits_fallback, reason_logits_final_raw], dim=-1),
                "utility": torch.cat([action_logits_final_raw, reason_logits_final_raw], dim=-1),
                "candidate_visual": torch.cat([candidate_out["visual"], reason_logits_final_raw], dim=-1),
                "candidate_reason": torch.cat([candidate_out["reason"], reason_logits_final_raw], dim=-1),
                "candidate_blend": torch.cat([candidate_out["blend"], reason_logits_final_raw], dim=-1),
                "candidate_predicate": torch.cat([candidate_out["predicate"], reason_logits_final_raw], dim=-1),
                "candidate_blend_predicate": torch.cat([candidate_out["blend_predicate"], reason_logits_final_raw], dim=-1),
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
