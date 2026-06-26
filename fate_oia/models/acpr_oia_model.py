from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .acpr_action_combo_aux import ACPRActionComboAux
from .acpr_calibration import ACPRCalibrationHead
from .acpr_dino_field import ACPRDinoFieldExtractor
from .acpr_ego_regions import ACPREgoRegionEncoder
from .acpr_grounded_evidence_memory import ACPREvidenceGroundingLoss, ACPRGroundedEvidencePooler
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
        gem_enabled: bool = False,
        gem_kwargs: dict | None = None,
        pair_memory_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.threshold_enabled = bool(threshold_enabled)
        self.gem_enabled = bool(gem_enabled)
        self.dino = ACPRDinoFieldExtractor(selected_layers=selected_layers, pretrained_weights=pretrained_weights, use_mock_dino=use_mock_dino)
        self.ego = ACPREgoRegionEncoder(grid_hw=(45, 80), dim=dim)
        self.predicate_head = ACPRScenePredicateHead(scene_config=scene_config, dim=dim, num_layers=len(selected_layers))
        self.trunk = ACPRLabelTrunk(dim=dim, action_dim=action_dim, reason_dim=reason_dim)
        self.predicate_reason = ACPRPredicateReasoner(dim=dim, reason_dim=reason_dim, num_predicates=self.predicate_head.num_predicates, predicate_names=self.predicate_head.names, grammar_path=grammar_path)
        pair_cfg = dict(pair_memory_kwargs or {})
        self.pair_memory = ACPRPairMemory(
            dim=dim,
            memory_size=int(pair_cfg.get("memory_size", 8192)),
            memory_device=str(pair_cfg.get("memory_device", pair_cfg.get("device", "cpu"))),
        )
        self.reason_pair_proj = nn.Linear(dim, dim)
        self.action_combo_aux = ACPRActionComboAux(dim=dim, action_dim=action_dim)
        self.calibration = ACPRCalibrationHead(num_labels=action_dim + reason_dim)
        self.threshold_head = ACPRThresholdHead(action_dim=action_dim, reason_dim=reason_dim, **(threshold_kwargs or {}))
        gem_cfg = dict(gem_kwargs or {})
        self.evidence_memory = ACPRGroundedEvidencePooler(
            dim=dim,
            slots_config=gem_cfg.get("slots_config", "configs/acpr_gem_evidence_slots.yaml"),
            topk=int(gem_cfg.get("topk", 256)),
        )
        self.evidence_grounding_loss_fn = ACPREvidenceGroundingLoss(entropy_weight=float(gem_cfg.get("entropy_weight", 0.001)))

    def forward(
        self,
        images: torch.Tensor,
        epoch: int = 0,
        gem_grounding: dict | None = None,
        oracle_evidence: bool = False,
    ) -> dict[str, torch.Tensor | dict | tuple[int, int] | int | list[str] | bool]:
        field = self.dino(images)
        patch_tokens_raw = field["patch_tokens_by_layer"]
        patch0, ego_features, region_masks, ego_stats = self.ego(patch_tokens_raw[:, 0])
        patch = patch_tokens_raw.clone()
        patch[:, 0] = patch0
        grounding_targets = None if gem_grounding is None else gem_grounding.get("grounding_targets")
        grounding_mask = None if gem_grounding is None else gem_grounding.get("grounding_mask")
        if self.gem_enabled:
            evidence = self.evidence_memory(patch, grounding_targets=grounding_targets, grounding_mask=grounding_mask)
            evidence_tokens = evidence["evidence_tokens"]
        else:
            b, _, n, d = patch.shape
            m = self.evidence_memory.num_slots
            evidence_tokens = None
            evidence = {
                "evidence_tokens": patch.new_zeros(b, m, d),
                "evidence_attention": patch.new_zeros(b, m, n),
                "evidence_scores": patch.new_zeros(b, m, n),
                "evidence_slot_names": self.evidence_memory.slot_names,
                "evidence_slot_groups": self.evidence_memory.slot_groups,
                "evidence_grounding_targets": patch.new_zeros(b, m, n),
                "evidence_grounding_mask": patch.new_zeros(b, m),
                "evidence_available_rate": patch.new_zeros(()),
                "evidence_stats": {},
                "evidence_oracle_mode": False,
            }
        predicates = self.predicate_head(patch, region_masks=region_masks, evidence_tokens=evidence_tokens)
        trunk = self.trunk(
            patch,
            predicate_tokens=predicates["predicate_tokens"],
            evidence_tokens=evidence_tokens,
            evidence_attention=evidence.get("evidence_attention"),
            evidence_enabled=self.gem_enabled,
        )
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
            "patch_tokens_by_layer_raw": patch_tokens_raw,
            **trunk,
            **predicates,
            **reason_delta,
            **action_set,
            **evidence,
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
            "evidence_branch_delta_norms": {
                "label": float(trunk["label_evidence_delta_norm"].detach().cpu()) if torch.is_tensor(trunk.get("label_evidence_delta_norm")) else 0.0,
                "predicate": float(predicates["predicate_evidence_delta_norm"].detach().cpu()) if torch.is_tensor(predicates.get("predicate_evidence_delta_norm")) else 0.0,
            },
            "branch_logits": {
                "direct": logits_base,
                "direct_plus_predicate": logits_base,
                "base_fixed": logits_base,
                "deploy_fixed": logits_final_raw,
                "raw": logits_final_raw,
                "calibrated": logits_final_calibrated,
                "final_raw": logits_final_raw,
                "final_calibrated": logits_final_calibrated,
                "evidence_base_fixed": logits_base,
                "no_evidence_base_fixed": logits_base.detach(),
                "action_visual": trunk["action_visual_logits"],
                "action_reason": trunk["action_reason_logits"],
            },
        }
        if grounding_targets is not None and grounding_mask is not None:
            out["evidence_grounding_loss"] = self.evidence_grounding_loss_fn(
                evidence["evidence_attention"],
                grounding_targets,
                grounding_mask,
                evidence.get("evidence_scores"),
            )
        else:
            out["evidence_grounding_loss"] = logits_base.sum() * 0.0
        out["evidence_grounding_stats"] = evidence.get("evidence_stats", {})
        return out
