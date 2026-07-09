from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .acpr_dino_field import ACPRDinoFieldExtractor
from .tfc_action_head import TFCActionHead
from .tfc_calalign_head import TFCCalAlignHead
from .tfc_deletion_contrast import TFCDeletionContrast
from .tfc_dual_lane_adapter import TFCDualLaneAdapter
from .tfc_factor_bank import TFCFactorBank
from .tfc_prototype_bank import TFCPrototypeBank
from .tfc_pu_state import TFCPUStateBuilder
from .tfc_reason_head import TFCReasonHead
from .tfc_target_credit import TFCTargetCredit
from .tfc_topk_factor_measurement import TFCTopKFactorMeasurement


def _zero_deletion_stats(logits: torch.Tensor) -> dict:
    z = torch.zeros_like(logits)
    return {
        "selected_effect": z,
        "random_effect": z,
        "selected_vs_random_gap": z,
        "selected_gt_random_mask": z.bool(),
        "selected_gt_random_rate": z.mean(),
        "deletion_contrast_loss": z.mean(),
        "stats": {"selected_vs_random_gap_mean": 0.0, "selected_gt_random_rate": 0.0, "valid_pairs": 0},
    }


class ACPRTFCModel(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        factor_bank_path: str = "configs/acpr_tfc_factors.yaml",
        factor_topk_tokens: int = 64,
        num_factor_prototypes: int = 4,
        max_deletion_factors_per_sample: int = 4,
        same_region_background: str = "ema",
        use_mock_dino: bool = False,
        action_delta_max: float = 0.06,
        reason_delta_max: float = 0.15,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.dino = ACPRDinoFieldExtractor(selected_layers=selected_layers, pretrained_weights=pretrained_weights, use_mock_dino=use_mock_dino)
        self.lane_adapter = TFCDualLaneAdapter(dim=dim)
        self.factor_bank = TFCFactorBank.from_yaml(factor_bank_path, action_dim=action_dim, reason_dim=reason_dim)
        self.prototype_bank = TFCPrototypeBank(self.factor_bank.num_factors, dim=dim, num_prototypes=num_factor_prototypes)
        self.measure_action = TFCTopKFactorMeasurement(dim=dim, topk=factor_topk_tokens)
        self.measure_reason = TFCTopKFactorMeasurement(dim=dim, topk=factor_topk_tokens)
        self.target_credit = TFCTargetCredit(self.factor_bank.num_factors, action_dim=action_dim, reason_dim=reason_dim, dim=dim)
        self.action_head = TFCActionHead(dim=dim, action_dim=action_dim, max_delta=action_delta_max)
        self.reason_head = TFCReasonHead(dim=dim, reason_dim=reason_dim, max_delta=reason_delta_max)
        self.pu_state = TFCPUStateBuilder()
        self.calalign = TFCCalAlignHead(action_dim=action_dim, reason_dim=reason_dim)
        self.deletion_action = TFCDeletionContrast()
        self.deletion_reason = TFCDeletionContrast()
        self.deletion = self.deletion_action
        self.max_deletion_factors_per_sample = int(max_deletion_factors_per_sample)
        self.same_region_background = str(same_region_background)

    def deletion_max_factors_for_epoch(self, epoch: int) -> int:
        if epoch <= 5:
            return max(1, min(2, self.max_deletion_factors_per_sample))
        return self.max_deletion_factors_per_sample

    def forward(
        self,
        images: torch.Tensor,
        action_targets: torch.Tensor | None = None,
        reason_targets: torch.Tensor | None = None,
        epoch: int = 0,
        split: str = "train",
        run_deletion: bool = False,
    ) -> dict:
        if split == "test":
            reason_targets_for_pu = None
        else:
            reason_targets_for_pu = reason_targets
        field = self.dino(images)
        patch = field["patch_tokens_by_layer"]
        lanes = self.lane_adapter(patch)
        proto = self.prototype_bank()
        spatial_names = [s.region_prior for s in self.factor_bank.specs]
        factor_names = self.factor_bank.names
        meas_action = self.measure_action(lanes["patch_action"], proto["factor_queries"], spatial_names, factor_names=factor_names)
        meas_reason = self.measure_reason(lanes["patch_reason"], proto["factor_queries"], spatial_names, factor_names=factor_names)
        action_visual = self.action_head.visual_logits_from_patch(lanes["patch_action"])
        reason_visual = self.reason_head.visual_logits_from_patch(lanes["patch_reason"])
        compat = self.factor_bank.compatibility_matrices()
        credit_action = self.target_credit(
            meas_action["factor_probs"],
            meas_action["factor_rho"],
            meas_action["factor_features"],
            compat,
            action_margins=action_visual,
        )
        credit_reason = self.target_credit(
            meas_reason["factor_probs"],
            meas_reason["factor_rho"],
            meas_reason["factor_features"],
            compat,
            reason_margins=reason_visual,
        )
        deletion_stats_action = None
        deletion_stats_reason = None
        if run_deletion:
            max_deletion_factors = self.deletion_max_factors_for_epoch(epoch)
            deletion_stats_action = self.deletion_action(
                lanes["patch_action"],
                meas_action["topk_indices"],
                credit_action["credit_action_norm"],
                self.action_head.visual_logits_from_patch,
                action_visual,
                action_targets,
                max_factors_per_sample=max_deletion_factors,
                same_region_background=self.same_region_background,
                random_indices=meas_action.get("random_indices"),
            )
            deletion_stats_reason = self.deletion_reason(
                lanes["patch_reason"],
                meas_reason["topk_indices"],
                credit_reason["credit_reason_norm"],
                self.reason_head.visual_logits_from_patch,
                reason_visual,
                reason_targets,
                max_factors_per_sample=max_deletion_factors,
                same_region_background=self.same_region_background,
                random_indices=meas_reason.get("random_indices"),
            )
        if deletion_stats_action is None:
            deletion_stats_action = _zero_deletion_stats(action_visual)
        if deletion_stats_reason is None:
            deletion_stats_reason = _zero_deletion_stats(reason_visual)
        pu_state = self.pu_state(
            reason_targets_for_pu,
            credit_reason["credit_reason"],
            meas_reason["factor_probs"],
            meas_reason["factor_rho"],
            epoch,
            deletion_gate_reason=deletion_stats_reason["selected_gt_random_mask"],
        )
        action = self.action_head(
            lanes["patch_action"],
            meas_action["factor_features"],
            credit_action["credit_action_norm"],
            credit_action["credit_confidence_action"],
            deletion_stats_action,
            epoch,
            action_targets=action_targets if split != "test" else None,
        )
        reason = self.reason_head(
            lanes["patch_reason"],
            meas_reason["factor_features"],
            credit_reason["credit_reason_norm"],
            credit_reason["credit_confidence_reason"],
            pu_state,
            epoch,
            deletion_stats=deletion_stats_reason,
        )
        cal = self.calalign(
            action["action_logits"],
            reason["reason_logits"],
            credit_action["credit_confidence_action"],
            credit_reason["credit_confidence_reason"],
            action_margins=action["action_logits"].detach(),
            reason_support=pu_state["support_credit"],
            reason_contra=pu_state["contra_credit"],
            reason_rho=pu_state["rho_reason"],
        )
        out = {
            **field,
            "patch_action": lanes["patch_action"],
            "patch_reason": lanes["patch_reason"],
            "action_visual_logits": action["action_visual_logits"],
            "action_tfc_delta": action["action_tfc_delta"],
            "action_rank_safety_mask": action["action_rank_safety_mask"],
            "action_deploy_safety_mask": action["action_deploy_safety_mask"],
            "action_logits_base": action["action_logits"],
            "action_logits_deploy": cal["action_logits_deploy"],
            "reason_visual_logits": reason["reason_visual_logits"],
            "reason_tfc_delta": reason["reason_tfc_delta"],
            "reason_logits_base": reason["reason_logits"],
            "reason_logits_deploy": cal["reason_logits_deploy"],
            "factor_probs_action": meas_action["factor_probs"],
            "factor_rho_action": meas_action["factor_rho"],
            "factor_probs_reason": meas_reason["factor_probs"],
            "factor_rho_reason": meas_reason["factor_rho"],
            "factor_features_action": meas_action["factor_features"],
            "factor_features_reason": meas_reason["factor_features"],
            "factor_prototypes": proto["prototypes"],
            "factor_queries": proto["factor_queries"],
            "native_similarity": self.factor_bank.native_similarity.to(images.device),
            "factor_conflict": self.factor_bank.factor_conflict.to(images.device),
            "compatibility": {key: value.to(images.device) for key, value in compat.items()},
            "credit_action": credit_action["credit_action"],
            "credit_reason": credit_reason["credit_reason"],
            "credit_action_norm": credit_action["credit_action_norm"],
            "credit_reason_norm": credit_reason["credit_reason_norm"],
            "credit_confidence_action": credit_action["credit_confidence_action"],
            "credit_confidence_reason": credit_reason["credit_confidence_reason"],
            "action_theta": cal["action_theta"],
            "reason_theta": cal["reason_theta"],
            "theta_delta_action": cal["theta_delta_action"],
            "theta_delta_reason": cal["theta_delta_reason"],
            "logits_deploy": cal["logits_deploy"],
            "pu_state": pu_state,
            "deletion_stats": deletion_stats_action,
            "deletion_stats_action": deletion_stats_action,
            "deletion_stats_reason": deletion_stats_reason,
            "artifact_stats": {
                "factor_support_mean_action": meas_action["factor_probs"].mean().detach(),
                "factor_support_mean_reason": meas_reason["factor_probs"].mean().detach(),
                "selected_vs_random_gap_mean": torch.as_tensor(deletion_stats_action["stats"]["selected_vs_random_gap_mean"], device=images.device),
                "selected_vs_random_gap_mean_reason": torch.as_tensor(deletion_stats_reason["stats"]["selected_vs_random_gap_mean"], device=images.device),
                "deletion_max_factors_used": torch.as_tensor(
                    self.deletion_max_factors_for_epoch(epoch) if run_deletion else 0,
                    device=images.device,
                ),
                "same_region_background_is_ema": torch.as_tensor(
                    1.0 if self.same_region_background.lower() == "ema" else 0.0,
                    device=images.device,
                ),
                "traffic_control_upper_region_mass_action": meas_action["grounding_audit_stats"]["traffic_control_upper_region_mass"],
                "obstacle_front_center_mass_action": meas_action["grounding_audit_stats"]["obstacle_front_center_mass"],
                "left_lane_corridor_mass_action": meas_action["grounding_audit_stats"]["left_lane_corridor_mass"],
                "right_lane_corridor_mass_action": meas_action["grounding_audit_stats"]["right_lane_corridor_mass"],
                "traffic_control_upper_region_mass_reason": meas_reason["grounding_audit_stats"]["traffic_control_upper_region_mass"],
                "obstacle_front_center_mass_reason": meas_reason["grounding_audit_stats"]["obstacle_front_center_mass"],
                "left_lane_corridor_mass_reason": meas_reason["grounding_audit_stats"]["left_lane_corridor_mass"],
                "right_lane_corridor_mass_reason": meas_reason["grounding_audit_stats"]["right_lane_corridor_mass"],
            },
            "topk_indices_action": meas_action["topk_indices"],
            "topk_indices_reason": meas_reason["topk_indices"],
            "factor_attention_entropy_action": meas_action["attention_entropy"],
            "factor_attention_entropy_reason": meas_reason["attention_entropy"],
        }
        return out
