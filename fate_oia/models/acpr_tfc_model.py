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
        self.deletion = TFCDeletionContrast()

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
        meas_action = self.measure_action(lanes["patch_action"], proto["factor_queries"], spatial_names)
        meas_reason = self.measure_reason(lanes["patch_reason"], proto["factor_queries"], spatial_names)
        compat = self.factor_bank.compatibility_matrices()
        credit_action = self.target_credit(
            meas_action["factor_probs"],
            meas_action["factor_rho"],
            meas_action["factor_features"],
            compat,
        )
        credit_reason = self.target_credit(
            meas_reason["factor_probs"],
            meas_reason["factor_rho"],
            meas_reason["factor_features"],
            compat,
        )
        pu_state = self.pu_state(
            reason_targets_for_pu,
            credit_reason["credit_reason"],
            meas_reason["factor_probs"],
            meas_reason["factor_rho"],
            epoch,
        )
        action_visual = self.action_head.visual_logits_from_patch(lanes["patch_action"])
        deletion_stats = None
        if run_deletion:
            deletion_stats = self.deletion(
                lanes["patch_action"],
                meas_action["topk_indices"],
                credit_action["credit_action_norm"],
                self.action_head.visual_logits_from_patch,
                action_visual,
                action_targets,
                random_indices=meas_action.get("random_indices"),
            )
        action = self.action_head(
            lanes["patch_action"],
            meas_action["factor_features"],
            credit_action["credit_action_norm"],
            credit_action["credit_confidence_action"],
            deletion_stats,
            epoch,
        )
        reason = self.reason_head(
            lanes["patch_reason"],
            meas_reason["factor_features"],
            credit_reason["credit_reason_norm"],
            credit_reason["credit_confidence_reason"],
            pu_state,
            epoch,
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
        if deletion_stats is None:
            z = torch.zeros_like(action["action_logits"])
            deletion_stats = {
                "selected_effect": z,
                "random_effect": z,
                "selected_vs_random_gap": z,
                "selected_gt_random_mask": z.bool(),
                "selected_gt_random_rate": z.mean(),
                "deletion_contrast_loss": z.mean(),
                "stats": {"selected_vs_random_gap_mean": 0.0, "selected_gt_random_rate": 0.0, "valid_pairs": 0},
            }
        out = {
            **field,
            "patch_action": lanes["patch_action"],
            "patch_reason": lanes["patch_reason"],
            "action_visual_logits": action["action_visual_logits"],
            "action_tfc_delta": action["action_tfc_delta"],
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
            "deletion_stats": deletion_stats,
            "artifact_stats": {
                "factor_support_mean_action": meas_action["factor_probs"].mean().detach(),
                "factor_support_mean_reason": meas_reason["factor_probs"].mean().detach(),
                "selected_vs_random_gap_mean": torch.as_tensor(deletion_stats["stats"]["selected_vs_random_gap_mean"], device=images.device),
            },
            "topk_indices_action": meas_action["topk_indices"],
            "topk_indices_reason": meas_reason["topk_indices"],
            "factor_attention_entropy_action": meas_action["attention_entropy"],
            "factor_attention_entropy_reason": meas_reason["attention_entropy"],
        }
        return out
