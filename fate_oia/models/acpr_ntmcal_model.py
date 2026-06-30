from __future__ import annotations

import torch
from torch import nn

from .acpr_dino_field import ACPRDinoFieldExtractor
from .acpr_ego_regions import ACPREgoRegionEncoder
from .acpr_label_trunk import ACPRLabelTrunk
from .acpr_ntmcal_action_predicate_head import NativeTextActionPredicateHead
from .acpr_ntmcal_observation_builder import NativeTextObservationBuilder
from .acpr_ntmcal_pair_memory import NativeTextReasonPairMemory
from .acpr_ntmcal_predicate_bank import NativePredicateBank
from .acpr_ntmcal_pu_state import NativeTextPUReasonState
from .acpr_ntmcal_reason_residual import NativeTextReasonResidual
from .acpr_ntmcal_text_atoms import NativeTextAtomEncoder
from .acpr_ntmcal_threshold_head import NativeTextMetricCalibrator
from .acpr_ntmcal_topk_predicate_measurement import NativeTextTopKPredicateMeasurement


class ACPRNTMCalModel(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        predicate_config: str = "configs/acpr_ntmcal_native_text_predicates.yaml",
        reason_formula_config: str = "configs/acpr_ntmcal_reason_formulas.yaml",
        use_mock_dino: bool = False,
        predicate_topk: int = 64,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.dino = ACPRDinoFieldExtractor(selected_layers=selected_layers, pretrained_weights=pretrained_weights, use_mock_dino=use_mock_dino)
        self.ego = ACPREgoRegionEncoder(grid_hw=(45, 80), dim=dim)
        self.trunk = ACPRLabelTrunk(dim=dim, action_dim=action_dim, reason_dim=reason_dim)
        self.predicate_bank = NativePredicateBank.from_yaml(predicate_config)
        self.atom_encoder = NativeTextAtomEncoder(self.predicate_bank.atom_vocab, dim=dim)
        self.predicate_measurement = NativeTextTopKPredicateMeasurement(
            self.predicate_bank, self.atom_encoder, dim=dim, selected_layers=selected_layers, topk=predicate_topk
        )
        self.observation_builder = NativeTextObservationBuilder(self.predicate_bank, reason_formula_config)
        self.pu_builder = NativeTextPUReasonState(self.observation_builder.support, self.observation_builder.contra)
        self.reason_residual = NativeTextReasonResidual(dim=dim, reason_dim=reason_dim)
        self.action_predicate_head = NativeTextActionPredicateHead(len(self.predicate_bank.specs), dim=dim, action_dim=action_dim)
        self.ntmcal_threshold = NativeTextMetricCalibrator(action_dim=action_dim, reason_dim=reason_dim)
        self.pair_memory = NativeTextReasonPairMemory(reason_dim=reason_dim, dim=dim)

    def forward(
        self,
        images: torch.Tensor,
        *,
        epoch: int = 0,
        split: str = "train",
        reason_labels: torch.Tensor | None = None,
        file_names=None,
        structured_records=None,
        force_zero_reason_delta: bool = False,
    ) -> dict:
        field = self.dino(images)
        patch = field["patch_tokens_by_layer"]
        patch0, ego_features, region_masks, ego_stats = self.ego(patch[:, 0])
        patch = patch.clone()
        patch[:, 0] = patch0
        pred = self.predicate_measurement(patch, region_masks=region_masks)
        trunk = self.trunk(patch, predicate_tokens=pred["predicate_tokens"])
        base_action_logits = trunk["action_logits_direct"]
        base_reason_logits = trunk["reason_logits_visual"]

        train_observation = self.training and split == "train"
        obs = self.observation_builder(
            reason_labels if train_observation else None,
            file_names=file_names if train_observation else None,
            structured_records=structured_records if train_observation else None,
            split="train" if train_observation else "test",
            batch_size=images.shape[0],
            device=images.device,
        )
        pu = self.pu_builder(reason_labels if train_observation else None, pred["predicate_q"], pred["predicate_rho"], epoch)
        reason_res = self.reason_residual(
            base_reason_logits,
            trunk["label_nodes"][:, self.action_dim :],
            pu["support_score"],
            pu["contra_score"],
            pu["reason_rho"],
            epoch=epoch,
        )
        reason_delta = torch.zeros_like(reason_res["reason_delta"]) if force_zero_reason_delta else reason_res["reason_delta"]
        reason_logits = base_reason_logits + reason_delta
        action_pred = self.action_predicate_head(base_action_logits, pred["predicate_q"], pred["predicate_rho"], pred["predicate_tokens"], epoch=epoch)
        action_logits = base_action_logits + action_pred["action_predicate_delta"]
        cal = self.ntmcal_threshold(
            action_logits,
            reason_logits,
            pu["support_score"],
            pu["contra_score"],
            pu["reason_rho"],
            base_reason_logits,
            pred["predicate_q"],
            pred["predicate_rho"],
            epoch=epoch,
        )
        out = {**field, **trunk, **pred, **reason_res, **action_pred, **cal}
        out.update(
            {
                "action_logits_base": base_action_logits,
                "reason_logits_base": base_reason_logits,
                "action_logits_ntmcal": action_logits,
                "reason_logits_ntmcal": reason_logits,
                "native_text_atoms": self.predicate_bank.audit(),
                "native_text_observations": obs,
                "pu_state": pu,
                "support_score": pu["support_score"],
                "contra_score": pu["contra_score"],
                "reason_reliability": pu["reason_reliability"],
                "ntmcal_stats": {
                    "ego": ego_stats,
                    "predicate": pred["predicate_stats"],
                    "observation": obs["source_stats"],
                    "pu": pu["stats"],
                    "reason_delta": reason_res["reason_delta_stats"],
                    "action_predicate": action_pred["action_predicate_stats"],
                    "threshold": cal["threshold_stats"],
                },
                "branch_logits": {
                    "base_fixed": torch.cat([base_action_logits, base_reason_logits], -1),
                    "deploy_fixed": cal["logits_deploy"],
                },
            }
        )
        return out
