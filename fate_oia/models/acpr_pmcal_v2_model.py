from __future__ import annotations

from typing import Any
import torch
from torch import nn

from .acpr_action_combo_aux import ACPRActionComboAux
from .acpr_dino_field import ACPRDinoFieldExtractor
from .acpr_ego_regions import ACPREgoRegionEncoder
from .acpr_pmcal_label_head import ACPRPMCalLabelHead
from .pmcal_action_predicate_head import PMCalActionPredicateHead
from .pmcal_predicate_measurement import PMCalPredicateMeasurementLayer
from .pmcal_predicate_observation_builder import PMCalPredicateObservationBuilder
from .pmcal_pu_calalign_head import PMCalPUCalAlignHead
from .pmcal_pu_reason_state import PMCalPUReasonState
from .pmcal_reason_formula_bank import PMCalReasonFormulaBank
from .pmcal_reason_formula_head import PMCalReasonFormulaHead


class ACPRPMCalV2Model(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        scene_config: str = "configs/acpr_scene_predicates.yaml",
        grammar_path: str = "configs/acpr_reason_predicate_grammar.yaml",
        text_prompt_config: str | None = None,
        use_mock_dino: bool = False,
        threshold_kwargs: dict | None = None,
        formula_residual_cap: float = 0.20,
        formula_gate_max: float = 0.35,
        action_predicate_cap: float = 0.06,
        action_predicate_gate_max: float = 0.35,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.dino = ACPRDinoFieldExtractor(selected_layers=selected_layers, pretrained_weights=pretrained_weights, use_mock_dino=use_mock_dino)
        self.ego = ACPREgoRegionEncoder(grid_hw=(45, 80), dim=dim)
        self.label_head = ACPRPMCalLabelHead(dim=dim, action_dim=action_dim, reason_dim=reason_dim)
        self.predicate_measurement = PMCalPredicateMeasurementLayer(scene_config=scene_config, text_prompt_config=text_prompt_config, dim=dim, num_predicates=32, num_layers=len(selected_layers))
        self.formula_bank = PMCalReasonFormulaBank(grammar_path, self.predicate_measurement.predicate_names)
        self.formula_head = PMCalReasonFormulaHead(self.formula_bank, cap=formula_residual_cap, gate_max=formula_gate_max)
        self.pu_builder = PMCalPUReasonState(self.formula_bank)
        self.action_head = PMCalActionPredicateHead(dim=dim, action_dim=action_dim, cap=action_predicate_cap, gate_max=action_predicate_gate_max)
        self.threshold_head = PMCalPUCalAlignHead(action_dim=action_dim, reason_dim=reason_dim, **(threshold_kwargs or {}))
        self.action_combo_aux = ACPRActionComboAux(dim=dim, action_dim=action_dim)
        self.observation_builder = PMCalPredicateObservationBuilder(scene_config=scene_config, grammar_path=grammar_path, text_prompt_config=text_prompt_config)

    def forward(
        self,
        images: torch.Tensor,
        *,
        epoch: int = 0,
        split: str = "train",
        reason_labels: torch.Tensor | None = None,
        action_labels: torch.Tensor | None = None,
        file_names: list[str] | None = None,
        structured_records: list[dict] | None = None,
        return_loss_inputs: bool = True,
        force_zero_reason_formula: bool = False,
    ) -> dict[str, Any]:
        field = self.dino(images)
        patch = field["patch_tokens_by_layer"]
        patch0, ego_features, region_masks, ego_stats = self.ego(patch[:, 0])
        patch = patch.clone()
        patch[:, 0] = patch0
        pred = self.predicate_measurement(patch, region_masks=region_masks)
        labels = self.label_head(patch, predicate_tokens=pred["predicate_tokens"])
        formula = self.formula_head(pred["q_pred"], pred["rho_pred"])
        formula_logits = torch.zeros_like(formula["reason_formula_logits"]) if force_zero_reason_formula else formula["reason_formula_logits"]
        reason_logits_visual = labels["reason_logits_visual"]
        reason_logits_final = reason_logits_visual + formula["reason_formula_gate"] * formula_logits
        act = self.action_head(labels["action_nodes"], pred["q_pred"], pred["rho_pred"], pred["predicate_tokens"])
        action_logits_final = act["action_logits_final"]
        thresholded = self.threshold_head(action_logits_final, reason_logits_final)
        file_names = file_names or ["" for _ in range(images.shape[0])]
        if split != "test" and reason_labels is not None:
            pu_state = self.pu_builder.build(reason_labels, pred["q_pred"], pred["rho_pred"])
            observations = self.observation_builder.build(
                file_names=file_names,
                reason_labels=reason_labels,
                structured_records=structured_records,
                split=split,
                device=images.device,
            )
        else:
            pu_state = {
                "positive_mask": torch.empty(0, device=images.device),
                "unknown_mask": torch.empty(0, device=images.device),
                "reliable_negative_mask": torch.empty(0, device=images.device),
                "reason_reliability": torch.empty(0, device=images.device),
                "support_score": formula["support_score"],
                "contra_score": formula["contra_score"],
                "pu_state_id": torch.empty(0, device=images.device, dtype=torch.long),
            }
            observations = self.observation_builder.build(
                file_names=file_names,
                reason_labels=None,
                structured_records=None,
                split="test",
                device=images.device,
            )
        action_set = self.action_combo_aux(labels["label_nodes"], action_logits_final)
        base = torch.cat([action_logits_final, reason_logits_final], dim=-1)
        out: dict[str, Any] = {
            **field,
            **labels,
            **pred,
            **formula,
            **act,
            **action_set,
            "ego_stats": ego_stats,
            "action_logits_base": action_logits_final,
            "reason_logits_base": reason_logits_final,
            "action_logits_deploy": thresholded["action_logits_deploy"],
            "reason_logits_deploy": thresholded["reason_logits_deploy"],
            "logits_base": base,
            "logits_deploy": thresholded["logits_deploy"],
            "threshold_logit": thresholded["threshold_logit"],
            "threshold_prob": thresholded["threshold_prob"],
            "action_threshold_prob": thresholded["action_threshold_prob"],
            "reason_threshold_prob": thresholded["reason_threshold_prob"],
            "action_logits_calibrated": thresholded["action_logits_calibrated"],
            "reason_logits_calibrated": thresholded["reason_logits_calibrated"],
            "reason_logits_final": reason_logits_final,
            "action_logits_final": action_logits_final,
            "pu_state": pu_state,
            "pu_positive_mask": pu_state["positive_mask"],
            "pu_unknown_mask": pu_state["unknown_mask"],
            "pu_reliable_negative_mask": pu_state["reliable_negative_mask"],
            "pu_reliability": pu_state["reason_reliability"],
            "predicate_observations": observations,
            "pair_mining_state": {},
            "memory_stats": {"pair_memory_count": 0},
            "calibration_stats": {"theta_mean": float(thresholded["threshold_logit"].detach().mean().cpu())},
            "action_independence_stats": {"final_action_uses_reason": False, "action_set_affects_final_action": False},
            "branch_logits": {
                "base_fixed": base,
                "deploy_fixed": thresholded["logits_deploy"],
                "calibrated": thresholded["logits_calibrated"],
                "action_visual": act["action_logits_visual"],
                "action_predicate": act["action_logits_predicate"],
                "reason_visual": reason_logits_visual,
                "reason_formula": formula_logits,
            },
        }
        return out
