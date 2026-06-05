from __future__ import annotations

import torch
from torch import nn

from fate_oia.losses.asymmetric_loss import AsymmetricLossMultiLabel
from fate_oia.losses.egcaf_gradient_budget import true_gradient_budget
from fate_oia.models.egcaf_factor_judge import FactorJudge


class EGCafLoss(nn.Module):
    def __init__(
        self,
        rho: float = 0.10,
        sparse_weight: float = 0.01,
        diversity_weight: float = 0.01,
        sufficiency_weight: float = 0.10,
        comprehensiveness_weight: float = 0.10,
        scene_state_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.asl = AsymmetricLossMultiLabel()
        self.rho = rho
        self.sparse_weight = sparse_weight
        self.diversity_weight = diversity_weight
        self.sufficiency_weight = sufficiency_weight
        self.comprehensiveness_weight = comprehensiveness_weight
        self.scene_state_weight = scene_state_weight
        self.judge = FactorJudge()

    def forward(self, outputs: dict, y_action: torch.Tensor, y_reason: torch.Tensor, shared_params=None, scene_state_targets: torch.Tensor | None = None, scene_state_available: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        loss_action_core = self.asl(outputs["action_core_logits"], y_action)
        loss_action_final = self.asl(outputs["action_final_logits"], y_action)
        loss_reason = self.asl(outputs["reason_logits"], y_reason)
        main = loss_action_core + loss_action_final + loss_reason
        weights = outputs["factor_weights"]
        sparse = (weights > 1e-4).float().mean()
        selected_sources = outputs.get("selected_factor_sources")
        if selected_sources is None:
            selected_sources = torch.zeros_like(outputs["selected_weights"])
        selected_sources = selected_sources.float()
        diversity = -torch.var(selected_sources, dim=-1).mean()
        z_all = outputs.get("z_all", outputs["z_selected_only"])
        judge = self.judge(z_all, outputs["z_selected_only"], outputs["z_without_selected"], outputs["z_without_random"], y_action)
        scene_state_loss = torch.zeros((), device=y_action.device)
        if scene_state_targets is not None and scene_state_available is not None and scene_state_available.any():
            idx = scene_state_available.bool()
            scene_state_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["scene_state_logits"][idx], scene_state_targets[idx].float())
        aux_raw = (
            self.sparse_weight * sparse
            + self.diversity_weight * diversity
            + self.sufficiency_weight * judge["loss_sufficiency"]
            + self.comprehensiveness_weight * judge["loss_comprehensiveness"]
            + self.scene_state_weight * scene_state_loss
        )
        aux, budget = true_gradient_budget(main, aux_raw, shared_params, rho=self.rho) if shared_params is not None else (
            aux_raw,
            {"norm_main": 0.0, "norm_aux": 0.0, "budget_scale": 1.0, "rho": self.rho, "used_true_grad_norm": False},
        )
        total = main + aux
        stats = {
            "loss_action_core": float(loss_action_core.detach().cpu()),
            "loss_action_final": float(loss_action_final.detach().cpu()),
            "loss_reason": float(loss_reason.detach().cpu()),
            "loss_main": float(main.detach().cpu()),
            "loss_sparse_active_rate": float(sparse.detach().cpu()),
            "loss_diversity_source_var_neg": float(diversity.detach().cpu()),
            "loss_sufficiency": float(judge["loss_sufficiency"].detach().cpu()),
            "loss_comprehensiveness": float(judge["loss_comprehensiveness"].detach().cpu()),
            "loss_scene_state": float(scene_state_loss.detach().cpu()),
            "drop_selected_loss": float(judge["drop_selected_loss"].detach().cpu()),
            "drop_random_loss": float(judge["drop_random_loss"].detach().cpu()),
            "drop_selected": float(judge["drop_selected_loss"].detach().cpu()),
            "drop_random": float(judge["drop_random_loss"].detach().cpu()),
            "selected_vs_random_action_loss_drop": float(judge["selected_vs_random_action_loss_drop"].detach().cpu()),
            "total_loss": float(total.detach().cpu()),
            **budget,
        }
        outputs["factor_judge_stats"] = judge
        return total, stats
