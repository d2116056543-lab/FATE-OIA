from __future__ import annotations

import torch
from torch import nn

from fate_oia.losses.asymmetric_loss import AsymmetricLossMultiLabel
from fate_oia.losses.egcaf_gradient_budget import true_gradient_budget
from fate_oia.models.egcaf_factor_judge import FactorJudge


class EGCafLoss(nn.Module):
    def __init__(self, rho: float = 0.10, sparse_weight: float = 0.01, diversity_weight: float = 0.01, sufficiency_weight: float = 0.10, comprehensiveness_weight: float = 0.10) -> None:
        super().__init__()
        self.asl = AsymmetricLossMultiLabel()
        self.rho = rho
        self.sparse_weight = sparse_weight
        self.diversity_weight = diversity_weight
        self.sufficiency_weight = sufficiency_weight
        self.comprehensiveness_weight = comprehensiveness_weight
        self.judge = FactorJudge()

    def forward(self, outputs: dict, y_action: torch.Tensor, y_reason: torch.Tensor, shared_params=None) -> tuple[torch.Tensor, dict[str, float]]:
        loss_action_core = self.asl(outputs["action_core_logits"], y_action)
        loss_action_final = self.asl(outputs["action_final_logits"], y_action)
        loss_reason = self.asl(outputs["reason_logits"], y_reason)
        main = loss_action_core + loss_action_final + loss_reason
        weights = outputs["factor_weights"]
        sparse = weights.mean()
        diversity = (outputs["selected_weights"].sum(-1) - 1.0).abs().mean()
        judge = self.judge(outputs["z_selected_only"], outputs["z_without_selected"], outputs["z_without_random"], y_action)
        aux_raw = self.sparse_weight * sparse + self.diversity_weight * diversity + self.sufficiency_weight * judge["loss_sufficiency"] + self.comprehensiveness_weight * judge["loss_comprehensiveness"]
        aux, budget = true_gradient_budget(main, aux_raw, shared_params, rho=self.rho) if shared_params is not None else (aux_raw, {"norm_main": 0.0, "norm_aux": 0.0, "budget_scale": 1.0, "rho": self.rho, "used_true_grad_norm": False})
        total = main + aux
        stats = {
            "loss_action_core": float(loss_action_core.detach().cpu()),
            "loss_action_final": float(loss_action_final.detach().cpu()),
            "loss_reason": float(loss_reason.detach().cpu()),
            "loss_main": float(main.detach().cpu()),
            "loss_sparse": float(sparse.detach().cpu()),
            "loss_diversity": float(diversity.detach().cpu()),
            "loss_sufficiency": float(judge["loss_sufficiency"].detach().cpu()),
            "loss_comprehensiveness": float(judge["loss_comprehensiveness"].detach().cpu()),
            "drop_selected": float(judge["drop_selected"].detach().cpu()),
            "drop_random": float(judge["drop_random"].detach().cpu()),
            "total_loss": float(total.detach().cpu()),
            **budget,
        }
        return total, stats
