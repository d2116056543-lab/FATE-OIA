from __future__ import annotations

import torch
from torch import nn

from fate_oia.engine.eval_acpr_mosaic_trust_icdor import evaluate_icdor


class _EvalModel(nn.Module):
    def forward(self, images: torch.Tensor, *, route_mode: str, latent_enabled: bool, return_masks: bool, **_: object):
        batch = images.shape[0]
        action = torch.tensor([[2.0, -2.0, -2.0, -2.0]], device=images.device).repeat(batch, 1)
        reason = torch.cat((torch.full((batch, 1), 2.0, device=images.device), torch.full((batch, 20), -2.0, device=images.device)), dim=1)
        return {
            "action_visual_logits": action,
            "action_shadow_logits": action + 0.1,
            "action_final_logits": action,
            "action_logits_deploy": action - 0.1,
            "reason_visual_observed_logits": reason,
            "reason_logits_latent": reason + 0.1,
            "reason_observation_prob": torch.sigmoid(reason),
            "reason_propensity": torch.full_like(reason, 0.5),
            "reason_observed_logits": reason,
            "reason_logits_deploy": reason - 0.1,
            "factor_presence_prob": torch.full((batch, 24), 0.3, device=images.device),
            "factor_visibility_prob": torch.full((batch, 24), 0.4, device=images.device),
            "factor_positive_evidence": torch.full((batch, 24), 0.1, device=images.device),
            "factor_negative_evidence": torch.full((batch, 24), 0.1, device=images.device),
            "factor_uncertainty": torch.full((batch, 24), 0.2, device=images.device),
            "factor_soft_masks": torch.full((batch, 24, 12, 20), 0.1, device=images.device),
            "prototype_stats": {"prototype_effective_count": torch.full((24,), 2.0, device=images.device)},
            "support_weights": torch.zeros(batch, 24, 4, device=images.device),
            "veto_weights": torch.zeros(batch, 24, 4, device=images.device),
            "action_support_logits": torch.zeros(batch, 4, device=images.device),
            "action_veto_logits": torch.zeros(batch, 4, device=images.device),
            "support_dustbin": torch.ones(batch, 4, device=images.device),
            "veto_dustbin": torch.ones(batch, 4, device=images.device),
            "reason_factor_router_weights": torch.zeros(batch, 21, 24, device=images.device),
        }


class _FineTransportEvalModel(_EvalModel):
    fine_transport_diagnostics = {"fine_off": True, "coarse_off": True}

    def __init__(self) -> None:
        super().__init__()
        self.factor_mask_modes: list[str] = []

    def forward(self, images: torch.Tensor, *, factor_mask_mode: str = "configured", **kwargs: object):
        self.factor_mask_modes.append(factor_mask_mode)
        output = super().forward(images, **kwargs)
        delta = {"configured": 0.0, "coarse": -0.25, "fine": 0.25}[factor_mask_mode]
        for key in (
            "action_visual_logits", "action_shadow_logits", "action_final_logits", "action_logits_deploy",
        ):
            output[key] = output[key] + delta
        for key in (
            "reason_visual_observed_logits", "reason_logits_latent", "reason_observed_logits", "reason_logits_deploy",
        ):
            output[key] = output[key] + delta
        return output


def test_evaluator_reports_all_icdor_action_reason_branches_without_test_mutation() -> None:
    loader = [{
        "image": torch.zeros(2, 3, 8, 8),
        "action": torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]]),
        "reason": torch.cat((torch.ones(2, 1), torch.zeros(2, 20)), dim=1),
        "file_name": ["a.jpg", "b.jpg"],
        "split": ["test", "test"],
    }]

    result = evaluate_icdor(_EvalModel(), loader, torch.device("cpu"), epoch=0, route_mode="off", latent_enabled=False)

    assert result["metrics_summary"]["split"] == "test"
    assert {"raw", "deploy_fixed", "test_oracle_diagnostic"} <= set(result["metrics_summary"])
    assert {
        "visual", "shadow", "final", "threshold_off", "deploy", "support_only", "veto_only",
        "factor_off", "factor_shuffled", "wrong_target", "equal_mass_random",
    } <= set(result["branch_metrics"]["action"])
    assert {"visual_observed", "latent_semantic", "observation_model", "final_observed", "factor_route_off", "factor_route_shuffled", "threshold_off", "deploy"} <= set(result["branch_metrics"]["reason"])
    assert result["logits"]["action_final_logits.pt"].shape == (2, 4)
    assert {
        "action_factor_off_logits.pt", "action_factor_shuffled_logits.pt",
        "action_wrong_target_logits.pt", "action_equal_mass_random_logits.pt",
        "reason_factor_route_off_logits.pt", "reason_factor_route_shuffled_logits.pt",
    } <= set(result["logits"])
    assert result["file_names"] == ["a.jpg", "b.jpg"]
    assert len(result["prototype_rows"]) == 24
    assert len(result["reason_rows"]) == 21
    assert {
        "propensity_mean", "residual_alpha_mean", "escape_weight_mean",
        "allowed_factor_mass_mean", "disallowed_factor_mass_mean",
        "reason_factor_mask_area_mean", "reason_factor_mask_entropy",
        "semantic_compatibility_mean", "absence_factor_mass_mean", "absence_negative_evidence_mean",
    } <= set(result["reason_rows"][0])
    assert result["reason_rows"][0]["posterior_q_observed_zero_available"] is False
    assert result["reason_rows"][0]["synthetic_hidden_positive_auprc_available"] is False
    assert result["reason_rows"][0]["top_q_observed_zero_manual_precision_available"] is False
    assert not any("synthetic_hidden" in key for key in result["logits"])
    assert result["route_rows"][0]["action_id"] == 0
    summaries = [row for row in result["route_rows"] if row.get("summary") == "per_action_route_effect"]
    assert len(summaries) == 4
    assert all(
        {"route_to_visual_rms_ratio", "support_delta_rms", "veto_delta_rms", "route_credibility_effective_mean"}
        <= set(row)
        for row in summaries
    )
    assert result["failure_rows"]
    assert result["failure_rows"][0]["file_name"] in {"a.jpg", "b.jpg"}


def test_evaluator_executes_configured_fine_transport_counterfactuals() -> None:
    loader = [{
        "image": torch.zeros(2, 3, 8, 8),
        "action": torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]]),
        "reason": torch.cat((torch.ones(2, 1), torch.zeros(2, 20)), dim=1),
        "file_name": ["a.jpg", "b.jpg"],
        "split": ["test", "test"],
    }]
    model = _FineTransportEvalModel()
    result = evaluate_icdor(model, loader, torch.device("cpu"), epoch=0, route_mode="off", latent_enabled=False)

    assert {"configured", "coarse", "fine"} <= set(model.factor_mask_modes)
    assert set(result["branch_metrics"]["fine_transport"]) == {"fine_off", "coarse_off"}
    assert {"action", "reason"} <= set(result["branch_metrics"]["fine_transport"]["fine_off"])
    assert result["fine_transport_rows"][0]["fine_off_available"] is True
    assert result["fine_transport_rows"][0]["coarse_off_available"] is True
    assert result["fine_transport_rows"][0]["fine_off_action_shadow_delta_abs_mean"] > 0.0
    assert result["fine_transport_rows"][0]["fine_off_reason_latent_delta_abs_mean"] > 0.0
