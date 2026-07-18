from __future__ import annotations

from fate_oia.utils.mosaic_icdor_artifacts import (
    ICDOR_REQUIRED_ZERO_GRADIENTS,
    _mechanism_summary_valid,
    build_icdor_mechanism_summary,
    validate_icdor_pilot_mechanism,
)


def test_mechanism_summary_reports_each_credo_route_and_refuses_missing_evidence() -> None:
    summary = build_icdor_mechanism_summary(
        epoch=2,
        visual_credibility={"credibility": [0.65, 0.35], "mean_abs_cV_delta": 0.02},
        branch_metrics={
            "action": {
                "visual": {"Act_mAP": 0.70},
                "shadow": {"Act_mAP": 0.71},
                "final": {"Act_mAP": 0.70},
            },
            "reason": {
                "visual_observed": {"Exp_mAP": 0.30},
                "final_observed": {"Exp_mAP": 0.32},
                "factor_route_off": {"Exp_mAP": 0.29},
                "factor_route_shuffled": {"Exp_mAP": 0.28},
            },
        },
        route_rows=[
            {
                "summary": "per_action_route_effect",
                "action_id": 0,
                "route_to_visual_rms_ratio": 0.04,
                "delta_gt_direction_agreement": 0.60,
                "support_delta_rms": 0.02,
                "veto_delta_rms": 0.01,
                "route_credibility_effective_mean": 0.05,
            }
        ],
        factor_rows=[{"factor_id": 0, "cV_mean": 0.65}],
        factor_audit={
            "source_split": "audit_visual",
            "factor_stats": {
                "f0": {"scores": {"content_only": 0.70, "prior_only": 0.20}},
                "f1": {"scores": {"content_only": 0.10, "prior_only": 0.20}},
            },
        },
        fine_transport_rows=[{
            "fine_mask_delta_mean": 0.03,
            "anchor_separation_mean": 0.12,
            "fine_off_action_shadow_delta_abs_mean": 0.04,
            "fine_off_reason_latent_delta_abs_mean": 0.05,
            "coarse_off_action_shadow_delta_abs_mean": 0.01,
            "coarse_off_reason_latent_delta_abs_mean": 0.02,
        }],
        route_ownership_rows=[{"action_final_visual_equal": True}],
        reason_rows=[
            {
                "reason_id": reason_id,
                "factor_route_effect_abs_mean": 0.04,
                "factor_shuffle_effect_abs_mean": 0.03,
                "absence_factor_mass_mean": 0.02,
                "absence_negative_evidence_mean": 0.20,
            }
            for reason_id in (9, 15)
        ],
        hidden_recovery_rows=[{"available": True, "margin": 0.03}],
        target_transfer_rows=[{"available": True, "tet": 0.04, "tes": 0.02, "cca": 0.65}],
        gradient_rows=[
            {"loss": loss, "owner_group": owner, "grad_norm": 0.0, "finite": True}
            for loss, owners in ICDOR_REQUIRED_ZERO_GRADIENTS.items()
            for owner in owners
        ],
        edge_admission={"entries": {"support:a:forward": {"accepted": True}}},
        pu_enabled=True,
    )

    assert summary["available"] is True
    assert summary["continuous_credibility"]["nonzero_factor_count"] == 2
    assert summary["continuous_credibility"]["content_beats_prior_factor_count"] == 1
    assert summary["action_shadow"]["available"] is True
    assert summary["action_shadow"]["support_nonzero_action_count"] == 1
    assert summary["action_shadow"]["veto_nonzero_action_count"] == 1
    assert summary["action_shadow"]["route_credibility_effective_mean"] == 0.05
    assert summary["reason_transport"]["route_off_delta_exp_map"] > 0.0
    assert summary["fine_transport"]["available"] is True
    assert summary["fine_transport"]["fine_off_action_shadow_delta_abs_mean"] > 0.0
    assert summary["fine_transport"]["fine_off_reason_latent_delta_abs_mean"] > 0.0
    assert summary["pu"]["enabled_by_margin"] is True
    assert summary["target_effectiveness"]["available"] is True
    assert summary["gradient_firewall"]["available"] is True
    assert _mechanism_summary_valid(summary) is True

    missing = build_icdor_mechanism_summary(
        epoch=0,
        visual_credibility={},
        branch_metrics={},
        route_rows=[],
        factor_rows=[],
        factor_audit={},
        fine_transport_rows=[],
        route_ownership_rows=[],
        reason_rows=[],
        hidden_recovery_rows=[],
        target_transfer_rows=[],
        gradient_rows=[],
        edge_admission={},
        pu_enabled=True,
    )
    assert missing["available"] is False
    assert "branch_metrics" in missing["missing_evidence"]


def test_pilot_mechanism_validation_requires_learning_access_not_deployment_admission() -> None:
    base = {
        "available": True,
        "continuous_credibility": {"available": True, "content_beats_prior_factor_count": 1},
        "fine_transport": {
            "available": True,
            "fine_off_action_shadow_delta_abs_mean": 0.03,
            "fine_off_reason_latent_delta_abs_mean": 0.04,
        },
        "reason_transport": {
            "available": True,
            "route_off_logit_delta_abs_mean": 0.03,
            "shuffle_logit_delta_abs_mean": 0.02,
            "visual_exp_map": 0.30,
            "final_exp_map": 0.301,
            "no_lane_absence_polarity": {
                "available": True,
                "contract": "observability_times_absence",
            },
        },
        "action_shadow": {
            "available": True,
            "final_visual_exact": True,
            "route_to_visual_rms_ratio_mean": 0.02,
            "support_nonzero_action_count": 1,
            "veto_nonzero_action_count": 1,
            "final_act_map": 0.60,
        },
        "pu": {"available": True, "enabled_by_margin": True, "schedule_enabled": True},
        "gradient_firewall": {"available": True, "pass": True},
    }
    later = {
        **base,
        "epoch": 1,
        "action_shadow": {**base["action_shadow"], "final_act_map": 0.61},
        "continuous_credibility": {**base["continuous_credibility"], "content_beats_prior_factor_count": 2},
        "reason_transport": {**base["reason_transport"], "final_exp_map": 0.302},
    }

    result = validate_icdor_pilot_mechanism([{**base, "epoch": 0}, later])

    assert result["pass"] is True
    assert result["errors"] == []
