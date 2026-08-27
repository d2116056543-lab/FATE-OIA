from fate_oia.utils.tida_traffic_evidence_report import (
    build_traffic_evidence_summary,
    render_traffic_evidence_html,
)


def _effectiveness():
    return {
        "action": {
            "conditional_information_gain_bits": 0.0064,
            "conditional_nll_improvement": 0.0045,
            "conditional_nll_improvement_ci95": [0.0040, 0.0050],
            "relative_brier_reduction": 0.0127,
            "selected_minus_random_deletion_gap": 0.0078,
            "selected_minus_random_deletion_gap_ci95": [0.0068, 0.0089],
            "target_effective_route_rate": 0.18,
            "net_corrected_labels": 8,
            "utility_quality": {"helpfulness_auc": 0.77},
        },
        "reason": {
            "conditional_information_gain_bits": 0.0015,
            "conditional_nll_improvement": 0.0010,
            "conditional_nll_improvement_ci95": [0.0008, 0.0012],
            "relative_brier_reduction": 0.0042,
            "selected_minus_random_deletion_gap": 0.0018,
            "selected_minus_random_deletion_gap_ci95": [0.0012, 0.0022],
            "target_effective_route_rate": 0.066,
            "net_corrected_labels": 6,
            "utility_quality": {"helpfulness_auc": 0.96},
        },
        "interaction_risk_quartiles": [
            {"quartile": 1, "action_mf1_gain": 0.001},
            {"quartile": 4, "action_mf1_gain": 0.012},
        ],
    }


def _corrector():
    return {
        "test_labels_used_for_fit_or_selection": False,
        "deployment_routes": ["margin_only", "margin_only", "baseline", "traffic"],
        "test": {
            "base_Act_mF1": 0.7865,
            "margin_only_Act_mF1": 0.7893,
            "full_Act_mF1": 0.7906,
            "traffic_incremental_mf1": 0.0013,
            "traffic_correction_precision": 0.8,
            "traffic_errors_recovered": 4,
            "traffic_correct_damaged": 1,
            "traffic_net_corrected": 3,
            "traffic_rank_metrics_by_action": [
                {"traffic_ap_increment": 0.005, "traffic_auc_increment": 0.003},
                {"traffic_ap_increment": -0.002, "traffic_auc_increment": -0.001},
                {"traffic_ap_increment": 0.031, "traffic_auc_increment": 0.013},
                {"traffic_ap_increment": 0.014, "traffic_auc_increment": 0.004},
            ],
        },
    }


def test_summary_separates_temporal_and_selective_traffic_gain():
    summary = build_traffic_evidence_summary(
        image_metrics={"Act_mF1": 0.7817, "Exp_mF1": 0.4632},
        video_metrics={"Act_mF1": 0.7865, "Exp_mF1": 0.4631},
        final_metrics={"Act_mF1": 0.7906, "Exp_mF1": 0.4631},
        effectiveness=_effectiveness(),
        corrector=_corrector(),
    )

    assert abs(summary["task_gain"]["image_to_video_action_mf1"] - 0.0048) < 1e-9
    assert abs(summary["task_gain"]["image_to_final_action_mf1"] - 0.0089) < 1e-9
    assert abs(summary["task_gain"]["traffic_over_margin_action_mf1"] - 0.0013) < 1e-9
    assert summary["evidence_verdict"]["claim"] == "causal_transport_supported"
    assert summary["evidence_verdict"]["test_leakage_free"] is True
    assert summary["traffic_rank_gain"][2]["ap_increment"] == 0.031


def test_summary_downgrades_claim_when_deletion_ci_crosses_zero():
    effectiveness = _effectiveness()
    effectiveness["action"]["selected_minus_random_deletion_gap_ci95"] = [-0.001, 0.008]

    summary = build_traffic_evidence_summary(
        image_metrics={"Act_mF1": 0.78, "Exp_mF1": 0.46},
        video_metrics={"Act_mF1": 0.785, "Exp_mF1": 0.46},
        final_metrics={"Act_mF1": 0.786, "Exp_mF1": 0.46},
        effectiveness=effectiveness,
        corrector=_corrector(),
    )

    assert summary["evidence_verdict"]["claim"] == "predictive_association_only"
    assert "positive_deletion_ci" in summary["evidence_verdict"]["failed_checks"]


def test_html_contains_metrics_mechanism_and_case_links():
    summary = build_traffic_evidence_summary(
        image_metrics={"Act_mF1": 0.7817, "Exp_mF1": 0.4632},
        video_metrics={"Act_mF1": 0.7865, "Exp_mF1": 0.4631},
        final_metrics={"Act_mF1": 0.7906, "Exp_mF1": 0.4631},
        effectiveness=_effectiveness(),
        corrector=_corrector(),
    )

    html = render_traffic_evidence_html(
        summary,
        case_images=["case_001_recovered.png", "case_000_damaged.png"],
    )

    assert "Traffic Evidence Scorecard" in html
    assert "Conditional information" in html
    assert "Selected vs random deletion" in html
    assert "case_001_recovered.png" in html
    assert "0.7906" in html
