from fate_oia.utils.acpr_candidate_gate import ACPRActionCandidateGate


def test_candidate_gate_selects_improving_candidate_and_rejects_exploding_rate():
    gate = ACPRActionCandidateGate(["visual", "reason", "blend"], gate_ema=1.0, min_delta_f1=0.002)
    fallback = {
        "Act_mF1": 0.70,
        "per_action_F1": [0.8, 0.7, 0.6, 0.5],
        "predicted_positive_rate_per_action": [0.5, 0.4, 0.3, 0.2],
        "all_high_rate": 0.01,
    }
    metrics = {
        "fallback": fallback,
        "visual": {
            "Act_mF1": 0.71,
            "per_action_F1": [0.81, 0.7, 0.6, 0.5],
            "predicted_positive_rate_per_action": [0.51, 0.4, 0.3, 0.2],
            "all_high_rate": 0.01,
        },
        "reason": {
            "Act_mF1": 0.72,
            "per_action_F1": [0.8, 0.72, 0.6, 0.5],
            "predicted_positive_rate_per_action": [0.5, 0.90, 0.3, 0.2],
            "all_high_rate": 0.01,
        },
        "blend": {
            "Act_mF1": 0.71,
            "per_action_F1": [0.8, 0.7, 0.61, 0.5],
            "predicted_positive_rate_per_action": [0.5, 0.4, 0.3, 0.2],
            "all_high_rate": 0.05,
        },
    }

    result = gate.update_from_train_calib(metrics, fallback)

    assert result["selected_candidate_forward"] == "visual"
    assert result["selected_candidate_stop"] == "fallback"
    assert result["selected_candidate_left"] == "fallback"
    assert gate.selected_candidate_id[0].item() == 0
    assert gate.selected_gate[0].item() == 1.0

