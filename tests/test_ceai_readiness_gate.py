from fate_oia.utils.ceai_readiness import compute_trainer_readiness_state, default_readiness_state


def test_default_readiness_disables_r2a_and_action_router():
    state = default_readiness_state()
    assert state["a2r_active"] is True
    assert state["pair_active"] is True
    assert state["r2a_active"] is False
    assert state["router_action_scale"] == 0.0


def test_trainer_readiness_uses_branch_metrics_and_action_drop():
    prev = {
        "branch_metrics": {
            "base": {"Act_mF1": 0.70},
            "final": {"Act_mF1": 0.68},
            "reason_specialist": {"Exp_mF1": 0.45},
        },
        "diag": {
            "pair_attention_stats.pair_attention_concentration": 0.5,
            "pair_reliability_stats.q_ar_std": 0.05,
        },
    }
    state = compute_trainer_readiness_state(prev, previous_action_drop_epochs=1, evidence_gate_ok=True)
    assert state["r2a_active"] is False
    assert state["action_drop_epochs"] >= 2
    assert state["final_action_mf1"] < state["base_action_mf1"]


def test_trainer_readiness_can_activate_when_all_conditions_pass():
    prev = {
        "branch_metrics": {
            "base": {"Act_mF1": 0.70},
            "final": {"Act_mF1": 0.701},
            "reason_specialist": {"Exp_mF1": 0.45},
        },
        "diag": {
            "pair_attention_stats.pair_attention_concentration": 0.5,
            "pair_reliability_stats.q_ar_std": 0.05,
        },
    }
    state = compute_trainer_readiness_state(prev, evidence_gate_ok=False, evidence_not_used_for_action=True)
    assert state["r2a_active"] is True
    assert state["router_action_scale"] > 0.0
