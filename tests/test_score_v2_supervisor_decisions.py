from __future__ import annotations

from fate_oia.engine.supervise_score_v2_oia import RUN_C_REFERENCE, score_v2_stage1_decision


def test_stage1_does_not_stop_before_min_gate_epoch_when_far_below_run_c() -> None:
    decision = score_v2_stage1_decision(
        epoch=5,
        best_joint=RUN_C_REFERENCE["joint"] - 0.04,
        best_exp_mf1=RUN_C_REFERENCE["exp_mf1"] - 0.03,
        best_exp_map=RUN_C_REFERENCE["exp_map"] - 0.01,
    )
    assert decision.continue_stage
    assert "warmup" in decision.reason.lower()


def test_stage1_stops_at_min_gate_epoch_when_far_below_run_c_and_no_ap_gain() -> None:
    decision = score_v2_stage1_decision(
        epoch=14,
        best_joint=RUN_C_REFERENCE["joint"] - 0.04,
        best_exp_mf1=RUN_C_REFERENCE["exp_mf1"] - 0.03,
        best_exp_map=RUN_C_REFERENCE["exp_map"] - 0.01,
    )
    assert not decision.continue_stage
    assert "epoch 14" in decision.reason


def test_stage1_continues_when_ap_improves_even_if_f1_lags() -> None:
    decision = score_v2_stage1_decision(
        epoch=5,
        best_joint=RUN_C_REFERENCE["joint"] - 0.03,
        best_exp_mf1=RUN_C_REFERENCE["exp_mf1"] - 0.02,
        best_exp_map=RUN_C_REFERENCE["exp_map"] + 0.02,
    )
    assert decision.continue_stage
    assert decision.next_stage is None or decision.next_stage == "calibration_analysis"
