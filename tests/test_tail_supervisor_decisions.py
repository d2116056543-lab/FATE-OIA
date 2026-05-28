from __future__ import annotations

from fate_oia.engine.supervise_tail_adapter_oia import (
    RUN_C_REFERENCE,
    TailStageDecision,
    should_continue_p1,
    should_continue_p2,
)


def test_p1_stops_when_exp_gain_is_too_small_after_two_epochs() -> None:
    decision = should_continue_p1(epoch=2, best_exp_mf1=RUN_C_REFERENCE["exp_mf1"] + 0.001)
    assert isinstance(decision, TailStageDecision)
    assert not decision.continue_stage
    assert "Exp_mF1" in decision.reason


def test_p2_stops_when_joint_and_exp_remain_below_run_c_after_three_epochs() -> None:
    decision = should_continue_p2(
        epoch=3,
        best_joint=RUN_C_REFERENCE["joint"] - 0.010,
        best_exp_mf1=RUN_C_REFERENCE["exp_mf1"] - 0.010,
        best_act_mf1=RUN_C_REFERENCE["act_mf1"],
    )
    assert not decision.continue_stage
    assert "Run C" in decision.reason


def test_p2_stops_when_action_drops_without_exp_gain() -> None:
    decision = should_continue_p2(
        epoch=1,
        best_joint=RUN_C_REFERENCE["joint"],
        best_exp_mf1=RUN_C_REFERENCE["exp_mf1"] + 0.001,
        best_act_mf1=RUN_C_REFERENCE["act_mf1"] - 0.020,
    )
    assert not decision.continue_stage
    assert "action" in decision.reason.lower()
