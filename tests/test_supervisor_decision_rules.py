from fate_oia.engine.supervise_oia_adaptive import decide_next_branch, should_stop_run


def test_decision_prefers_threshold_when_gain_is_large():
    decision = decide_next_branch(
        {
            "threshold_gain_exp_mF1": 0.025,
            "thresholded_joint_gain": 0.006,
            "fusion_fix_recommended": True,
            "long_tail_learning_problem": True,
        }
    )
    assert decision["recommended_next_run"] == "threshold_only"


def test_stop_run_after_two_bad_epochs():
    result = should_stop_run(
        completed_epoch_metrics=[
            {"epoch": 1, "joint": 0.540, "Exp_mF1": 0.370, "Act_mF1_fused": 0.710},
            {"epoch": 2, "joint": 0.541, "Exp_mF1": 0.369, "Act_mF1_fused": 0.709},
        ],
        baseline_joint=0.547844,
        baseline_exp_mF1=0.381301,
    )
    assert result["stop"] is True
    assert "no_improvement" in result["reason"]

