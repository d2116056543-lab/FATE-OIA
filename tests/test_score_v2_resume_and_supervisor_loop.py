from __future__ import annotations

from pathlib import Path

from fate_oia.engine.supervise_score_v2_oia import next_epoch_command, next_training_epoch
from fate_oia.engine.supervise_score_v2_stage2_oia import _next_epoch, score_v2_stage2_decision


def test_next_epoch_command_uses_resume_after_first_checkpoint(tmp_path: Path) -> None:
    latest = tmp_path / "checkpoint_latest.pth"
    cmd = next_epoch_command(
        python_executable="python",
        train_module="fate_oia.engine.train_score_v2_oia",
        output_dir=tmp_path,
        epoch=1,
        base_args=["--batch_size", "2"],
    )
    assert "--epochs" in cmd
    assert "1" in cmd
    assert "--resume" not in cmd

    latest.write_bytes(b"fake")
    cmd = next_epoch_command(
        python_executable="python",
        train_module="fate_oia.engine.train_score_v2_oia",
        output_dir=tmp_path,
        epoch=2,
        base_args=["--batch_size", "2"],
    )
    assert "--epochs" in cmd
    assert "2" in cmd
    assert "--resume" in cmd
    assert str(latest) in cmd


def test_next_training_epoch_continues_after_existing_metrics() -> None:
    assert next_training_epoch([]) == 1
    assert next_training_epoch([{"epoch": 1}, {"epoch": 5}]) == 6


def test_stage2_next_epoch_continues_existing_output_dir() -> None:
    assert _next_epoch([]) == 1
    assert _next_epoch([{"epoch": 1}, {"epoch": 7}]) == 8


def test_stage2_decision_waits_until_min_gate_epoch_and_stops_bad_plateau() -> None:
    decision = score_v2_stage2_decision(
        epoch=5,
        rows=[{"epoch": 5, "test_joint": 0.40, "test_Exp_mF1": 0.2, "test_Exp_mAP": 0.2}],
        reference_joint=0.547844,
        min_gate_epoch=14,
        patience=3,
    )
    assert decision.continue_stage
    rows = [
        {"epoch": 14, "test_joint": 0.5000, "test_Exp_mF1": 0.3200, "test_Exp_mAP": 0.3000},
        {"epoch": 15, "test_joint": 0.5010, "test_Exp_mF1": 0.3210, "test_Exp_mAP": 0.3010},
        {"epoch": 16, "test_joint": 0.5005, "test_Exp_mF1": 0.3205, "test_Exp_mAP": 0.3005},
        {"epoch": 17, "test_joint": 0.5004, "test_Exp_mF1": 0.3204, "test_Exp_mAP": 0.3004},
    ]
    decision = score_v2_stage2_decision(epoch=17, rows=rows, reference_joint=0.547844, min_gate_epoch=14, patience=3)
    assert not decision.continue_stage
    assert "plateau" in decision.reason
