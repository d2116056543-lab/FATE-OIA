from __future__ import annotations

from pathlib import Path

from fate_oia.engine.supervise_score_v2_oia import next_epoch_command


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
