from pathlib import Path

from fate_oia.optim.heca_optimization import (
    corruption_microbatch_index,
    identity_corruption_mode,
)


def test_identity_corruption_rotates_once_per_microbatch() -> None:
    assert [identity_corruption_mode(i) for i in range(6)] == [
        "schema", "cross_sample", "state", "schema", "cross_sample", "state"
    ]


def test_identity_corruption_uses_global_microbatch_phase_after_accumulation() -> None:
    modes = [
        identity_corruption_mode(
            corruption_microbatch_index(
                epoch=1, micro_step=micro_step, microbatches_per_epoch=5
            )
        )
        for micro_step in range(5)
    ]
    assert modes == ["state", "schema", "cross_sample", "state", "schema"]


def test_trainer_passes_microbatch_phase_to_identity_loss() -> None:
    source = Path("fate_oia/engine/train_acpr_meter_oia.py").read_text(encoding="utf-8")
    assert "corruption_microbatch_index(" in source
    assert "epoch, micro_step, len(train_loader)" in source
    assert "corruption_step=corruption_step" in source
