import inspect
from pathlib import Path

from fate_oia.engine.train_acpr_meter_oia import _compute_losses
from fate_oia.optim.heca_optimization import (
    HECAScheduleState,
    identity_corruption_mode,
)


def test_identity_corruption_rotates_once_per_microbatch() -> None:
    assert [identity_corruption_mode(i) for i in range(6)] == [
        "schema", "cross_sample", "state", "schema", "cross_sample", "state"
    ]


def test_identity_corruption_covers_ten_batches_and_an_epoch_tail() -> None:
    assert [identity_corruption_mode(index) for index in range(10)] == [
        "schema",
        "cross_sample",
        "state",
        "schema",
        "cross_sample",
        "state",
        "schema",
        "cross_sample",
        "state",
        "schema",
    ]
    # A short final accumulation window continues from the persisted batch phase.
    assert [identity_corruption_mode(index) for index in range(14, 16)] == [
        "state",
        "schema",
    ]


def test_identity_corruption_counter_survives_resume_and_batch_size_change() -> None:
    state = HECAScheduleState(
        update=2,
        total_updates=20,
        corruption_microbatch_index=7,
    )
    before = [
        identity_corruption_mode(state.corruption_microbatch_index + index)
        for index in range(5)
    ]
    restored = HECAScheduleState.from_state_dict(state.state_dict())
    after = [
        identity_corruption_mode(restored.corruption_microbatch_index + index)
        for index in range(5)
    ]
    assert before == after == ["cross_sample", "state", "schema", "cross_sample", "state"]


def test_trainer_passes_microbatch_phase_to_identity_loss() -> None:
    source = Path("fate_oia/engine/train_acpr_meter_oia.py").read_text(encoding="utf-8")
    assert "corruption_step = schedule_state.corruption_microbatch_index" in source
    assert "corruption_step=corruption_step" in source
    assert "schedule_state.corruption_microbatch_index += 1" in source
    assert source.index("schedule_state.corruption_microbatch_index += 1") < source.index(
        "if is_update:"
    )


def test_loss_graph_requires_explicit_corruption_step() -> None:
    parameter = inspect.signature(_compute_losses).parameters["corruption_step"]
    assert parameter.default is inspect.Parameter.empty
