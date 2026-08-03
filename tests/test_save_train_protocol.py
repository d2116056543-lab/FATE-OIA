import inspect

from fate_oia.engine import train_save_oia
from fate_oia.engine.train_save_oia import (
    build_save_split_manifest,
    build_save_splits,
    is_utility_cadence,
    save_ramps,
    utility_update_for_microbatch,
)


def test_splits_are_deterministic_and_disjoint():
    first = build_save_splits([f"{i}.jpg" for i in range(100)], seed=7)
    second = build_save_splits([f"{i}.jpg" for i in range(100)], seed=7)
    assert first == second
    assert not (set(first["main"]) & set(first["audit"]))
    assert not (set(first["main"]) & set(first["calib"]))


def test_pilot_manifest_binds_full_partition_and_active_subset() -> None:
    names = [f"{index}.jpg" for index in range(100)]
    full = build_save_splits(names, seed=7)
    active = build_save_splits(names, seed=7, max_train=12, max_audit=4, max_calib=5)
    manifest = build_save_split_manifest(names, full_split=full, active_split=active)

    assert manifest["universe_partition"]["disjoint"] is True
    assert manifest["active_subset"]["main"]["count"] == 12
    assert manifest["active_subset"]["audit"]["count"] == 4
    assert manifest["active_subset"]["calib"]["count"] == 5
    assert manifest["active_subset"]["is_full_partition"] is False


def test_ramps_follow_save_schedule():
    assert save_ramps(0.0) == {"warmup": 0.0, "grounding": 0.25, "mechanism": 0.0}
    assert save_ramps(.05)["grounding"] == 1.0
    assert save_ramps(.10)["mechanism"] == 1.0


def test_train_only_utility_cadence_runs_once_per_four_completed_updates():
    active = [
        micro for micro in range(32)
        if is_utility_cadence(micro_step=micro, optimizer_step=micro // 8, grad_accum=8)
    ]
    assert active == [31]
    assert utility_update_for_microbatch(micro_step=31, optimizer_step=3, grad_accum=8) == 4


def test_training_writes_recovery_checkpoint_before_epoch_evaluation() -> None:
    source = inspect.getsource(train_save_oia.main)
    assert "checkpoint_pre_eval.pth" in source
    assert source.index("checkpoint_pre_eval.pth") < source.index("evaluate_save_oia(")


def test_training_can_resume_the_pre_evaluation_checkpoint() -> None:
    source = inspect.getsource(train_save_oia.main)
    assert 'parser.add_argument("--resume")' in source
    assert "load_checkpoint(" in source
    assert "start_epoch = int(payload[\"epoch\"]) + 1" in source
