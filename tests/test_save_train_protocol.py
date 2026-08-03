from fate_oia.engine.train_save_oia import build_save_splits, is_utility_cadence, save_ramps


def test_splits_are_deterministic_and_disjoint():
    first = build_save_splits([f"{i}.jpg" for i in range(100)], seed=7)
    second = build_save_splits([f"{i}.jpg" for i in range(100)], seed=7)
    assert first == second
    assert not (set(first["main"]) & set(first["audit"]))
    assert not (set(first["main"]) & set(first["calib"]))


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
