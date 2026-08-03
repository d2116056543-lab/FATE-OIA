from fate_oia.engine.profile_save_oia import (
    SAVE_PROFILE_CANDIDATES,
    choose_fastest_stable,
    validate_profile_row,
)


def test_profile_uses_the_same_single_owner_loss_registry_as_training():
    source = __import__("inspect").getsource(__import__("fate_oia.engine.profile_save_oia", fromlist=["x"]))
    assert "build_save_loss_registry" in source
    assert "save_grounding_loss" in source


def test_profile_uses_only_planned_candidates_and_selects_throughput():
    assert SAVE_PROFILE_CANDIDATES == ((6, 5), (4, 8), (3, 11))
    rows = [
        _row(6, 5, speed=7.0, reserved=46.0),
        _row(4, 8, speed=6.0, reserved=40.0),
        _row(3, 11, speed=5.0, reserved=35.0),
    ]
    chosen = choose_fastest_stable(rows)
    assert (chosen["batch_size"], chosen["gradient_accumulation_steps"]) == (4, 8)


def test_profile_rejects_mock_or_reencoded_rows():
    row = _row(4, 8, speed=6.0, reserved=40.0)
    row["real_dino"] = False
    assert not validate_profile_row(row)
    row["real_dino"] = True
    row["ordinary_dino_calls_per_microbatch"] = 2
    assert not validate_profile_row(row)


def _row(batch, accum, *, speed, reserved):
    return {
        "batch_size": batch,
        "gradient_accumulation_steps": accum,
        "samples_per_second": speed,
        "reserved_gb": reserved,
        "real_dino": True,
        "bf16": True,
        "warmup_microbatches": 20,
        "measured_microbatches": 50,
        "ordinary_dino_calls_per_microbatch": 1,
        "core_paths": {
            "predicate": True,
            "private_reason": True,
            "utility_cadence": True,
            "paired_view_cadence": True,
        },
        "finite": True,
        "oom": False,
    }
