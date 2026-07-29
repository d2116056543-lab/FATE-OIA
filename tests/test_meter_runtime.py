import inspect

from fate_oia.engine import profile_acpr_meter_oia as profiler


def test_tesa_runtime_profile_uses_fixed_fast_stable_candidates():
    assert profiler.DEFAULT_CANDIDATES[0] == (6, 5)
    assert all(batch * accum >= 30 for batch, accum in profiler.DEFAULT_CANDIDATES)


def test_tesa_profile_checks_one_dino_call_and_hard_memory_limit():
    source = inspect.getsource(profiler._profile_candidate)
    assert "ordinary_dino_calls == 1" in source
    assert 'config["runtime"]["hard_reserved_gb"]' in source
    assert "cleanup_profile_trial(device)" in source
