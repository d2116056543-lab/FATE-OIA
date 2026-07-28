import inspect

import torch

from fate_oia.engine import profile_acpr_meter_oia as profiler
from fate_oia.utils.meter_runtime import (
    METERRuntimeProfile,
    choose_meter_profile,
    effective_cuda_memory_gb,
)


def test_runtime_profile_rejects_hard_memory_limit() -> None:
    result = choose_meter_profile(
        [
            METERRuntimeProfile(
                8, 4, reserved_gb=40.0, samples_per_sec=10.0
            ),
            METERRuntimeProfile(
                16, 2, reserved_gb=46.0, samples_per_sec=100.0
            ),
        ]
    )
    assert result.reserved_gb < 45.0


def test_wddm_allocator_overcount_uses_physical_device_measurement() -> None:
    assert effective_cuda_memory_gb(
        allocator_reserved_gb=57.2,
        physical_used_gb=30.7,
        physical_total_gb=48.0,
    ) == 30.7


def test_normal_allocator_measurement_keeps_conservative_maximum() -> None:
    assert effective_cuda_memory_gb(
        allocator_reserved_gb=38.0,
        physical_used_gb=40.0,
        physical_total_gb=48.0,
    ) == 40.0


def test_profiler_samples_memory_after_low_frequency_events() -> None:
    source = inspect.getsource(profiler.profile_one)
    assert source.count("sample_cuda_memory()") >= 4
    for field in (
        "memory_peak_after_ordinary_gb",
        "memory_peak_after_counterfactual_gb",
        "memory_peak_after_meta_gb",
        "memory_peak_after_calibration_gb",
    ):
        assert field in source


def test_profile_cleanup_releases_trial_before_next_candidate(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(profiler.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(
        profiler.torch.cuda,
        "synchronize",
        lambda device: calls.append(("sync", str(device))),
    )
    monkeypatch.setattr(
        profiler.torch.cuda, "empty_cache", lambda: calls.append("empty")
    )
    monkeypatch.setattr(
        profiler.torch.cuda,
        "reset_peak_memory_stats",
        lambda device: calls.append(("reset", str(device))),
    )

    profiler.cleanup_profile_trial(torch.device("cuda"))

    assert calls == [
        "gc",
        ("sync", "cuda"),
        "empty",
        ("reset", "cuda"),
    ]


def test_profile_selector_rejects_nonisolated_candidate() -> None:
    config = {
        "runtime": {
            "hard_max_reserved_gb": 45.0,
            "target_reserved_gb": 42.0,
        }
    }
    profile = {
        "oom": False,
        "finite": True,
        "reserved_gb": 20.0,
        "event_adjusted_samples_per_sec": 100.0,
        "isolation_pass": False,
    }

    assert profiler._select_profile([profile], config) is None


def test_profiler_runs_each_candidate_in_synchronous_child_process() -> None:
    source = inspect.getsource(profiler.main)
    assert "subprocess.run(" in source
    assert '"--single_trial"' in source
    assert "_wait_for_gpu_baseline(" in source
    for field in (
        '"git_head"',
        '"source_tree_hash"',
        '"config_hash"',
        '"schema_hash"',
        '"schema_version"',
        '"selected_validation"',
    ):
        assert field in source
