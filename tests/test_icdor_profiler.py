from __future__ import annotations

import pytest

from fate_oia.engine.profile_acpr_mosaic_trust_icdor import (
    ICDORProfileError,
    profile_runtime_candidates,
    select_runtime_candidate,
    wall_clock_samples_per_second,
)


def test_wall_clock_throughput_penalizes_synchronous_data_wait() -> None:
    fast_loader = wall_clock_samples_per_second(8, [1.0, 1.0], [0.01, 0.01])
    blocked_loader = wall_clock_samples_per_second(8, [1.0, 1.0], [0.20, 0.20])
    assert fast_loader > blocked_loader
    assert fast_loader == pytest.approx(16.0 / 2.02)


def test_profiler_selects_the_fastest_complete_measurement_under_reserved_limit() -> None:
    selected = select_runtime_candidate(
        [
            {"batch_size": 8, "grad_accum": 4, "num_workers": 4, "warmup_steps": 20, "measured_steps": 100, "samples_per_sec": 8.0, "max_reserved_gb": 44.0, "status": "PASS"},
            {"batch_size": 6, "grad_accum": 5, "num_workers": 2, "warmup_steps": 20, "measured_steps": 100, "samples_per_sec": 7.0, "max_reserved_gb": 42.0, "status": "PASS"},
            {"batch_size": 4, "grad_accum": 8, "num_workers": 0, "warmup_steps": 20, "measured_steps": 100, "samples_per_sec": 7.0, "max_reserved_gb": 41.0, "status": "PASS"},
        ],
        max_reserved_gb=43.5,
    )

    assert selected["batch_size"] == 6
    assert selected["effective_batch"] == 30


def test_profiler_rejects_incomplete_or_unmeasured_candidates() -> None:
    with pytest.raises(ICDORProfileError, match="100 measured"):
        select_runtime_candidate(
            [{"batch_size": 4, "grad_accum": 8, "num_workers": 0, "warmup_steps": 20, "measured_steps": 3, "samples_per_sec": 7.0, "max_reserved_gb": 20.0, "status": "PASS"}],
            max_reserved_gb=43.5,
        )


def test_profiler_executes_every_configured_candidate_and_worker() -> None:
    calls = []

    def measure(candidate, workers):
        calls.append((candidate["batch_size"], workers))
        return {
            **candidate,
            "num_workers": workers,
            "warmup_steps": 20,
            "measured_steps": 100,
            "samples_per_sec": float(candidate["batch_size"] + workers),
            "max_reserved_gb": 20.0,
            "status": "PASS",
        }

    payload = profile_runtime_candidates(
        [{"batch_size": 6, "grad_accum": 5}, {"batch_size": 4, "grad_accum": 8}],
        [4, 0],
        measure,
        max_reserved_gb=43.5,
    )

    assert calls == [(6, 4), (6, 0), (4, 4), (4, 0)]
    assert payload["pass"] is True
    assert payload["selected"]["batch_size"] == 6
    assert payload["selected"]["num_workers"] == 4

