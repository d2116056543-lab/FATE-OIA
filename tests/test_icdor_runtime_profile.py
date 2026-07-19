from __future__ import annotations

import pytest

from fate_oia.engine.profile_acpr_mosaic_trust_icdor import (
    ICDORProfileError,
    profile_runtime_candidates,
)


def _real_measurement(candidate: dict[str, int], workers: int) -> dict[str, object]:
    return {
        **candidate,
        "num_workers": workers,
        "status": "PASS",
        "warmup_steps": 2,
        "measured_steps": 4,
        "samples_per_sec": 3.0,
        "max_reserved_gb": 10.0,
    }


def test_diagnostic_profile_is_explicitly_partial_and_real() -> None:
    payload = profile_runtime_candidates(
        [{"batch_size": 8, "grad_accum": 4}],
        [4],
        _real_measurement,
        max_reserved_gb=43.5,
        diagnostic=True,
        minimum_warmup_steps=2,
        minimum_measured_steps=4,
    )

    assert payload["status"] == "PARTIAL_DIAGNOSTIC"
    assert payload["profile_scope"] == "pilot_diagnostic"
    assert payload["selected"]["measurement_origin"] == "real_phase_d_execution"
    assert payload["batch_size"] == 8
    assert payload["grad_accum"] == 4


def test_short_measurement_cannot_be_promoted_to_formal_profile() -> None:
    with pytest.raises(ICDORProfileError, match="20 warmup steps"):
        profile_runtime_candidates(
            [{"batch_size": 8, "grad_accum": 4}],
            [4],
            _real_measurement,
            max_reserved_gb=43.5,
        )


def test_diagnostic_profile_rejects_subminimum_step_counts() -> None:
    with pytest.raises(ICDORProfileError, match="at least 2 warmup and 4 measured"):
        profile_runtime_candidates(
            [{"batch_size": 8, "grad_accum": 4}],
            [4],
            _real_measurement,
            max_reserved_gb=43.5,
            diagnostic=True,
            minimum_warmup_steps=1,
            minimum_measured_steps=4,
        )
