from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from fate_oia.engine.profile_acpr_mosaic_trust_icdor import (
    ICDORProfileError,
    measure_real_phase_d,
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


def test_profiler_emits_candidate_progress_without_changing_selection() -> None:
    events = []

    def measure(candidate, workers):
        return {
            **candidate,
            "num_workers": workers,
            "warmup_steps": 20,
            "measured_steps": 100,
            "samples_per_sec": 4.0,
            "max_reserved_gb": 20.0,
            "status": "PASS",
        }

    payload = profile_runtime_candidates(
        [{"batch_size": 4, "grad_accum": 8}], [2], measure,
        max_reserved_gb=43.5, progress_callback=events.append,
    )

    assert payload["selected"]["batch_size"] == 4
    assert [event["event"] for event in events] == ["candidate_start", "candidate_end"]
    assert events[0]["num_workers"] == 2


def test_profiler_resume_skips_only_completed_real_candidates() -> None:
    calls = []

    def measure(candidate, workers):
        calls.append((candidate["batch_size"], workers))
        return {
            **candidate,
            "num_workers": workers,
            "warmup_steps": 20,
            "measured_steps": 100,
            "samples_per_sec": 4.0,
            "max_reserved_gb": 20.0,
            "status": "PASS",
        }

    partial_rows = [{
        "batch_size": 6, "grad_accum": 5, "num_workers": 4,
        "warmup_steps": 20, "measured_steps": 100, "samples_per_sec": 4.0,
        "max_reserved_gb": 20.0, "status": "PASS",
        "measurement_origin": "real_phase_d_execution",
    }]
    checkpoints = []
    payload = profile_runtime_candidates(
        [{"batch_size": 6, "grad_accum": 5}], [4, 0], measure,
        max_reserved_gb=43.5, completed_records=partial_rows,
        checkpoint_callback=lambda rows: checkpoints.append(list(rows)),
    )

    assert calls == [(6, 0)]
    assert len(payload["candidates"]) == 2
    assert len(checkpoints) == 1
    assert checkpoints[0][-1]["num_workers"] == 0


def test_real_profiler_emits_periodic_step_progress() -> None:
    class TinyProfileModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action = nn.Linear(3, 4)
            self.reason = nn.Linear(3, 21)

        def forward(self, image, **_kwargs):
            pooled = image.mean(dim=(2, 3))
            return {
                "action_final_logits": self.action(pooled),
                "reason_observed_logits": self.reason(pooled),
            }

    model = TinyProfileModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = {
        "image": torch.randn(1, 3, 2, 2),
        "action": torch.zeros(1, 4),
        "reason": torch.zeros(1, 21),
    }
    events = []
    record = measure_real_phase_d(
        model, optimizer, [batch], device=torch.device("cpu"), batch_size=1,
        grad_accum=1, num_workers=0, progress_callback=events.append,
        progress_every=25,
    )

    assert record["status"] == "PASS"
    assert [event["step"] for event in events] == [25, 50, 75, 100, 120]
    assert events[-1]["measured_steps_complete"] == 100


def test_real_profiler_reuses_one_grounding_index_and_unpacks_every_loader_split() -> None:
    source = Path("fate_oia/engine/profile_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    assert "grounding_index = BDD100KGroundingIndex" in source
    assert "loader, _, _, _, _, _ = build_icdor_loaders(" in source
    assert "visual_grounding_index=grounding_index" in source

