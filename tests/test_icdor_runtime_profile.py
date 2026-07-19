from __future__ import annotations

import json

import pytest

from fate_oia.engine.profile_acpr_mosaic_trust_icdor import (
    ICDORProfileError,
    profile_runtime_candidates,
    summarize_actual_training_epoch,
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


def test_diagnostic_profile_is_explicitly_model_only() -> None:
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
    assert payload["selected"]["measurement_origin"] == "model_only_phase_d_execution"
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


def test_actual_training_epoch_summary_requires_real_step_and_control_artifacts(tmp_path) -> None:
    epoch_dir = tmp_path / "epoch_000"
    epoch_dir.mkdir()
    (epoch_dir / "runtime_stats.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"step": 0, "step_sec": 4.0, "load_gap_sec": 1.0, "gpu_reserved_gb": 40.0},
                {"step": 1, "step_sec": 3.0, "load_gap_sec": 0.1, "gpu_reserved_gb": 42.5},
            )
        ) + "\n",
        encoding="utf-8",
    )
    (epoch_dir / "target_transfer_summary.json").write_text(
        json.dumps(
            {
                "full_target_transfer": {
                    "collection_runtime": {
                        "elapsed_seconds": 144.0,
                        "intervention_forward_calls": 36,
                        "static_context_reuse_enabled": True,
                        "static_visual_reexecution_during_controls": 0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_actual_training_epoch(tmp_path, max_reserved_gb=43.5)

    assert summary["status"] == "PASS"
    assert summary["measurement_origin"] == "train_icdor_epoch_execution"
    assert summary["profile_scope"] == "actual_training_epoch"
    assert summary["train_step_count"] == 2
    assert summary["max_reserved_gb"] == 42.5
    assert summary["target_control_seconds"] == 144.0
    assert summary["target_static_context_reuse_verified"] is True


def test_actual_training_epoch_summary_rejects_unreused_static_visual_context(tmp_path) -> None:
    epoch_dir = tmp_path / "epoch_000"
    epoch_dir.mkdir()
    (epoch_dir / "runtime_stats.jsonl").write_text(
        json.dumps({"step": 0, "step_sec": 4.0, "load_gap_sec": 0.1, "gpu_reserved_gb": 40.0}) + "\n",
        encoding="utf-8",
    )
    (epoch_dir / "target_transfer_summary.json").write_text(
        json.dumps(
            {
                "full_target_transfer": {
                    "collection_runtime": {
                        "elapsed_seconds": 1.0,
                        "intervention_forward_calls": 1,
                        "static_context_reuse_enabled": False,
                        "static_visual_reexecution_during_controls": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ICDORProfileError, match="static visual context reuse"):
        summarize_actual_training_epoch(tmp_path, max_reserved_gb=43.5)
