from __future__ import annotations

import math

from fate_oia.engine.diagnose_acpr_mosaic_ad_components import _summarize_training_rows


def test_component_diagnostic_summarizes_real_phase_d_signals() -> None:
    rows = {
        "loss_components.jsonl": [
            {"loss_action": 0.3, "loss_reason": 0.5, "dataloader_stall": False}
        ],
        "action_anchor_stats.jsonl": [
            {
                "constraint_pass": True,
                "dot_action_aux": -0.25,
                "action_grad_norm": 1.0,
                "aux_grad_norm": 0.5,
            }
        ],
        "selective_observation_stats.jsonl": [
            {
                "posterior_available": True,
                "posterior_mean": 0.4,
                "propensity_mean": 0.6,
            }
        ],
        "posterior_recovery_stats.jsonl": [
            {"summary": True, "improvement": 0.1, "available": True}
        ],
    }
    result = _summarize_training_rows(rows)
    assert result["finite_losses"] is True
    assert result["loader_stalls"] == 0
    assert result["posterior_active_rate"] == 1.0
    assert result["anchor_pass_rate"] == 1.0
    assert math.isclose(result["anchor_cosine_mean"], -0.5)
    assert result["posterior_recovery"]["improvement"] == 0.1


def test_component_diagnostic_rejects_nonfinite_losses() -> None:
    rows = {
        "loss_components.jsonl": [{"loss_action": float("nan"), "dataloader_stall": False}],
        "action_anchor_stats.jsonl": [],
        "selective_observation_stats.jsonl": [],
        "posterior_recovery_stats.jsonl": [],
    }
    assert _summarize_training_rows(rows)["finite_losses"] is False
