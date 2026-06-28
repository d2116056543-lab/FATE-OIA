from __future__ import annotations

from fate_oia.engine.train_acpr_interactflow_psi import epoch_metric_blockers


def _healthy_metrics() -> dict:
    return {
        "innovation": {
            "exp29_calibrated_pred_positive_rate_0p5": 0.18,
            "predicate_positive_rate": 0.12,
            "ledger_identity_error": 0.0,
            "ledger_flow_delta_abs_mean": 0.25,
        },
        "action": {"Act_stopF1": 0.10},
        "ExpCal_mF1": 0.20,
        "Exp_mAP": 0.22,
    }


def test_epoch_metric_blockers_accepts_healthy_metrics_under_memory_cap() -> None:
    blockers = epoch_metric_blockers(_healthy_metrics(), max_memory_reserved_gib=42.0, hard_memory_cap_gib=46.0)
    assert blockers == []


def test_epoch_metric_blockers_rejects_expcal_all_zero_or_all_positive() -> None:
    low = _healthy_metrics()
    low["innovation"]["exp29_calibrated_pred_positive_rate_0p5"] = 0.0
    high = _healthy_metrics()
    high["innovation"]["exp29_calibrated_pred_positive_rate_0p5"] = 1.0

    assert any("exp29_calibrated_pred_positive_rate" in item for item in epoch_metric_blockers(low))
    assert any("exp29_calibrated_pred_positive_rate" in item for item in epoch_metric_blockers(high))


def test_epoch_metric_blockers_rejects_dead_predicate_flow_identity_and_overcap_memory() -> None:
    metrics = _healthy_metrics()
    metrics["innovation"]["predicate_positive_rate"] = 0.0
    metrics["innovation"]["ledger_identity_error"] = 1e-4
    metrics["innovation"]["ledger_flow_delta_abs_mean"] = 0.0

    blockers = epoch_metric_blockers(metrics, max_memory_reserved_gib=57.5, hard_memory_cap_gib=46.0)

    assert any("predicate_positive_rate" in item for item in blockers)
    assert any("ledger_identity_error" in item for item in blockers)
    assert any("ledger_flow_delta_abs_mean" in item for item in blockers)
    assert any("memory_reserved" in item for item in blockers)
