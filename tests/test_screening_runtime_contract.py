from __future__ import annotations

import importlib


def test_screening_runtime_limits_are_bounded_without_changing_formal_defaults() -> None:
    module = importlib.import_module("fate_oia.engine.train_acpr_mosaic_trust_icdor")

    screening = module._screening_runtime_limits(
        screening=True,
        max_audit_samples=32,
        max_target_samples=128,
        bootstrap_replicates=1000,
    )
    assert screening == {
        "audit_samples": 8,
        "target_samples": 8,
        "bootstrap_replicates": 8,
    }

    formal = module._screening_runtime_limits(
        screening=False,
        max_audit_samples=512,
        max_target_samples=512,
        bootstrap_replicates=1000,
    )
    assert formal == {
        "audit_samples": 512,
        "target_samples": 512,
        "bootstrap_replicates": 1000,
    }


def test_screening_runtime_limits_never_increase_user_caps() -> None:
    module = importlib.import_module("fate_oia.engine.train_acpr_mosaic_trust_icdor")

    result = module._screening_runtime_limits(
        screening=True,
        max_audit_samples=4,
        max_target_samples=3,
        bootstrap_replicates=5,
    )
    assert result == {
        "audit_samples": 4,
        "target_samples": 3,
        "bootstrap_replicates": 5,
    }
