from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_resolved_config_usage_tracks_all_five_yaml_sources_and_rejects_unused_keys() -> None:
    module = importlib.import_module("fate_oia.utils.mosaic_config_usage")
    sources = (
        Path("configs/fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml"),
        Path("configs/mosaic_icdor_factor_candidates.yaml"),
        Path("configs/mosaic_icdor_action_routes.yaml"),
        Path("configs/mosaic_icdor_reason_routes.yaml"),
        Path("configs/mosaic_icdor_certificate_rules.yaml"),
    )
    resolved = module.resolve_icdor_config_tree(sources)
    assert len(resolved["source_sha256"]) == 5
    tracker = module.ConfigUsageTracker(resolved)
    first_path = next(iter(tracker.leaf_paths))
    tracker.consume(first_path, consumer_file="x.py", consumer_symbol="f")
    row = tracker.rows[first_path]
    assert row == {
        "path": first_path,
        "status": "consumed",
        "consumer_file": "x.py",
        "consumer_symbol": "f",
    }
    with pytest.raises(ValueError, match="unused"):
        tracker.finalize(require_all_consumed=True)

