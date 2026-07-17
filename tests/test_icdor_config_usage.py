from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

from fate_oia.engine.train_acpr_mosaic_trust_icdor import _record_resolved_config_usage, load_config


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


def test_credibility_and_fine_transport_config_leaves_are_runtime_consumed(tmp_path: Path) -> None:
    """CREDO v4 values must reach runtime code, not only the YAML manifest."""
    module = importlib.import_module("fate_oia.utils.mosaic_config_usage")
    sources = (
        Path("configs/fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml"),
        Path("configs/mosaic_icdor_factor_candidates.yaml"),
        Path("configs/mosaic_icdor_action_routes.yaml"),
        Path("configs/mosaic_icdor_reason_routes.yaml"),
        Path("configs/mosaic_icdor_certificate_rules.yaml"),
    )
    resolved = module.resolve_icdor_config_tree(sources)

    payload = _record_resolved_config_usage(
        resolved,
        tmp_path,
        "fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml",
    )
    rows = {row["path"]: row for row in payload["rows"]}
    prefix = "fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml."
    expected = (
        "credibility.independent_of_reason_labels",
        "credibility.observable_cV_min_for_admission",
        "credibility.bootstrap_replicates",
        "credibility.ema_decay",
        "credibility.image_only_cap",
        "credibility.unknown_cap",
        "credibility.no_reliable_negative_cap",
        "fine_transport.enabled",
        "fine_transport.point_eta",
        "fine_transport.curve_eta",
        "fine_transport.region_eta",
        "fine_transport.local_reread_offset_max",
        "fine_transport.fine_off_diagnostic",
        "fine_transport.coarse_off_diagnostic",
    )
    for suffix in expected:
        row = rows[prefix + suffix]
        assert row["status"] == "consumed"
        assert row["consumer_symbol"] != "load_config/run_manifest"


def test_load_config_rejects_credo_credibility_or_transport_contract_drift(tmp_path: Path) -> None:
    """Formal CREDO values are a training contract, not tunable leftovers."""
    source = Path("configs/fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["credibility"]["unknown_cap"] = 0.01
    drifted_credibility = tmp_path / "drifted_credibility.yaml"
    drifted_credibility.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="credibility.unknown_cap"):
        load_config(drifted_credibility)

    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["fine_transport"]["local_reread_offset_max"] = 0.04
    drifted_transport = tmp_path / "drifted_transport.yaml"
    drifted_transport.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="fine_transport.local_reread_offset_max"):
        load_config(drifted_transport)

