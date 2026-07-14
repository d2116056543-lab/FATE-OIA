from __future__ import annotations

import json

import pytest

from fate_oia.engine.build_mosaic_trust_ablation_table import build_row, write_table


def _write_artifacts(run_dir) -> None:
    epoch_dir = run_dir / "epoch_007"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    (epoch_dir / "branch_metrics.json").write_text(
        json.dumps(
            {
                "action": {"visual": {"mAP": 0.72}, "factor_route_off": {"mAP": 0.53}},
                "reason": {"final_observed": {"mAP": 0.67}, "factor_route_shuffled": {"mAP": 0.41}},
            }
        ),
        encoding="utf-8",
    )
    (epoch_dir / "per_label_metrics.json").write_text(
        json.dumps({"action": {"brake": {"AP": 0.82}}, "reason": {"pedestrian": {"AP": 0.76}}}),
        encoding="utf-8",
    )
    per_target = {
        "factor_id": "pedestrian",
        "target_id": "brake",
        "direction": "support",
        "selected_effect": 0.4,
        "matched_random_effect": 0.075,
        "signed_effect": 0.4,
        "tet": 0.4,
        "tes": 0.325,
        "cca": 0.35,
        "ap_delta": 5.0 / 12.0,
        "admitted": True,
    }
    (epoch_dir / "target_transfer_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "mosaic_target_transfer.v1",
                "per_target": [per_target],
                "summary": {"pair_count": 1, "mean_tes": 0.325, "admitted_rate": 1.0},
            }
        ),
        encoding="utf-8",
    )
    (epoch_dir / "target_transfer_stats.jsonl").write_text(json.dumps(per_target) + "\n", encoding="utf-8")


def test_ablation_table_uses_branch_label_and_transfer_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "experiment_a"
    _write_artifacts(run_dir)

    row = build_row(run_dir)
    output_json, output_csv = tmp_path / "table.json", tmp_path / "table.csv"
    write_table([row], output_json=output_json, output_csv=output_csv)

    assert row["branch.action.visual.mAP"] == pytest.approx(0.72)
    assert row["per_label.reason.pedestrian.AP"] == pytest.approx(0.76)
    assert row["transfer.mean_tes"] == pytest.approx(0.325)
    assert row["transfer.admitted_rate"] == pytest.approx(1.0)
    assert json.loads(output_json.read_text(encoding="utf-8"))[0]["transfer.pair_count"] == 1
    assert "branch.reason.factor_route_shuffled.mAP" in output_csv.read_text(encoding="utf-8-sig")


def test_ablation_table_fails_closed_on_unavailable_or_missing_transfer_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "experiment_b"
    _write_artifacts(run_dir)
    epoch_dir = run_dir / "epoch_007"
    (epoch_dir / "branch_metrics.json").write_text(
        json.dumps({"action": {"visual": {"mAP": "unavailable"}}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unavailable"):
        build_row(run_dir)

    _write_artifacts(run_dir)
    (epoch_dir / "target_transfer_stats.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="target_transfer_stats.jsonl"):
        build_row(run_dir)


def test_ablation_table_rejects_transfer_summary_that_does_not_match_real_rows(tmp_path) -> None:
    run_dir = tmp_path / "experiment_c"
    _write_artifacts(run_dir)
    summary_path = run_dir / "epoch_007" / "target_transfer_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["summary"]["mean_tes"] = 0.0
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="mean_tes"):
        build_row(run_dir)
