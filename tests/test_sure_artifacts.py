from __future__ import annotations

from pathlib import Path

from fate_oia.engine.audit_sure_oia_implementation import run_synthetic_smoke


def test_synthetic_smoke_writes_required_artifacts(tmp_path: Path) -> None:
    stats = run_synthetic_smoke(tmp_path)
    assert stats["selected_edges"] < stats["candidate_edges"]
    assert (tmp_path / "smoke" / "run_manifest.json").exists()
    assert (tmp_path / "smoke" / "epoch_000" / "metrics_summary.json").exists()
    assert (tmp_path / "smoke" / "epoch_000" / "relation_stats.json").exists()
    assert (tmp_path / "smoke" / "epoch_000" / "gradnorm_stats.json").exists()
    assert (tmp_path / "smoke" / "epoch_000" / "action_safe_stats.json").exists()
