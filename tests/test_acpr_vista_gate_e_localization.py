from __future__ import annotations

from pathlib import Path

from fate_oia.engine.audit_acpr_vista_gates import _gate_e


def test_gate_e_blocks_without_real_localization_audit(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("bdd100k_root: E:\\sbw\\BDD100K\n", encoding="utf-8")
    payload = _gate_e(str(cfg))
    assert payload["pass"] is False
    assert payload["bdd100k_root_configured"] is True
    assert "localization" in payload["reason"]
