from __future__ import annotations

from pathlib import Path

from fate_oia.engine.audit_acpr_vista_implementation import audit


def test_vista_audit_runs_on_mock_forward(tmp_path):
    payload = audit("configs/fate_oia_train_360x640_acpr_vista_v1.yaml", str(tmp_path), device="cpu", write_review_pass=True)
    assert payload["architecture_checks"]["model_forward"]
    assert "missing_items" in payload

