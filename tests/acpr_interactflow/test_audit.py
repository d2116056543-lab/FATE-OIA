from __future__ import annotations

from pathlib import Path

from fate_oia.engine.audit_acpr_interactflow import run_audit


def test_audit_writes_report(tmp_path: Path):
    report = run_audit("configs/acpr_interactflow_pp_v1_psi_damo_11902.yaml", str(tmp_path), device="cpu", write_review_pass=False)
    assert "functional_checks" in report
    assert report["smoke_result"]["forward_ok"] is True

