from __future__ import annotations

from pathlib import Path


def test_information_boundary_contract_is_hard_gated_by_final_audit() -> None:
    source = Path("fate_oia/engine/audit_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    assert "action_information_firewall_pass" in source
    assert "test_forward_no_annotation_leakage" in source
    assert "reason labels/logits/propensity" in source
    assert "factor raw probability" in source

