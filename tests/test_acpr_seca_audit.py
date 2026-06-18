from pathlib import Path


def test_seca_audit_source_contains_hard_gates():
    text = Path("fate_oia/engine/audit_acpr_seca_implementation.py").read_text(encoding="utf-8")
    assert "REVIEW_PASS_ACPR_SECA_V1.txt" in text
    assert "zero_gate_present" in text
    assert "pair_budget_present" in text
    assert "forbidden_patterns" in text
