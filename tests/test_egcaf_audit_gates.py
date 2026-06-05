from pathlib import Path


def test_audit_declares_review_pass_and_artifact_gates():
    text = Path("fate_oia/engine/audit_egcaf_oia_implementation.py").read_text(encoding="utf-8")
    assert "REVIEW_PASS_EGCAF_OIA_V1.txt" in text
    assert "glob(\"test_egcaf_*.py\")" in text
    assert "selected_vs_random_drop.json" in text
