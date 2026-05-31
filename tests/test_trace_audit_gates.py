from pathlib import Path

def test_audit_gate_blocks_missing_T():
    text = Path("fate_oia/engine/audit_trace_oia_implementation.py").read_text()
    assert "bad_T_shape" in text and "REVIEW_PASS_TRACE_OIA.txt" in text
