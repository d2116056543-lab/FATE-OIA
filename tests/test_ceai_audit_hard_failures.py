from pathlib import Path

from fate_oia.engine.audit_ceai_oia_implementation import validate_evidence_schema, validate_format_ast_yaml


def test_audit_rejects_fake_evidence_zero_zero():
    errors = validate_evidence_schema({"selected_mean": 0.0, "random_mean": 0.0, "evidence_gate_active": False})
    assert errors


def test_audit_format_check_rejects_single_line_python(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text("import os; x = 1", encoding="utf-8")
    errors = validate_format_ast_yaml([bad], [])
    assert errors
