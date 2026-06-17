from pathlib import Path
from fate_oia.engine.audit_acpr_pmt_s_implementation import run_static_audit


def test_pmt_audit_static_passes_core_files():
    result = run_static_audit(Path("."))
    assert result["functional_checks"]["required_files"]
    assert result["functional_checks"]["forbidden_patterns_absent"]
