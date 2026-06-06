from pathlib import Path
from fate_oia.engine.audit_diva_caf_oia_implementation import static_source_audit

def test_static_audit_detects_required_sources():
    result = static_source_audit(Path('.'))
    assert isinstance(result, dict)
    assert 'checks' in result
