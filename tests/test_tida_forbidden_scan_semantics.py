from fate_oia.engine.audit_tida_oia_implementation import FORBIDDEN, _code_remote, _forbidden_findings


def test_forbidden_scan_accepts_explicit_no_compression():
    findings = _forbidden_findings(
        "token_compression: none\nrequire_no_token_compression: true\n",
        "config.yaml",
        "cache_distill",
        FORBIDDEN["cache_distill"],
    )
    assert findings == []


def test_forbidden_scan_rejects_real_compression():
    findings = _forbidden_findings(
        "token_compression: keep_merge\n",
        "config.yaml",
        "cache_distill",
        FORBIDDEN["cache_distill"],
    )
    assert len(findings) == 1


def test_code_remote_points_to_fate_oia_repository():
    assert _code_remote() in {"origin", "github"}
