from fate_oia.engine.audit_acpr_gem_implementation import required_audit_keys


def test_audit_schema_contains_required_gem_sections():
    keys = set(required_audit_keys())
    for key in [
        "pass",
        "git_head",
        "config_checks",
        "forbidden_patterns",
        "evidence_memory_checks",
        "oracle_checks",
        "trunk_integration_checks",
        "predicate_integration_checks",
        "gate_results",
        "memory_results",
        "review_pass_path",
    ]:
        assert key in keys
