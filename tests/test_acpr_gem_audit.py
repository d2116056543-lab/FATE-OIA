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
        "pair_memory_runtime_checks",
        "gate_results",
        "memory_results",
        "review_pass_path",
    ]:
        assert key in keys


def test_audit_source_blocks_old_pair_memory_and_zero_worker_loader():
    from pathlib import Path

    src = Path("fate_oia/engine/audit_acpr_gem_implementation.py").read_text(encoding="utf-8")
    assert "pair_memory_enqueue_no_cat" in src
    assert "pair_memory_ring_buffer" in src
    assert "pair_memory_device_config" in src
    assert "train_loader_uses_config_workers" in src
