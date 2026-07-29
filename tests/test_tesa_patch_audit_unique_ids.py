from fate_oia.engine.tesa_diagnostics import StratifiedPatchAudit


def test_patch_audit_counts_exact_unique_ids() -> None:
    audit = StratifiedPatchAudit(max_unique=128)
    for index in range(160):
        audit.add(f"{index}.jpg", action_ids=[index % 4], factor_ids=[index % 21])
    assert audit.unique_count == 128
