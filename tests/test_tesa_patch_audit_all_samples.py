from fate_oia.engine.tesa_diagnostics import StratifiedPatchAudit


def test_patch_audit_keeps_all_eligible_targets() -> None:
    audit = StratifiedPatchAudit(max_unique=128)
    audit.add("a.jpg", action_ids=[0, 1], factor_ids=[2, 3, 4])
    assert len(audit.records) == 6
