from fate_oia.engine.tesa_diagnostics import StratifiedPatchAudit


def test_patch_audit_counts_exact_unique_ids() -> None:
    audit = StratifiedPatchAudit(max_unique=128)
    for index in range(160):
        audit.add(f"{index}.jpg", action_ids=[index % 4], factor_ids=[index % 21])
    assert audit.unique_count == 128


def test_patch_audit_tracks_epoch_and_prior_cumulative_ids() -> None:
    audit = StratifiedPatchAudit(max_unique=3, previous_ids={"old.jpg"})
    audit.add("old.jpg", action_ids=[0], factor_ids=[1])
    audit.add("new.jpg", action_ids=[1], factor_ids=[2])
    assert audit.unique_count == 2
    assert audit.cumulative_unique_count == 2
