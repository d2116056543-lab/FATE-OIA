from __future__ import annotations

import inspect

from fate_oia.engine import audit_acpr_mosaic_trust_icdor as audit


def test_audit_enforces_v5_credibility_and_schedule_contracts() -> None:
    """A V4 audit cannot certify a V5 CREDO-MAP worktree."""
    source = inspect.getsource(audit)

    assert "_V5_REQUIRED_FORWARD_OUTPUTS" in source
    assert "verify_v5_forward_contract" in source
    assert "action_shadow_credibility_floor" in source
    assert "reason_semantic_credibility_floor" in source
    assert "JOINT_SHADOW" in source
    assert "ADMISSION_CONSOLIDATION" in source
    assert "online_target_probe_due" in source
    assert "full_target_audit_due" in source
    assert "latent_reason_core_loss(" in source
    assert "pu_gate=model.reason_pu_gate" in source
    assert "factor_semantic_contract" in source
    assert "factor_audit_aligned_loss_chain" in source
    assert "factor_only_model" in source
    assert "factor_only_collector" in source
    assert "labelwise_pu" in source
    assert "route_distribution * route_mass" in source
    assert "factor_prior_presence_logits" in source
    assert "_git_branch" in source
    assert "manifest target branch does not match current branch" in source

    assert '"adaptive_schedule.full_target_audit_due("' in source
    assert '"epoch=epoch, every_epochs=refresh_every"' in source

    assert "observable_cV_min_for_admission" not in source
    assert "action_credibility_min_for_admission" not in source
    assert '"FOUNDATION"' not in source
    assert '"SAFE_JOINT"' not in source
    assert "observations,\n            None," not in source
