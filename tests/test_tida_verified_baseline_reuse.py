from fate_oia.engine.train_tida_oia import verified_image_baseline_reuse_allowed


def test_logit_flow_owner_only_run_can_reuse_verified_image_baseline():
    owners = {"logit_flow_action", "logit_flow_reason"}

    assert verified_image_baseline_reuse_allowed(owners) is True


def test_private_local_query_owners_can_reuse_verified_image_baseline():
    owners = {"action_local_query", "reason_local_query"}

    assert verified_image_baseline_reuse_allowed(owners) is True


def test_unscoped_all_owner_run_must_recompute_image_baseline():
    assert verified_image_baseline_reuse_allowed(None) is False
