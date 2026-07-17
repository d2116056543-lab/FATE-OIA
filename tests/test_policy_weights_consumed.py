from fate_oia.engine.train_acpr_mosaic_trust_icdor import resolve_icdor_policy_weights


def test_policy_weights_are_consumed_by_loss_resolution():
    resolved = resolve_icdor_policy_weights({"action_shadow": 0.5, "reason_visual_observed": 1.0})
    assert resolved["action_shadow"] == 0.5
    assert resolved["reason_visual_observed"] == 1.0

