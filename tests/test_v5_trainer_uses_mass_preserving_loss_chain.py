from __future__ import annotations

import inspect

from fate_oia.engine.train_acpr_mosaic_trust_icdor import compute_icdor_training_losses


def test_trainer_calls_v5_action_and_reason_loss_chain() -> None:
    """V5 must train the mass/credit/latent-core path rather than legacy strength gating."""
    source = inspect.getsource(compute_icdor_training_losses)
    assert "route_strength_target" not in source
    assert "strength_weight" not in source
    assert "selected_control_logits=output[\"action_matched_random_logits\"]" in source
    assert "wrong_target_logits=output[\"action_wrong_target_logits\"]" in source
    assert "latent_reason_core_loss(" in source
    assert "pu_gate=model.reason_pu_gate" in source
