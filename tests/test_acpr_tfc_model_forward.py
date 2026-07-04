import torch

from fate_oia.models.acpr_tfc_model import ACPRTFCModel


def test_tfc_model_forward_shapes_and_firewall():
    model = ACPRTFCModel(use_mock_dino=True, factor_topk_tokens=8)
    images = torch.randn(2, 3, 360, 640)
    action = torch.zeros(2, 4); reason = torch.zeros(2, 21); reason[:, 0] = 1
    out = model(images, action, reason, epoch=7, split="train", run_deletion=True)
    for key in [
        "action_visual_logits", "action_tfc_delta", "action_logits_base", "action_logits_deploy",
        "reason_visual_logits", "reason_tfc_delta", "reason_logits_base", "reason_logits_deploy",
        "factor_probs_action", "factor_rho_action", "factor_probs_reason", "factor_rho_reason",
        "credit_action", "credit_reason", "credit_confidence_action", "credit_confidence_reason",
        "action_theta", "reason_theta", "theta_delta_action", "theta_delta_reason", "pu_state",
        "deletion_stats", "artifact_stats",
    ]:
        assert key in out
    assert out["action_logits_deploy"].shape == (2, 4)
    assert out["reason_logits_deploy"].shape == (2, 21)
    assert torch.isfinite(out["action_logits_deploy"]).all()
    assert torch.allclose(out["action_logits_deploy"], out["action_logits_base"] - out["action_theta"], atol=1e-6)
