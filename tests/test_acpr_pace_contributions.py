import torch

from fate_oia.engine.train_acpr_oia import build_model, load_config


def test_model_outputs_exact_reason_and_predicate_contribution_tensors():
    cfg = load_config("configs/fate_oia_train_360x640_acpr_pace_v1.yaml")
    cfg["model"]["use_mock_dino"] = True
    cfg["threshold"]["enabled"] = False
    model = build_model(cfg, torch.device("cpu")).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 360, 640), epoch=0)
    assert out["predicate_reason_action_contrib_final"].shape == (2, 4, 21)
    assert out["predicate_reason_contrib_by_predicate"].shape[:2] == (2, 21)
    assert out["predicate_reason_positive_contrib_by_predicate"].shape == out["predicate_reason_contrib_by_predicate"].shape
    assert out["predicate_reason_negative_contrib_by_predicate"].shape == out["predicate_reason_contrib_by_predicate"].shape
    action_delta = out["action_logits_base"] - out["action_logits_legacy_base"]
    assert torch.allclose(out["predicate_reason_action_contrib_final"].sum(-1), action_delta, atol=1e-5, rtol=1e-5)
    reason_delta = out["predicate_reason_delta"]
    reason_recon = out["predicate_reason_contrib_by_predicate"].sum(-1) + out["predicate_reason_mlp_residual_delta"]
    assert torch.allclose(reason_recon, reason_delta, atol=1e-5, rtol=1e-5)
