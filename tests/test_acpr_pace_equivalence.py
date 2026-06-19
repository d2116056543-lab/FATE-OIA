import torch

from fate_oia.engine.train_acpr_oia import build_model, load_config


def test_pace_disabled_keeps_legacy_action_path():
    cfg = load_config("configs/fate_oia_train_360x640_acpr_pace_v1.yaml")
    cfg["model"]["use_mock_dino"] = True
    cfg["threshold"]["enabled"] = False
    cfg["pace"]["enabled"] = False
    model = build_model(cfg, torch.device("cpu")).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 360, 640), epoch=0)
    assert torch.allclose(out["action_logits_base"], out["action_logits_legacy_base"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["predicate_action_delta_bounded"], torch.zeros_like(out["predicate_action_delta_bounded"]), atol=1e-6)
