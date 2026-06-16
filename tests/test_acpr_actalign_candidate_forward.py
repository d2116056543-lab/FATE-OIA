import torch

from fate_oia.models.acpr_oia_model import ACPROIAModel


def test_model_forward_returns_candidates_and_preserves_zero_gate_invariant():
    torch.manual_seed(12)
    base = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, actalign_enabled=False)
    torch.manual_seed(12)
    model = ACPROIAModel(
        use_mock_dino=True,
        threshold_enabled=True,
        actalign_enabled=True,
        actalign_kwargs={"mode": "candidate_probe"},
    )
    x = torch.randn(2, 3, 360, 640)
    out_base = base(x)
    out = model(x)

    assert "action_candidate_logits" in out
    assert "action_utility_stats" in out
    assert "pred_delta_max_abs" in out
    assert "pred_delta_per_action_mean" in out
    assert out["pred_delta_per_action_mean"].shape == (4,)
    for name in ["fallback", "visual", "reason", "blend", "predicate", "blend_predicate"]:
        assert name in out["action_candidate_logits"]
    assert torch.allclose(out["action_logits_utility"], out["action_logits_fallback"], atol=1e-6)
    assert torch.allclose(out["action_logits_final_raw"], out_base["action_logits_final_raw"], atol=1e-6)
    assert torch.allclose(out["reason_logits_final_raw"], out_base["reason_logits_final_raw"], atol=1e-6)
