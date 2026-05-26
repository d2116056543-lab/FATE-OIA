import torch

from fate_oia.models.fate_oia_model import FATEOIAFeatureModel


def test_fixed_alpha_fusion_matches_weighted_logits():
    torch.manual_seed(7)
    model = FATEOIAFeatureModel(
        dim=16,
        action_dim=4,
        reason_dim=6,
        fusion_mode="fixed_alpha",
        fusion_fixed_alpha=0.25,
    )
    out = model(torch.randn(2, 5, 16))
    expected = 0.25 * out["action_visual_logits"] + 0.75 * out["action_reason_logits"]
    assert torch.allclose(out["action_fused_logits"], expected, atol=1e-6)
    assert torch.allclose(out["fusion_gate"], torch.full_like(out["fusion_gate"], 0.25))


def test_gated_floor_never_collapses_to_zero_or_one():
    torch.manual_seed(11)
    model = FATEOIAFeatureModel(
        dim=16,
        action_dim=4,
        reason_dim=6,
        fusion_mode="gated_floor",
        fusion_gate_floor=0.2,
    )
    out = model(torch.randn(3, 7, 16))
    assert torch.all(out["fusion_gate"] >= 0.2)
    assert torch.all(out["fusion_gate"] <= 0.8)

