import torch

from fate_oia.models.p3le_shared_encoder import P3LESharedLabelQueryEncoder


def test_shared_encoder_preserves_base_action_paths():
    model = P3LESharedLabelQueryEncoder(dim=32, action_dim=4, reason_dim=21)
    out = model(torch.randn(2, 9, 32))
    for key, shape in {
        "action_visual_logits": (2, 4),
        "reason_to_action_logits": (2, 4),
        "action_fused_logits": (2, 4),
        "reason_logits": (2, 21),
        "action_tokens": (2, 4, 32),
        "reason_tokens": (2, 21, 32),
    }.items():
        assert key in out
        assert tuple(out[key].shape) == shape
    loss = out["action_fused_logits"].mean() + out["reason_logits"].mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
