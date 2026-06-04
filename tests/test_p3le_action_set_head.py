import torch

from fate_oia.models.p3le_action_set_head import ActionSetHead


def test_action_set_head_uses_fixed_exact_vector_prior_buffer():
    head = ActionSetHead(dim=16, action_dim=4)
    assert "prototype_vectors" in dict(head.named_buffers())
    assert head.prototype_vectors.shape == (8, 4)
    assert torch.allclose(head.prototype_vectors[0], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert hasattr(head, "prototype_residual")
    out = head(torch.randn(3, 4, 16))
    assert out["action_set_logits"].shape == (3, 4)
    assert out["action_prototype_usage_mean"].shape == (8,)

