import torch

from fate_oia.models.aie_reason_rereader import AIEReasonRereader


def test_reason_residual_head_starts_at_zero_without_blocking_gradients():
    module = AIEReasonRereader(dim=32, reason_dim=21, num_predicates=32)
    head = module.delta_head[-1]
    torch.testing.assert_close(head.weight, torch.zeros_like(head.weight))
    torch.testing.assert_close(head.bias, torch.zeros_like(head.bias))

    private = torch.randn(3, 21, 32)
    loss = head(private).mean()
    loss.backward()
    assert head.weight.grad is not None
    assert torch.isfinite(head.weight.grad).all()
    assert head.weight.grad.abs().sum() > 0
