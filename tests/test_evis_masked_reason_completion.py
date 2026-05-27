import torch

from fate_oia.models.masked_reason_completion import MaskedReasonCompletion


def test_mrc_masks_only_masked_labels_and_backpropagates():
    m = MaskedReasonCompletion(32, reason_dim=5, num_heads=4)
    states = torch.randn(3, 4, 32, requires_grad=True)
    labels = torch.tensor([[1,0,0,1,0],[0,1,0,0,1],[0,0,1,0,0]], dtype=torch.float32)
    mask = torch.tensor([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]], dtype=torch.bool)
    out = m(states, labels, mrc_mask=mask)
    assert out["mrc_mask"].sum().item() == 3
    assert out["mrc_loss"].item() > 0
    out["mrc_loss"].backward()
    assert states.grad is not None
    assert float(states.grad.abs().sum()) > 0
