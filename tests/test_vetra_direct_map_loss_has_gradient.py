import torch
from fate_oia.losses.vetra_map_loss import VETRAMAPLoss


def test_smooth_ap_has_finite_nonzero_gradient_and_state():
    logits = torch.tensor([[2.,-1.,.4,-.2],[-1.,2.,-.3,.6],[.5,.3,1.,-1.]], requires_grad=True)
    target = torch.tensor([[1.,0.,1.,0.],[0.,1.,0.,1.],[1.,0.,1.,0.]])
    loss = VETRAMAPLoss()(logits, target); loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None and float(logits.grad.abs().sum()) > 0
