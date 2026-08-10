import torch
from fate_oia.losses.vetra_map_loss import VETRAMAPLoss


def test_map_scalar_state_round_trips_exactly():
    loss = VETRAMAPLoss(); loss.train()
    loss(torch.randn(4,4,requires_grad=True), torch.tensor([[1,0,1,0],[0,1,0,1],[1,0,0,1],[0,1,1,0.]])).backward()
    clone = VETRAMAPLoss(); clone.load_state_dict(loss.state_dict())
    for key, value in loss.state_dict().items(): assert torch.equal(value, clone.state_dict()[key])
