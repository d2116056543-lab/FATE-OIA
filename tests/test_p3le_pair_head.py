import torch

from fate_oia.models.p3le_pair_head import PairAwareTensorHead, build_pair_seed_targets


def test_pair_tensor_shape_and_low_risk_seed_targets():
    head = PairAwareTensorHead(dim=32, action_dim=4, reason_dim=21, rank=8)
    out = head(torch.randn(2, 4, 32), torch.randn(2, 21, 32), torch.randn(2, 32))
    assert tuple(out["pair_tensor"].shape) == (2, 4, 21)
    action = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32)
    reason = torch.zeros(1, 21)
    reason[:, [0, 1, 2, 3, 10, 20]] = 1
    targets = build_pair_seed_targets(action, reason)
    assert targets.sum() > 0
    assert targets.sum() < float(action.sum() * reason.sum())
