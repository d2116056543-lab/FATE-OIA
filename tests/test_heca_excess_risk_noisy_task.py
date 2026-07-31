import torch

from fate_oia.optim.heca_optimization import HECAExcessRiskBalancer


def test_noisy_reason_cannot_take_unbounded_shared_weight() -> None:
    balancer = HECAExcessRiskBalancer()
    balancer.update_floors(torch.tensor(0.2), torch.tensor(0.2))
    weights = balancer.weights(torch.tensor(0.3), torch.tensor(100.0))
    assert 0.45 <= weights["action"] <= 0.70
    assert 0.30 <= weights["reason"] <= 0.55
    assert abs(weights["action"] + weights["reason"] - 1.0) < 1e-7

