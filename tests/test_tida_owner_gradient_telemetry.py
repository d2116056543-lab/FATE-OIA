import torch

from fate_oia.engine.train_tida_oia import owner_gradient_norms


def test_owner_gradient_norms_report_each_owner_without_changing_gradients():
    first = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    second = torch.nn.Parameter(torch.tensor([1.0]))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = None

    result = owner_gradient_norms({"active": [first], "inactive": [second]})

    assert result == {"active": 5.0, "inactive": 0.0}
    assert torch.equal(first.grad, torch.tensor([3.0, 4.0]))
