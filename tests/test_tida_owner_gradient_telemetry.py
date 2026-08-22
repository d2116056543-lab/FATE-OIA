import torch

from fate_oia.engine.train_tida_oia import (
    owner_gradient_norms,
    owner_parameter_snapshots,
    owner_parameter_update_norms,
)


def test_owner_gradient_norms_report_each_owner_without_changing_gradients():
    first = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    second = torch.nn.Parameter(torch.tensor([1.0]))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = None

    result = owner_gradient_norms({"active": [first], "inactive": [second]})

    assert result == {"active": 5.0, "inactive": 0.0}
    assert torch.equal(first.grad, torch.tensor([3.0, 4.0]))


def test_owner_parameter_update_norms_measure_real_step_delta():
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0]))
    owners = {"first": [first], "second": [second]}
    before = owner_parameter_snapshots(owners)

    with torch.no_grad():
        first.add_(torch.tensor([3.0, 4.0]))

    result = owner_parameter_update_norms(owners, before)
    assert result == {"first": 5.0, "second": 0.0}
