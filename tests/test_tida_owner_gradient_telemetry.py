import torch

from fate_oia.engine.train_tida_oia import (
    module_state_sha256,
    owner_gradient_norms,
    owner_parameter_snapshots,
    owner_parameter_update_norms,
)


def test_module_state_sha256_is_stable_for_equal_state() -> None:
    module = torch.nn.Sequential(
        torch.nn.Linear(5, 3),
        torch.nn.BatchNorm1d(3),
    )
    clone = torch.nn.Sequential(
        torch.nn.Linear(5, 3),
        torch.nn.BatchNorm1d(3),
    )
    clone.load_state_dict(module.state_dict())

    assert module_state_sha256(module) == module_state_sha256(module)
    assert module_state_sha256(module) == module_state_sha256(clone)


def test_module_state_sha256_includes_dtype_and_shape() -> None:
    float_module = torch.nn.Module()
    byte_module = torch.nn.Module()
    float_module.register_buffer("state", torch.tensor([0.0], dtype=torch.float32))
    byte_module.register_buffer("state", torch.tensor([0, 0, 0, 0], dtype=torch.uint8))

    assert module_state_sha256(float_module) != module_state_sha256(byte_module)


def test_module_state_sha256_ignores_verified_dino_vproj_alias() -> None:
    class LazyAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(3, 3)
            self.vproj = None

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.vproj = self.proj
            return self.proj(value)

    module = torch.nn.Module()
    module.attn = LazyAttention()
    before = module_state_sha256(module)

    module.attn(torch.ones(1, 3))

    assert module_state_sha256(module) == before


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


def test_owner_update_norms_report_zero_for_empty_frozen_owner():
    owners = {"frozen": []}
    before = owner_parameter_snapshots(owners)
    assert owner_parameter_update_norms(owners, before) == {"frozen": 0.0}
