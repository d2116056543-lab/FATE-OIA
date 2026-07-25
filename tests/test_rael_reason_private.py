"""P11 behavioral contracts for the action-safe RAEL reason-private adapter."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import pytest
import torch


def _module() -> Any:
    spec = importlib.util.find_spec("fate_oia.models.rael_reason_private")
    assert spec is not None, "P11 reason-private module must exist before its contracts can run"
    return importlib.import_module("fate_oia.models.rael_reason_private")


def _inputs(
    *, batch: int = 2, dtype: torch.dtype = torch.float32, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(211)
    semantic = torch.randn(batch, 21, 384, dtype=dtype, device=device, requires_grad=True)
    action_tokens = torch.randn(batch, 4, 384, dtype=dtype, device=device, requires_grad=True)
    logits = torch.tensor(
        [[-1.5, 0.7, 1.2, -0.3], [0.1, -0.7, 0.4, 1.1]], dtype=dtype, device=device
    )
    return semantic, action_tokens, logits[:batch].clone().detach().requires_grad_(True)


def _configured(module: Any) -> Any:
    adapter = module.RAELReasonPrivateAdapter()
    with torch.no_grad():
        adapter.private_up.weight.zero_()
        adapter.private_up.bias.zero_()
        adapter.action_context_projection.weight.copy_(torch.eye(384))
        adapter.action_context_projection.bias.zero_()
        adapter.reason_global_head.weight.copy_(torch.linspace(-0.2, 0.2, 21 * 384).view(21, 384))
        adapter.reason_global_head.bias.copy_(torch.linspace(-0.1, 0.1, 21))
    return adapter


def test_p11_component_omission_is_a_normal_assertion_not_collection_error() -> None:
    _module()


def test_formal_formula_uses_sigmoid_weighted_fully_detached_action_context() -> None:
    module = _module()
    adapter = _configured(module)
    semantic, action_tokens, action_logits = _inputs()
    with torch.no_grad():
        adapter.gamma_ra_raw.fill_(torch.atanh(torch.tensor(0.5)))
    output = adapter(semantic, action_tokens, action_logits)
    gamma = 0.25 * torch.tanh(adapter.gamma_ra_raw)
    expected_h = torch.einsum("ba,bad->bd", torch.sigmoid(action_logits), action_tokens).detach()
    expected = semantic + gamma * expected_h.unsqueeze(1)
    expected_logits = torch.einsum("brd,rd->br", expected, adapter.reason_global_head.weight) + adapter.reason_global_head.bias
    assert output["formal_reason_tokens"].shape == (2, 21, 384)
    assert output["z_R_global"].shape == (2, 21)
    assert torch.allclose(output["action_context"], expected_h)
    assert torch.allclose(output["formal_reason_tokens"], expected)
    assert torch.allclose(output["z_R_global"], expected_logits)
    assert not output["action_context"].requires_grad
    assert output["z_R_global"].requires_grad


def test_private_rank_and_per_reason_norm_cap_are_active() -> None:
    module = _module()
    adapter = _configured(module)
    semantic, action_tokens, action_logits = _inputs()
    with torch.no_grad():
        adapter.private_down.weight.fill_(0.35)
        adapter.private_down.bias.fill_(0.1)
        adapter.private_up.weight.fill_(0.75)
        adapter.private_up.bias.fill_(0.5)
    output = adapter(semantic, action_tokens, action_logits)
    private_norm = output["private_delta"].norm(dim=-1)
    semantic_norm = semantic.detach().norm(dim=-1)
    assert adapter.private_down.out_features == 64
    assert adapter.private_up.in_features == 64
    assert bool((private_norm <= 0.4 * semantic_norm + 1.0e-5).all())
    assert float(output["diagnostics"]["private_norm_ratio"].max()) <= 0.40001


def test_reason_loss_updates_semantic_and_private_but_has_zero_action_gradients() -> None:
    module = _module()
    adapter = _configured(module)
    semantic, action_tokens, action_logits = _inputs()
    with torch.no_grad():
        adapter.gamma_ra_raw.fill_(0.15)
    adapter(semantic, action_tokens, action_logits)["z_R_global"].square().mean().backward()
    assert semantic.grad is not None and float(semantic.grad.norm()) > 0.0
    # This controlled formula fixture zeros the up projection, so first-step
    # private learning is correctly carried by ``private_up``.
    assert adapter.private_up.weight.grad is not None and float(adapter.private_up.weight.grad.norm()) > 0.0
    assert action_tokens.grad is None or float(action_tokens.grad.norm()) == 0.0
    assert action_logits.grad is None or float(action_logits.grad.norm()) == 0.0


def test_private_bootstrap_updates_private_path_on_step_zero_and_action_projection_on_step_one() -> None:
    module = _module()
    torch.manual_seed(307)
    adapter = module.RAELReasonPrivateAdapter()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.04)
    semantic, action_tokens, action_logits = _inputs()
    first = adapter(semantic, action_tokens, action_logits)
    first["z_R_global"].mul(torch.linspace(-0.7, 0.8, 21).view(1, 21)).mean().backward()
    assert adapter.gamma_ra_raw.grad is not None and float(adapter.gamma_ra_raw.grad.abs()) > 0.0
    assert adapter.private_up.weight.grad is not None and float(adapter.private_up.weight.grad.norm()) > 0.0
    assert adapter.action_context_projection.weight.grad is None or float(adapter.action_context_projection.weight.grad.norm()) == 0.0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = adapter(
        semantic.detach().clone().requires_grad_(True),
        action_tokens.detach().clone().requires_grad_(True),
        action_logits.detach().clone().requires_grad_(True),
    )
    second["z_R_global"].mul(torch.linspace(0.8, -0.7, 21).view(1, 21)).mean().backward()
    assert adapter.action_context_projection.weight.grad is not None
    assert float(adapter.action_context_projection.weight.grad.norm()) > 0.0


def test_action_inputs_and_api_are_private_invariant_and_no_action_logit_is_exported() -> None:
    module = _module()
    adapter = _configured(module)
    semantic, action_tokens, action_logits = _inputs()
    action_before, token_before = action_logits.detach().clone(), action_tokens.detach().clone()
    baseline = adapter(semantic, action_tokens, action_logits)
    with torch.no_grad():
        adapter.private_up.weight.add_(0.9)
    changed = adapter(semantic, action_tokens, action_logits)
    assert torch.equal(action_logits.detach(), action_before)
    assert torch.equal(action_tokens.detach(), token_before)
    assert not torch.allclose(baseline["formal_reason_tokens"], changed["formal_reason_tokens"])
    assert "action_logits" not in baseline and "z_A_global" not in baseline


def test_rejects_bad_shapes_and_returns_only_detached_diagnostics() -> None:
    module = _module()
    adapter = _configured(module)
    semantic, action_tokens, action_logits = _inputs()
    with pytest.raises(ValueError, match="semantic_reason_tokens"):
        adapter(semantic[:, :20], action_tokens, action_logits)
    with pytest.raises(ValueError, match="action_bridged_tokens"):
        adapter(semantic, action_tokens[:, :3], action_logits)
    with pytest.raises(ValueError, match="final_action_logits"):
        adapter(semantic, action_tokens, action_logits.unsqueeze(-1))
    for value in adapter(semantic, action_tokens, action_logits)["diagnostics"].values():
        assert torch.is_tensor(value) and not value.requires_grad and value.grad_fn is None
        assert torch.isfinite(value).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the P11 bf16 cap edge cases")
def test_bf16_private_cap_and_diagnostics_are_exact_for_normal_small_tiny_and_zero_semantics() -> None:
    module = _module()
    adapter = module.RAELReasonPrivateAdapter().to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        adapter.private_down.weight.fill_(0.6)
        adapter.private_down.bias.fill_(0.2)
        adapter.private_up.weight.fill_(0.8)
        adapter.private_up.bias.fill_(0.5)

    for value in (1.0, 1.0e-2, 1.0e-5, 0.0):
        semantic = torch.full((2, 21, 384), value, device="cuda", dtype=torch.bfloat16)
        action_tokens = torch.randn(2, 4, 384, device="cuda", dtype=torch.bfloat16)
        action_logits = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
        output = adapter(semantic, action_tokens, action_logits)
        semantic_norm = semantic.float().norm(dim=-1)
        private_norm = output["private_delta"].float().norm(dim=-1)
        actual_ratio = torch.where(
            semantic_norm > 0,
            private_norm / semantic_norm,
            torch.zeros_like(private_norm),
        )
        reported_ratio = output["diagnostics"]["private_norm_ratio"].float()
        assert torch.isfinite(output["private_delta"]).all()
        assert torch.isfinite(reported_ratio).all()
        assert float(actual_ratio.max()) <= 0.4 + 1.0e-6
        assert float((reported_ratio - actual_ratio).abs().max()) < 1.0e-4
        if value == 0.0:
            assert torch.count_nonzero(output["private_delta"]) == 0
            assert torch.count_nonzero(reported_ratio) == 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the P11 bf16 two-step probe")
def test_cuda_bf16_two_step_and_detached_diagnostic_retention() -> None:
    module = _module()
    adapter = module.RAELReasonPrivateAdapter().to(device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.02)
    retained = []
    for _ in range(2):
        semantic, action_tokens, action_logits = _inputs(dtype=torch.bfloat16, device="cuda")
        output = adapter(semantic, action_tokens, action_logits)
        output["z_R_global"].float().square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        retained.extend(output["diagnostics"].values())
    assert all(value.grad_fn is None and torch.isfinite(value).all() for value in retained)
