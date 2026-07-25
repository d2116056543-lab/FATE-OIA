"""P8 contracts: action-safe semantic bridging without reason supervision inputs."""

from __future__ import annotations

from collections.abc import Mapping
import gc
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]


def _bridge_module():
    spec = importlib.util.find_spec("fate_oia.models.rael_action_reason_bridge")
    assert spec is not None, "P8 bridge must exist before its contracts can run"
    return importlib.import_module("fate_oia.models.rael_action_reason_bridge")


def _foundation():
    from fate_oia.models.rael_category_foundation import RAELActionCategoryFoundation

    return RAELActionCategoryFoundation(dim=384, num_heads=6)


def _inputs(*, batch: int = 2, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
    torch.manual_seed(29)
    action = torch.randn(batch, 4, 384, device=device, dtype=dtype, requires_grad=True)
    semantic = torch.randn(batch, 21, 384, device=device, dtype=dtype, requires_grad=True)
    return action, semantic


def _grad_is_nonzero(value: torch.Tensor | None) -> bool:
    return value is not None and bool(torch.isfinite(value).all()) and bool(value.abs().sum() > 0)


class ActionFirewallViolation(AssertionError):
    """Carries a complete owner-gradient matrix when a forbidden route appears."""

    def __init__(self, loss_name: str, expected: Mapping[str, str], actual: Mapping[str, float]) -> None:
        super().__init__(f"{loss_name} firewall failed; expected={dict(expected)} actual={dict(actual)}")
        self.loss_name = loss_name
        self.expected = dict(expected)
        self.actual = dict(actual)


def _owner_parameters(owner: Tensor | nn.Module | tuple[Tensor | nn.Module, ...]) -> tuple[Tensor, ...]:
    if isinstance(owner, Tensor):
        return (owner,)
    if isinstance(owner, nn.Module):
        return tuple(owner.parameters())
    parameters: list[Tensor] = []
    for item in owner:
        parameters.extend(_owner_parameters(item))
    return tuple(parameters)


def _clear_owner_gradients(owner_tensors: Mapping[str, Tensor | nn.Module | tuple[Tensor | nn.Module, ...]]) -> None:
    for owner in owner_tensors.values():
        for parameter in _owner_parameters(owner):
            parameter.grad = None


def _owner_gradient_norms(owner_tensors: Mapping[str, Tensor | nn.Module | tuple[Tensor | nn.Module, ...]]) -> dict[str, float]:
    norms: dict[str, float] = {}
    for name, owner in owner_tensors.items():
        squared_norm = 0.0
        for parameter in _owner_parameters(owner):
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), f"{name} gradient must be finite"
                squared_norm += float(parameter.grad.detach().square().sum().item())
        norms[name] = squared_norm**0.5
    return norms


def assert_action_firewall(
    owner_tensors: Mapping[str, Tensor | nn.Module | tuple[Tensor | nn.Module, ...]],
    losses: Mapping[str, Tensor],
) -> dict[str, dict[str, float]]:
    """Verify branch-split owner gradients without relying on disconnected constants."""

    required_losses = {"action", "reason", "private"}
    if set(losses) != required_losses:
        raise ValueError(f"losses must be exactly {sorted(required_losses)}")
    required_owners = {"action_visual", "semantic", "bridge", "p7", "private"}
    if set(owner_tensors) != required_owners:
        raise ValueError(f"owner_tensors must be exactly {sorted(required_owners)}")

    expected = {
        "action": {"action_visual": ">0", "semantic": ">0", "bridge": ">0", "p7": ">0", "private": "=0"},
        "reason": {"semantic": ">0", "private": ">0", "action_visual": "=0", "bridge": "=0", "p7": "=0"},
        "private": {"semantic": ">0", "private": ">0", "action_visual": "=0", "bridge": "=0", "p7": "=0"},
    }
    matrix: dict[str, dict[str, float]] = {}
    for loss_name in ("action", "reason", "private"):
        _clear_owner_gradients(owner_tensors)
        losses[loss_name].backward(retain_graph=True)
        norms = _owner_gradient_norms(owner_tensors)
        matrix[loss_name] = norms
        for owner_name, requirement in expected[loss_name].items():
            valid = norms[owner_name] > 1.0e-12 if requirement == ">0" else norms[owner_name] <= 1.0e-12
            if not valid:
                raise ActionFirewallViolation(loss_name, expected[loss_name], norms)
    _clear_owner_gradients(owner_tensors)
    return matrix


class _SharedSemanticProducer(nn.Module):
    """A trainable shared image-semantic producer used to exercise branch ownership."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(384, 384, bias=False)

    def forward(self, image_semantic_seed: Tensor) -> Tensor:
        return torch.tanh(self.projection(image_semantic_seed))


def _firewall_fixture(
    *,
    reason_action_read: str | None = None,
    private_action_read: str | None = None,
) -> tuple[dict[str, Tensor | nn.Module | tuple[Tensor | nn.Module, ...]], dict[str, Tensor]]:
    """Build shared semantics plus isolated action/reason/private branches.

    The semantic producer is intentionally shared.  Only the action branch
    reads the bridge output; reason/private branches read the shared producer
    directly, so their gradients remain useful without touching action owners.
    """

    module = _bridge_module()
    torch.manual_seed(311)
    foundation = _foundation()
    bridge = module.RAELActionReasonBridge(dim=384, num_heads=6)
    with torch.no_grad():
        bridge.gamma_as_raw.fill_(1.0)
    semantic_producer = _SharedSemanticProducer()
    reason_private_head = nn.Linear(384, 21)
    private_offset = nn.Parameter(torch.randn(384))
    action_visual = torch.randn(2, 4, 384, requires_grad=True)
    image_semantic_seed = torch.randn(2, 21, 384)
    shared_semantic = semantic_producer(image_semantic_seed)
    action_output = bridge(action_visual, shared_semantic, foundation)

    action_loss = action_output["z_A_global"].square().mean()
    reason_loss = reason_private_head(shared_semantic.mean(dim=1)).square().mean()
    private_loss = (shared_semantic.mean(dim=1) + private_offset).square().mean()
    if reason_action_read == "global":
        reason_loss = reason_loss + 0.1 * action_output["z_A_global"].square().mean()
    elif reason_action_read == "bridged":
        reason_loss = reason_loss + 0.1 * action_output["action_bridged_tokens"].square().mean()
    if private_action_read == "global":
        private_loss = private_loss + 0.1 * action_output["z_A_global"].square().mean()
    elif private_action_read == "bridged":
        private_loss = private_loss + 0.1 * action_output["action_bridged_tokens"].square().mean()

    return (
        {
            "action_visual": action_visual,
            "semantic": semantic_producer,
            "bridge": bridge,
            "p7": foundation,
            "private": (reason_private_head, private_offset),
        },
        {"action": action_loss, "reason": reason_loss, "private": private_loss},
    )


def test_p8_has_action_safe_public_contract_without_reason_supervision_inputs() -> None:
    module = _bridge_module()
    bridge = module.RAELActionReasonBridge(dim=384, num_heads=6)
    parameters = set(inspect.signature(bridge.forward).parameters)
    forbidden = {
        "reason_logits",
        "reason_labels",
        "reason_targets",
        "reason_gt",
        "reason_private",
        "private_reason_tokens",
        "pu_state",
        "posterior",
    }
    assert forbidden.isdisjoint(parameters)
    assert parameters == {"action_visual_tokens", "semantic_reason_tokens", "action_foundation"}
    assert bridge.parameter_owner == "action_reason_bridge"
    assert bridge.learning_rate == pytest.approx(2.0e-4)
    source = inspect.getsource(module)
    for forbidden_text in ("reason_logits", "reason_labels", "private_reason", "pu_state", "posterior"):
        assert forbidden_text not in source


def test_zero_init_is_exact_p7_equivalence_and_only_project_global_makes_formal_logits() -> None:
    module = _bridge_module()
    foundation = _foundation()
    bridge = module.RAELActionReasonBridge(dim=384, num_heads=6)
    action, semantic = _inputs()
    original_project_global = foundation.project_global
    calls = 0

    def counted_project_global(tokens: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_project_global(tokens)

    object.__setattr__(foundation, "project_global", counted_project_global)
    output = bridge(action, semantic, foundation)

    assert output["action_bridged_tokens"].shape == (2, 4, 384)
    assert output["z_A_global"].shape == (2, 4)
    assert torch.equal(output["action_bridged_tokens"], action)
    assert torch.equal(output["z_A_global"], original_project_global(action))
    assert calls == 1
    assert float(output["gamma_AS"].item()) == 0.0
    assert set(output) >= {
        "action_bridged_tokens",
        "z_A_global",
        "gamma_AS",
        "attention_weights",
        "diagnostics",
    }
    assert output["attention_weights"].shape == (2, 6, 4, 21)
    assert output["diagnostics"]["bridge_rms"].shape == (2,)
    assert output["diagnostics"]["global_rms"].shape == (2,)
    assert output["diagnostics"]["attention_entropy"].shape == (2, 4)


def test_gamma_is_bounded_and_semantic_shuffle_changes_bridged_action_after_activation() -> None:
    module = _bridge_module()
    foundation = _foundation()
    bridge = module.RAELActionReasonBridge(dim=384, num_heads=6)
    action, semantic = _inputs()
    with torch.no_grad():
        bridge.gamma_as_raw.fill_(1.0e6)
    output = bridge(action, semantic, foundation)
    assert abs(float(output["gamma_AS"].item())) <= 0.2500001

    with torch.no_grad():
        bridge.gamma_as_raw.fill_(1.0)
    direct = bridge(action, semantic, foundation)["action_bridged_tokens"]
    shuffled = bridge(action, semantic.roll(shifts=1, dims=0), foundation)["action_bridged_tokens"]
    assert not torch.allclose(direct, action)
    assert not torch.allclose(direct, shuffled)


def test_diagnostics_detach_without_breaking_formal_action_backward_or_p16_contract() -> None:
    module = _bridge_module()
    foundation = _foundation()
    bridge = module.RAELActionReasonBridge(dim=384, num_heads=6)
    action, semantic = _inputs()
    output = bridge(action, semantic, foundation)

    assert output["action_bridged_tokens"].requires_grad
    assert output["z_A_global"].requires_grad
    diagnostic_tensors = (output["attention_weights"], output["gamma_AS"], *output["diagnostics"].values())
    assert all(not tensor.requires_grad and tensor.grad_fn is None for tensor in diagnostic_tensors)
    optimizer = torch.optim.SGD(bridge.parameters(), lr=1.0)
    output["z_A_global"].square().mean().backward()
    assert _grad_is_nonzero(bridge.gamma_as_raw.grad)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    foundation.zero_grad(set_to_none=True)
    action_two, semantic_two = _inputs()
    second_output = bridge(action_two, semantic_two, foundation)
    second_diagnostics = (
        second_output["attention_weights"],
        second_output["gamma_AS"],
        *second_output["diagnostics"].values(),
    )
    assert all(not tensor.requires_grad and tensor.grad_fn is None for tensor in second_diagnostics)
    second_output["z_A_global"].square().mean().backward()
    assert _grad_is_nonzero(bridge.cross_attention.in_proj_weight.grad)

    assert module.P16_OWNER_MATRIX_INTEGRATION_STATUS == "deferred_to_p16"
    required = set(module.REQUIRED_P16_FIREWALL_CHECKS)
    assert any("P4" in item for item in required)
    assert any("P11" in item for item in required)
    assert any("assembled model" in item for item in required)


def test_same_version_behavior_artifacts_have_runtime_schema_and_hashes() -> None:
    audit = ROOT / ".review" / "P8_behavior_audit.py"
    completed = subprocess.run([sys.executable, str(audit)], cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    source_hash = __import__("hashlib").sha256((ROOT / "fate_oia/models/rael_action_reason_bridge.py").read_bytes()).hexdigest()
    test_hash = __import__("hashlib").sha256((ROOT / "tests/test_rael_action_reason_firewall.py").read_bytes()).hexdigest()
    harness_hash = __import__("hashlib").sha256(audit.read_bytes()).hexdigest()
    required = {
        "schema_version",
        "command",
        "test_nodes",
        "git_head",
        "timestamp_utc",
        "python",
        "torch",
        "cuda",
        "device",
        "base_production_sha256",
        "base_test_sha256",
        "base_harness_sha256",
    }
    for name in ("P8_BEHAVIOR_RED.json", "P8_BEHAVIOR_GREEN.json"):
        payload = json.loads((ROOT / ".review" / name).read_text(encoding="utf-8"))
        assert payload["pass"] is True
        assert required.issubset(payload)
        assert payload["production_sha256"] == payload["base_production_sha256"] == source_hash
        assert payload["test_sha256"] == payload["base_test_sha256"] == test_hash
        assert payload["harness_sha256"] == payload["base_harness_sha256"] == harness_hash
        assert payload["git_head"]
        assert payload["timestamp_utc"].endswith("Z")
        assert "test_rael_action_reason_firewall.py" in " ".join(payload["test_nodes"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for P8 diagnostic-retention probe")
def test_cuda_diagnostics_retention_does_not_retain_training_graph() -> None:
    module = _bridge_module()
    device = torch.device("cuda")
    foundation = _foundation().to(device=device)
    bridge = module.RAELActionReasonBridge(dim=384, num_heads=6).to(device=device)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    archived: list[tuple[Tensor, Tensor, tuple[Tensor, ...]]] = []

    for _ in range(200):
        action, semantic = _inputs(batch=1, device=device)
        output = bridge(action, semantic, foundation)
        diagnostics = tuple(output["diagnostics"].values())
        retained = (output["attention_weights"], output["gamma_AS"], diagnostics)
        assert not retained[0].requires_grad and retained[0].grad_fn is None
        assert not retained[1].requires_grad and retained[1].grad_fn is None
        assert all(not value.requires_grad and value.grad_fn is None for value in retained[2])
        archived.append(retained)
        del output, action, semantic

    gc.collect()
    torch.cuda.synchronize(device)
    retained_growth = torch.cuda.memory_allocated(device) - baseline
    # 200 detached attention maps/diagnostic scalars occupy far below this.
    # Retaining 200 autograd graphs would exceed the budget by orders of magnitude.
    assert retained_growth < 48 * 1024 * 1024
    assert len(archived) == 200


def test_bootstrap_gradients_and_action_firewall_are_real() -> None:
    module = _bridge_module()
    foundation = _foundation()
    bridge = module.RAELActionReasonBridge(dim=384, num_heads=6)
    action, semantic = _inputs()
    optimizer = torch.optim.SGD(bridge.parameters(), lr=1.0)

    first = bridge(action, semantic, foundation)
    first_loss = first["z_A_global"].square().mean()
    first_loss.backward()
    assert _grad_is_nonzero(bridge.gamma_as_raw.grad)
    assert _grad_is_nonzero(action.grad)
    assert _grad_is_nonzero(foundation.global_head.weight.grad)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    foundation.zero_grad(set_to_none=True)

    action_two, semantic_two = _inputs()
    second = bridge(action_two, semantic_two, foundation)
    second["z_A_global"].sum().backward()
    assert _grad_is_nonzero(bridge.cross_attention.in_proj_weight.grad)
    assert _grad_is_nonzero(bridge.cross_attention.out_proj.weight.grad)
    assert _grad_is_nonzero(semantic_two.grad)

    bridge.zero_grad(set_to_none=True)
    foundation.zero_grad(set_to_none=True)
    private_reason_owner = torch.randn(2, 21, 384, requires_grad=True)
    (private_reason_owner.square().mean()).backward()
    assert all(parameter.grad is None for parameter in bridge.parameters())
    assert all(parameter.grad is None for parameter in foundation.parameters())
    assert _grad_is_nonzero(private_reason_owner.grad)


def test_shared_semantic_owner_gradient_matrix_keeps_reason_and_private_losses_off_action_path() -> None:
    owners, losses = _firewall_fixture()
    matrix = assert_action_firewall(owners, losses)
    assert matrix["action"]["semantic"] > 1.0e-12
    assert matrix["action"]["bridge"] > 1.0e-12
    assert matrix["action"]["p7"] > 1.0e-12
    assert matrix["reason"]["semantic"] > 1.0e-12
    assert matrix["reason"]["private"] > 1.0e-12
    assert matrix["private"]["semantic"] > 1.0e-12
    assert matrix["private"]["private"] > 1.0e-12
    for loss_name in ("reason", "private"):
        assert matrix[loss_name]["action_visual"] <= 1.0e-12
        assert matrix[loss_name]["bridge"] <= 1.0e-12
        assert matrix[loss_name]["p7"] <= 1.0e-12


@pytest.mark.parametrize(
    ("reason_action_read", "private_action_read", "loss_name"),
    (
        ("global", None, "reason"),
        ("bridged", None, "reason"),
        (None, "global", "private"),
        (None, "bridged", "private"),
    ),
)
def test_firewall_red_detects_reason_or_private_callsite_that_reads_action_branch(
    reason_action_read: str | None,
    private_action_read: str | None,
    loss_name: str,
) -> None:
    owners, losses = _firewall_fixture(
        reason_action_read=reason_action_read,
        private_action_read=private_action_read,
    )
    with pytest.raises(ActionFirewallViolation) as captured:
        assert_action_firewall(owners, losses)
    assert captured.value.loss_name == loss_name
    assert captured.value.actual["action_visual"] > 1.0e-12
    assert captured.value.actual["bridge"] > 1.0e-12
    if reason_action_read == "global" or private_action_read == "global":
        assert captured.value.actual["p7"] > 1.0e-12


def test_rejects_shape_device_and_unapproved_keyword_paths() -> None:
    module = _bridge_module()
    foundation = _foundation()
    bridge = module.RAELActionReasonBridge(dim=384, num_heads=6)
    action, semantic = _inputs()
    with pytest.raises(ValueError, match="action_visual_tokens"):
        bridge(action[:, :3], semantic, foundation)
    with pytest.raises(ValueError, match="semantic_reason_tokens"):
        bridge(action, semantic[:, :20], foundation)
    with pytest.raises(TypeError):
        bridge(action, semantic, foundation, reason_logits=torch.zeros(2, 21))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for RAEL bf16 bootstrap probe")
def test_cuda_bf16_two_update_bootstrap_is_finite() -> None:
    module = _bridge_module()
    device = torch.device("cuda")
    foundation = _foundation().to(device=device, dtype=torch.bfloat16)
    bridge = module.RAELActionReasonBridge(dim=384, num_heads=6).to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.SGD(bridge.parameters(), lr=1.0)

    for _ in range(2):
        action, semantic = _inputs(batch=1, device=device, dtype=torch.bfloat16)
        output = bridge(action, semantic, foundation)
        loss = output["z_A_global"].float().square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(output["z_A_global"]).all()
        assert torch.isfinite(bridge.gamma_as_raw).all()
    assert _grad_is_nonzero(bridge.cross_attention.in_proj_weight.grad)
