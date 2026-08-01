import copy
from pathlib import Path

import torch

from fate_oia.models.meter_meta_adapters import HECASharedPrivateAdapters
from fate_oia.optim.heca_optimization import HECAScheduleState
from fate_oia.utils.meter_config import load_meter_config


def _action_grads(module: HECASharedPrivateAdapters, nodes: torch.Tensor, scale: float) -> dict[str, torch.Tensor]:
    module.zero_grad(set_to_none=True)
    inputs = nodes.detach().clone().requires_grad_(True)
    output = module(inputs, shared_action_gradient_scale=scale)
    output["action_nodes"].sum().backward()
    return {
        "shared_up": module.shared_adapter.up.weight.grad.detach().clone(),
        "private_up": module.action_private_adapter.up.weight.grad.detach().clone(),
        "input": inputs.grad.detach().clone(),
    }


def test_shared_gradient_scaling_preserves_forward_and_private_action_updates() -> None:
    torch.manual_seed(23)
    reference = HECASharedPrivateAdapters(dim=8, rank=4)
    with torch.no_grad():
        reference.shared_adapter.up.weight.normal_(mean=0.0, std=0.1)
    scaled = copy.deepcopy(reference)
    nodes = torch.randn(2, 25, 8)

    baseline_output = reference(nodes)
    scaled_output = scaled(nodes, shared_action_gradient_scale=0.25)
    torch.testing.assert_close(scaled_output["action_nodes"], baseline_output["action_nodes"])
    torch.testing.assert_close(scaled_output["reason_nodes"], baseline_output["reason_nodes"])

    baseline_grad = _action_grads(reference, nodes, scale=1.0)
    scaled_grad = _action_grads(scaled, nodes, scale=0.25)
    torch.testing.assert_close(scaled_grad["shared_up"], 0.25 * baseline_grad["shared_up"])
    torch.testing.assert_close(scaled_grad["private_up"], baseline_grad["private_up"])
    torch.testing.assert_close(scaled_grad["input"], baseline_grad["input"])


def test_next_window_shared_balance_is_checkpointed() -> None:
    state = HECAScheduleState(
        update=7,
        total_updates=80,
        shared_action_weight=0.63,
        shared_reason_weight=0.37,
    )
    restored = HECAScheduleState.from_state_dict(state.state_dict())
    assert restored.shared_action_weight == 0.63
    assert restored.shared_reason_weight == 0.37


def test_trainer_uses_one_backward_and_probe_only_owner_gradients() -> None:
    source = Path("fate_oia/engine/train_acpr_meter_oia.py").read_text(encoding="utf-8")
    assert source.count("scaled.backward()") == 1
    assert "shared_action_grads" not in source
    assert "shared_reason_grads" not in source
    assert source.index("if probe_due:") < source.index("action_shared =")
    assert "shared_action_gradient_scale=active_balance[\"action\"]" in source
    assert "shared_reason_gradient_scale=active_balance[\"reason\"]" in source
    assert 'probe_interval = int(config["training"]["gradient_ownership_probe_interval_updates"])' in source


def test_gate_g_memory_safe_configuration_is_the_only_supervisor_default() -> None:
    config = load_meter_config("configs/fate_oia_train_360x640_acpr_meter_oia_v3_heca.yaml")
    assert config["training"]["batch_size"] == 4
    assert config["training"]["gradient_accumulation_steps"] == 8
    assert config["training"]["effective_batch_size"] == 32
    supervisor = Path("fate_oia/engine/supervise_meter_oia_v3_heca_foreground.py").read_text(encoding="utf-8")
    assert "FALLBACK_LADDER = ((4, 8), (3, 10), (2, 15))" in supervisor
    assert "if requested != FALLBACK_LADDER[0]:" in supervisor
    assert "ladder = list(FALLBACK_LADDER)" in supervisor
    assert "(6, 5)" not in supervisor
    assert "(5, 6)" not in supervisor
