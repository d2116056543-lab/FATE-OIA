import torch
import yaml

from fate_oia.losses.tida_loss_registry import TIDALossRegistry


def test_each_required_loss_is_registered_exactly_once():
    registry = TIDALossRegistry()
    for name in registry.required_terms:
        registry.add(name, torch.tensor(1.0, requires_grad=True))
    assert set(registry.rows) == set(registry.required_terms)
    try:
        registry.add(registry.required_terms[0], torch.tensor(1.0))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate loss term was accepted")


def test_required_loss_names_and_yaml_weights_match_strict_contract():
    expected = (
        "terminal_hist", "terminal_no_history", "terminal_gain", "temporal_order", "repeated_last_contrast",
        "flow_transition_align",
        "action_asl", "action_smooth_ap", "action_base_protect", "action_delta", "action_route_sparse",
        "action_flow_credit", "action_flow_no_harm",
        "action_utility_calibration",
        "geometric_action_aux", "geometric_action_rank", "geometric_action_prefix", "geometric_action_delta",
        "traffic_action_aux", "traffic_action_rank", "traffic_action_delta",
        "trajectory_action_boundary", "trajectory_action_rank", "trajectory_selected_control",
        "trajectory_utility_calibration", "trajectory_delta",
        "reason_partial", "reason_rank", "reason_soft_f1", "reason_delta",
        "reason_flow_credit", "reason_flow_no_harm", "reason_positive_no_harm",
        "reason_utility_calibration",
        "geometric_reason_aux", "geometric_reason_rank", "geometric_reason_prefix", "geometric_reason_delta",
    )
    config = yaml.safe_load(open("configs/fate_oia_train_tida_oia_v1_15f.yaml", encoding="utf-8"))
    assert TIDALossRegistry.required_terms == expected
    assert set(config["loss"]) == set(expected)


def test_custom_loss_weights_are_used_by_total():
    weights = {name: 0.0 for name in TIDALossRegistry.required_terms}
    weights["terminal_hist"] = 2.5
    registry = TIDALossRegistry(weights)
    for name in registry.required_terms:
        registry.add(name, torch.tensor(2.0))
    assert registry.total().item() == 5.0


def test_future_pair_deletion_losses_are_registered_trainable_terms():
    assert "object_intent_action_pair_deletion" in TIDALossRegistry.optional_terms
    assert "object_intent_reason_pair_deletion" in TIDALossRegistry.optional_terms
    assert "object_intent_action_utility" in TIDALossRegistry.optional_terms
    assert "object_intent_reason_utility" in TIDALossRegistry.optional_terms
    assert TIDALossRegistry.default_weights["object_intent_action_pair_deletion"] >= 0.0
    assert TIDALossRegistry.default_weights["object_intent_reason_pair_deletion"] >= 0.0
