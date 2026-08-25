import pytest
import torch

from fate_oia.models.tida_object_intent_flow import TIDAObjectIntentTransport


def _inputs(batch=2, frames=6, tracks=8, dim=16):
    torch.manual_seed(17)
    xy = torch.rand(batch, frames, tracks, 2) * 2.0 - 1.0
    visibility = torch.ones(batch, frames, tracks, dtype=torch.bool)
    patches = torch.randn(batch, 20, dim)
    action_nodes = torch.randn(batch, 4, dim)
    reason_nodes = torch.randn(batch, 7, dim)
    return xy, visibility, patches, action_nodes, reason_nodes


def test_zero_init_is_exact_baseline_and_output_shapes_are_target_specific():
    model = TIDAObjectIntentTransport(
        dim=16, num_actions=4, num_reasons=7, heads=4,
        action_cap=0.08, reason_cap=0.06,
    )
    xy, visibility, patches, action_nodes, reason_nodes = _inputs()
    output = model(
        xy, visibility, patches, (4, 5), action_nodes, reason_nodes,
    )

    assert torch.equal(output["object_intent_action_delta"], torch.zeros(2, 4))
    assert torch.equal(output["object_intent_reason_delta"], torch.zeros(2, 7))
    assert output["object_intent_action_attention"].shape == (2, 4, 8)
    assert output["object_intent_reason_attention"].shape == (2, 7, 8)
    assert output["object_intent_future_xy"].shape == (2, 8, 4, 2)
    assert output["object_intent_action_selected_track"].shape == (2, 4)
    assert output["object_intent_reason_selected_track"].shape == (2, 7)


def test_first_backward_opens_output_path_without_cross_task_gradient():
    model = TIDAObjectIntentTransport(dim=16, num_actions=4, num_reasons=7, heads=4)
    inputs = _inputs()
    output = model(*inputs[:3], (4, 5), *inputs[3:])
    output["object_intent_action_candidate"][:, 0].sum().backward()

    assert model.action_output.weight.grad.abs().sum() > 0
    reason_parameters = list(model.reason_encoder.parameters()) + list(model.reason_output.parameters())
    assert all(parameter.grad is None or parameter.grad.abs().sum() == 0 for parameter in reason_parameters)


def test_invisible_tracks_cannot_receive_credit_and_time_reversal_changes_motion_evidence():
    model = TIDAObjectIntentTransport(dim=16, num_actions=4, num_reasons=7, heads=4)
    with torch.no_grad():
        model.action_output.weight.fill_(0.05)
    xy, visibility, patches, action_nodes, reason_nodes = _inputs()
    visibility[:, :, -2:] = False
    forward = model(xy, visibility, patches, (4, 5), action_nodes, reason_nodes)
    reverse = model(xy.flip(1), visibility.flip(1), patches, (4, 5), action_nodes, reason_nodes)

    assert torch.equal(forward["object_intent_action_attention"][..., -2:], torch.zeros(2, 4, 2))
    assert not torch.allclose(
        forward["object_intent_action_candidate"], reverse["object_intent_action_candidate"]
    )


def test_semantic_sampling_matches_terminal_patch_grid():
    patches = torch.arange(20.0).view(1, 20, 1)
    xy = torch.tensor([[[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]])

    sampled = TIDAObjectIntentTransport.sample_terminal_semantics(patches, (4, 5), xy)

    assert torch.allclose(sampled[0, :, 0], torch.tensor([0.0, 19.0, 9.5]))


def test_motion_features_use_real_elapsed_time_for_irregular_sampling():
    timestamps = torch.tensor([[-3.0, -1.0, -0.5, 0.0]])
    xy = torch.zeros(1, 4, 3, 2)
    # Two stationary tracks anchor ego compensation. The first track moves at
    # exactly 0.1 normalized image units per second despite irregular spacing.
    xy[0, :, 0, 0] = 0.1 * timestamps[0]
    visibility = torch.ones(1, 4, 3, dtype=torch.bool)

    geometry = TIDAObjectIntentTransport._ego_compensated_motion(
        xy, visibility, timestamps,
    )

    assert geometry["mean_velocity"][0, 0, 0].item() == pytest.approx(0.1, abs=1e-5)
    assert geometry["mean_velocity"][0, 1:].abs().max().item() == pytest.approx(0.0, abs=1e-6)
    # The 3-second future endpoint must use seconds, not an average per-frame displacement.
    assert geometry["future_xy"][0, 0, -1, 0].item() == pytest.approx(0.3, abs=1e-5)


def test_interaction_risk_is_ego_bottom_centered_and_future_aware():
    timestamps = torch.tensor([[-3.0, -2.0, -1.0, 0.0]])
    xy = torch.zeros(1, 4, 3, 2)
    # Track zero approaches the camera/ego anchor at normalized (0, 1).
    xy[0, :, 0, 1] = torch.tensor([-0.4, 0.0, 0.4, 0.8])
    # Track one moves away from that same anchor. Track two anchors camera motion.
    xy[0, :, 1, 1] = torch.tensor([0.2, -0.1, -0.4, -0.7])
    visibility = torch.ones(1, 4, 3, dtype=torch.bool)

    geometry = TIDAObjectIntentTransport._ego_compensated_motion(
        xy, visibility, timestamps,
    )

    assert torch.allclose(
        geometry["ego_relative_xy"][0, 0], torch.tensor([0.0, -0.2]), atol=1e-5
    )
    assert geometry["future_approach_risk"][0, 0] > geometry["future_approach_risk"][0, 1]
    assert geometry["interaction_risk"][0, 0] > geometry["interaction_risk"][0, 1]
    assert geometry["future_ego_distance"].shape == (1, 3, 4)


def test_delta_is_bounded_and_selected_deletion_is_target_specific():
    model = TIDAObjectIntentTransport(
        dim=16, num_actions=4, num_reasons=7, heads=4,
        action_cap=0.08, reason_cap=0.06,
    )
    with torch.no_grad():
        model.action_output.weight.fill_(100.0)
        model.reason_output.weight.fill_(100.0)
    output = model(*_inputs()[:3], (4, 5), *_inputs()[3:])

    assert output["object_intent_action_delta"].abs().max() <= 0.080001
    assert output["object_intent_reason_delta"].abs().max() <= 0.060001
    assert output["object_intent_action_selected_deleted_delta"].shape == (2, 4)
    assert output["object_intent_action_control_deleted_delta"].shape == (2, 4)
    assert torch.all(
        output["object_intent_action_selected_track"]
        != output["object_intent_action_control_track"]
    )


def test_multihot_action_corrections_are_not_forced_to_be_zero_sum():
    model = TIDAObjectIntentTransport(dim=16, num_actions=4, num_reasons=7, heads=4)
    with torch.no_grad():
        model.action_output.weight.normal_(mean=0.2, std=0.1)
    output = model(*_inputs()[:3], (4, 5), *_inputs()[3:])

    # BDD-OIA actions are multi-hot, so traffic may legitimately raise more
    # than one action. A zero-sum projection would impose false exclusivity.
    assert not torch.allclose(
        output["object_intent_action_candidate"].sum(-1),
        torch.zeros(output["object_intent_action_candidate"].shape[0]),
    )


def test_deployment_gates_are_persistent_and_train_calib_owned():
    model = TIDAObjectIntentTransport(dim=16, num_actions=4, num_reasons=7, heads=4)
    assert model.action_deploy_gate.tolist() == [0.0] * 4
    assert model.reason_deploy_gate.tolist() == [0.0] * 7

    action = torch.tensor([1.0, 0.0, 1.0, 0.0])
    reason = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    model.set_deployment_gates(action, reason, source="train_calib_epoch_2")
    state = model.state_dict()
    assert state["action_deploy_gate"].tolist() == action.tolist()
    assert state["reason_deploy_gate"].tolist() == reason.tolist()

    with pytest.raises(ValueError, match="train_calib"):
        model.set_deployment_gates(action, reason, source="test_oracle")


def test_closed_deployment_gate_does_not_cut_candidate_training_gradient():
    model = TIDAObjectIntentTransport(dim=16, num_actions=4, num_reasons=7, heads=4)
    output = model(*_inputs()[:3], (4, 5), *_inputs()[3:])
    assert output["object_intent_action_delta"].abs().sum() == 0

    loss = output["object_intent_action_candidate"][:, 0].sum()
    loss.backward()

    assert model.action_output.weight.grad.abs().sum() > 0


def test_harm_aware_utility_is_target_private_and_train_calib_deployed():
    model = TIDAObjectIntentTransport(dim=16, num_actions=4, num_reasons=7, heads=4)
    xy, visibility, patches, action_nodes, reason_nodes = _inputs()
    action_base = torch.randn(2, 4)
    reason_base = torch.randn(2, 7)
    output = model(
        xy, visibility, patches, (4, 5), action_nodes, reason_nodes,
        base_action_logits=action_base, base_reason_logits=reason_base,
    )

    assert output["object_intent_action_utility_logit"].shape == (2, 4)
    assert output["object_intent_reason_utility_logit"].shape == (2, 7)
    assert output["object_intent_action_utility_gate"].min() >= 0
    assert output["object_intent_action_utility_gate"].max() <= 1

    action_gate = torch.ones(4)
    reason_gate = torch.ones(7)
    model.set_deployment_policy(
        action_gate, reason_gate,
        action_scale=torch.full((4,), 64.0),
        reason_scale=torch.full((7,), 32.0),
        action_cutoff=torch.zeros(4), reason_cutoff=torch.zeros(7),
        source="train_calib_oof_epoch_2",
    )
    deployed = model(
        xy, visibility, patches, (4, 5), action_nodes, reason_nodes,
        base_action_logits=action_base, base_reason_logits=reason_base,
    )
    assert deployed["object_intent_action_delta"].abs().max() <= 0.080001
    assert deployed["object_intent_reason_delta"].abs().max() <= 0.060001

    deployed["object_intent_action_utility_logit"].sum().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.action_utility.parameters()
    )
    assert all(
        parameter.grad is None or parameter.grad.abs().sum() == 0
        for parameter in model.reason_utility.parameters()
    )
    with pytest.raises(ValueError, match="train_calib"):
        model.set_deployment_policy(
            action_gate, reason_gate,
            action_scale=torch.ones(4), reason_scale=torch.ones(7),
            action_cutoff=torch.zeros(4), reason_cutoff=torch.zeros(7),
            source="test_oracle",
        )


def test_static_and_future_motion_routes_are_decoupled_and_target_private():
    model = TIDAObjectIntentTransport(dim=16, num_actions=4, num_reasons=7, heads=4)
    with torch.no_grad():
        model.action_encoder.motion_mix.weight.zero_()
        model.action_encoder.motion_mix.bias.fill_(-20.0)
        model.reason_encoder.motion_mix.weight.zero_()
        model.reason_encoder.motion_mix.bias.fill_(-20.0)
    xy, visibility, patches, action_nodes, reason_nodes = _inputs()
    moving = xy.clone()
    moving[:, :, 0, 0] = torch.linspace(-0.8, 0.8, xy.shape[1])

    stationary_out = model(
        xy, visibility, patches, (4, 5), action_nodes, reason_nodes,
    )
    moving_out = model(
        moving, visibility, patches, (4, 5), action_nodes, reason_nodes,
    )

    for task, labels in (("action", 4), ("reason", 7)):
        semantic = moving_out[f"object_intent_{task}_semantic_attention"]
        motion = moving_out[f"object_intent_{task}_motion_attention"]
        mix = moving_out[f"object_intent_{task}_motion_mix"]
        assert semantic.shape == (2, labels, 8)
        assert motion.shape == (2, labels, 8)
        assert mix.shape == (2, labels)
        assert torch.all((mix >= 0.20) & (mix < 1))

    assert not torch.allclose(
        stationary_out["object_intent_action_motion_attention"],
        moving_out["object_intent_action_motion_attention"],
    )

    moving_out["object_intent_action_candidate"].sum().backward()
    reason_parameters = list(model.reason_encoder.parameters()) + list(model.reason_output.parameters())
    assert all(
        parameter.grad is None or parameter.grad.abs().sum() == 0
        for parameter in reason_parameters
    )


def test_motion_route_does_not_force_monotonic_interaction_risk_bias():
    model = TIDAObjectIntentTransport(dim=16, num_actions=4, num_reasons=7, heads=4)
    with torch.no_grad():
        model.action_encoder.motion_query.weight.zero_()
        model.action_encoder.motion_key.weight.zero_()
        for parameter in model.role_head.parameters():
            parameter.zero_()
    xy, visibility, patches, action_nodes, reason_nodes = _inputs(batch=1)
    # Two stationary tracks anchor camera motion. Track three approaches the
    # ego rapidly, but risk must remain evidence rather than a hard attention
    # prior shared by every action and reason target.
    xy.zero_()
    xy[0, :, 3, 1] = torch.linspace(-0.5, 0.8, xy.shape[1])
    xy[0, :, 1, 0] = -0.8
    xy[0, :, 2, 0] = 0.8

    output = model(xy, visibility, patches, (4, 5), action_nodes, reason_nodes)
    attention = output["object_intent_action_motion_attention"]
    assert output["object_intent_interaction_risk"][0, 3] > 0
    assert torch.allclose(
        attention,
        torch.full_like(attention, 1.0 / attention.shape[-1]),
        atol=1e-5,
    )


def test_pairwise_future_geometry_detects_crossing_not_parallel_motion():
    future = torch.zeros(1, 3, 4, 2)
    future[0, 0, :, 0] = torch.tensor([-0.6, -0.2, 0.2, 0.6])
    future[0, 1, :, 0] = torch.tensor([0.6, 0.2, -0.2, -0.6])
    future[0, 2, :, 0] = torch.tensor([-0.6, -0.2, 0.2, 0.6])
    future[0, 2, :, 1] = 0.8
    final_xy = future[:, :, 0]
    velocity = future[:, :, 1] - future[:, :, 0]

    pair = TIDAObjectIntentTransport._pairwise_future_geometry(
        final_xy, velocity, future,
    )

    assert pair["features"].shape[:3] == (1, 3, 3)
    assert pair["min_future_distance"][0, 0, 1] < pair["min_future_distance"][0, 0, 2]
    assert pair["distance_reduction"][0, 0, 1] > pair["distance_reduction"][0, 0, 2]


def test_pair_route_is_target_specific_diagonal_free_and_zero_init_compatible():
    model = TIDAObjectIntentTransport(dim=16, num_actions=4, num_reasons=7, heads=4)
    output = model(*_inputs()[:3], (4, 5), *_inputs()[3:])

    action_pair = output["object_intent_action_pair_attention"]
    reason_pair = output["object_intent_reason_pair_attention"]
    assert action_pair.shape == (2, 4, 8, 8)
    assert reason_pair.shape == (2, 7, 8, 8)
    diagonal = torch.arange(8)
    assert torch.equal(action_pair[..., diagonal, diagonal], torch.zeros(2, 4, 8))
    assert torch.equal(reason_pair[..., diagonal, diagonal], torch.zeros(2, 7, 8))
    assert torch.equal(output["object_intent_action_pair_candidate"], torch.zeros(2, 4))
    assert torch.equal(output["object_intent_reason_pair_candidate"], torch.zeros(2, 7))
    assert output["object_intent_action_selected_pair"].shape == (2, 4)
    assert output["object_intent_action_control_pair"].shape == (2, 4)
    assert torch.all(
        output["object_intent_action_selected_pair"]
        != output["object_intent_action_control_pair"]
    )
    assert output["object_intent_action_selected_pair_deleted_candidate"].shape == (2, 4)


def test_pair_output_first_backward_is_nonzero_and_task_private():
    model = TIDAObjectIntentTransport(dim=16, num_actions=4, num_reasons=7, heads=4)
    output = model(*_inputs()[:3], (4, 5), *_inputs()[3:])
    output["object_intent_action_pair_candidate"].sum().backward()

    assert model.action_pair_output.weight.grad.abs().sum() > 0
    reason_parameters = (
        list(model.reason_pair_encoder.parameters())
        + list(model.reason_pair_output.parameters())
    )
    assert all(
        parameter.grad is None or parameter.grad.abs().sum() == 0
        for parameter in reason_parameters
    )


def test_pair_deletion_reuses_values_and_exactly_renormalizes_remaining_mass():
    attention = torch.tensor([[[[0.0, 0.2, 0.3], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]]]])
    pair_values = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0],
                                 [0.0, 0.0], [0.0, 0.0], [3.0, 1.0],
                                 [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    support = torch.tensor([[[0.0, 1.0, 0.5], [0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0]]])
    flat_attention = attention.flatten(2)
    evidence = torch.einsum("blp,bpd->bld", flat_attention, pair_values)
    transported_support = torch.einsum("blp,bp->bl", flat_attention, support.flatten(1))
    output = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        output.weight.copy_(torch.tensor([[1.0, 0.0]]))

    actual = TIDAObjectIntentTransport._deleted_pair_candidate(
        output, 0.08, evidence, attention, transported_support,
        pair_values[:, [1, 2, 5]], torch.tensor([[1, 2, 5]]), support,
        torch.tensor([[5]]),
    )
    remaining_attention = flat_attention.clone()
    remaining_attention[..., 5] = 0
    remaining_attention /= remaining_attention.sum(-1, keepdim=True)
    expected_evidence = torch.einsum("blp,bpd->bld", remaining_attention, pair_values)
    expected_support = torch.einsum("blp,bp->bl", remaining_attention, support.flatten(1))
    expected = 0.04 * expected_support * torch.tanh(output(expected_evidence).squeeze(-1))

    assert torch.allclose(actual, expected, atol=1e-6)
