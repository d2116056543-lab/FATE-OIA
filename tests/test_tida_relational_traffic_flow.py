import torch

from fate_oia.models.tida_relational_traffic_flow import (
    TIDARelationalTrafficFlow,
    select_semantic_traffic_seeds,
)


def test_semantic_seed_selection_is_sparse_unique_and_predicate_anchored():
    torch.manual_seed(3)
    patch = torch.randn(2, 20, 8)
    attention = torch.rand(2, 5, 20)
    attention[:, 2, 7] = 10.0
    predicate_probability = torch.tensor([[0.1, 0.2, 0.9, 0.1, 0.1]]).expand(2, -1)
    selected = select_semantic_traffic_seeds(
        patch, attention, predicate_probability, grid_hw=(4, 5), topk=6
    )
    assert selected["tokens"].shape == (2, 1, 6, 8)
    assert selected["xy"].shape == (2, 1, 6, 2)
    assert torch.allclose(selected["weights"].sum(-1), torch.ones(2, 1))
    assert (selected["indices"].sort(-1).values.diff(dim=-1) > 0).all()
    assert (selected["indices"] == 7).any(-1).all()
    assert (selected["predicate_ids"][selected["indices"] == 7] == 2).all()


def _trajectory_inputs(batch=3, tracks=6, frames=5, dim=16):
    torch.manual_seed(9)
    appearance = torch.randn(batch, 1, tracks, frames, dim)
    xy = torch.randn(batch, 1, tracks, frames, 2).tanh()
    visibility = torch.ones(batch, 1, tracks, frames)
    pair_valid = torch.ones(batch, 1, tracks, frames - 1, dtype=torch.bool)
    exclusive = xy[..., 1:, :] - xy[..., :-1, :]
    common = torch.zeros(batch, frames - 1, 2)
    anchor = torch.full((batch, 1, tracks), 1.0 / tracks)
    return appearance, xy, visibility, pair_valid, common, exclusive, anchor


def test_relational_flow_is_zero_effect_at_init_but_trainable():
    model = TIDARelationalTrafficFlow(dim=16, num_actions=4, num_reasons=7, heads=4)
    action_nodes = torch.randn(3, 4, 16)
    reason_nodes = torch.randn(3, 7, 16)
    output = model(action_nodes, reason_nodes, *_trajectory_inputs())
    assert torch.equal(output["relational_action_delta"], torch.zeros(3, 4))
    assert torch.equal(output["relational_reason_delta"], torch.zeros(3, 7))
    assert output["relational_action_attention"].shape == (3, 4, 6)
    assert output["relational_reason_attention"].shape == (3, 7, 6)
    assert output["relational_action_pair_attention"].shape == (3, 4, 6, 6)
    assert output["relational_reason_pair_attention"].shape == (3, 7, 6, 6)
    assert output["relational_action_selected_deleted_delta"].shape == (3, 4)
    assert output["relational_action_random_deleted_delta"].shape == (3, 4)
    assert output["relational_reason_selected_deleted_delta"].shape == (3, 7)
    assert output["relational_reason_random_deleted_delta"].shape == (3, 7)
    assert output["relational_selected_track"].shape == (3,)
    assert output["relational_random_track"].shape == (3,)
    assert torch.all(output["relational_selected_track"] != output["relational_random_track"])
    assert output["relational_action_selected_track"].shape == (3, 4)
    assert output["relational_reason_selected_track"].shape == (3, 7)
    assert torch.all(
        output["relational_action_selected_track"]
        != output["relational_action_random_track"]
    )
    assert torch.all(
        output["relational_reason_selected_track"]
        != output["relational_reason_random_track"]
    )
    loss = (
        output["relational_action_candidate"][:, 0].sum()
        + output["relational_reason_candidate"][:, 0].sum()
    )
    loss.backward()
    assert model.action_output.weight.grad.abs().sum() > 0
    assert model.reason_output.weight.grad.abs().sum() > 0


def test_matched_control_prefers_similar_support_with_lower_target_credit():
    attention = torch.tensor([[[0.70, 0.20, 0.08, 0.02]]])
    support = torch.tensor([[0.80, 0.10, 0.78, 0.79]])

    control = TIDARelationalTrafficFlow._matched_control_track(attention, support)

    assert control.item() == 3


def test_action_and_reason_parameters_are_gradient_firewalled():
    model = TIDARelationalTrafficFlow(dim=16, num_actions=4, num_reasons=7, heads=4)
    output = model(
        torch.randn(2, 4, 16), torch.randn(2, 7, 16), *_trajectory_inputs(batch=2)
    )
    output["relational_action_candidate"][:, 0].sum().backward()
    reason_parameters = list(model.reason_encoder.parameters()) + list(model.reason_output.parameters())
    assert all(parameter.grad is None or parameter.grad.abs().sum() == 0 for parameter in reason_parameters)


def test_track_support_is_relative_to_best_semantic_anchor_not_topk_count():
    batch, tracks, frames = 1, 12, 4
    xy = torch.zeros(batch, 1, tracks, frames, 2)
    visibility = torch.ones(batch, 1, tracks, frames)
    pair_valid = torch.ones(batch, 1, tracks, frames - 1, dtype=torch.bool)
    displacement = torch.zeros(batch, 1, tracks, frames - 1, 2)
    equal_anchor = torch.full((batch, 1, tracks), 1.0 / tracks)

    _, _, _, support = TIDARelationalTrafficFlow._geometry(
        xy, visibility, pair_valid, displacement, equal_anchor
    )

    assert torch.allclose(support, torch.ones_like(support), atol=1e-6)


def test_closing_pair_gets_more_relation_weight_than_static_pair_at_same_distance():
    batch, tracks, frames = 1, 3, 3
    xy = torch.zeros(batch, 1, tracks, frames, 2)
    xy[:, :, 0, :, 0] = -0.5
    xy[:, :, 1, :, 0] = 0.5
    xy[:, :, 2, :, 0] = -0.5
    xy[:, :, 2, :, 1] = 1.0
    visibility = torch.ones(batch, 1, tracks, frames)
    pair_valid = torch.ones(batch, 1, tracks, frames - 1, dtype=torch.bool)
    displacement = torch.zeros(batch, 1, tracks, frames - 1, 2)
    displacement[:, :, 0, :, 0] = 0.2
    anchor = torch.full((batch, 1, tracks), 1.0 / tracks)

    _, _, relation_weight, _ = TIDARelationalTrafficFlow._geometry(
        xy, visibility, pair_valid, displacement, anchor
    )

    assert relation_weight[0, 0, 1] > relation_weight[0, 0, 2]


def test_common_mode_is_removed_and_temporal_order_changes_relations():
    model = TIDARelationalTrafficFlow(dim=16, num_actions=4, num_reasons=7, heads=4)
    with torch.no_grad():
        model.action_output.weight.fill_(0.05)
        model.reason_output.weight.fill_(0.05)
    inputs = _trajectory_inputs()
    action_nodes = torch.randn(3, 4, 16)
    reason_nodes = torch.randn(3, 7, 16)
    forward = model(action_nodes, reason_nodes, *inputs)
    reversed_inputs = list(inputs)
    reversed_inputs[0] = reversed_inputs[0].flip(3)
    reversed_inputs[1] = reversed_inputs[1].flip(3)
    reversed_inputs[2] = reversed_inputs[2].flip(3)
    reversed_inputs[3] = reversed_inputs[3].flip(3)
    reversed_inputs[4] = -reversed_inputs[4].flip(1)
    reversed_inputs[5] = -reversed_inputs[5].flip(3)
    reverse = model(action_nodes, reason_nodes, *reversed_inputs)
    assert torch.allclose(forward["relational_action_candidate"].mean(-1), torch.zeros(3), atol=1e-6)
    assert torch.allclose(forward["relational_reason_candidate"].mean(-1), torch.zeros(3), atol=1e-6)
    assert not torch.allclose(
        forward["relational_action_candidate"], reverse["relational_action_candidate"]
    )


def test_target_conditioned_pair_context_is_zero_init_and_gets_first_step_gradient():
    model = TIDARelationalTrafficFlow(dim=16, num_actions=4, num_reasons=7, heads=4)
    with torch.no_grad():
        model.action_output.weight.fill_(0.05)
    output = model(
        torch.randn(2, 4, 16), torch.randn(2, 7, 16), *_trajectory_inputs(batch=2)
    )
    assert torch.equal(
        model.action_encoder.target_relation_output.weight,
        torch.zeros_like(model.action_encoder.target_relation_output.weight),
    )
    output["relational_action_candidate"][:, 0].sum().backward()
    assert model.action_encoder.target_relation_output.weight.grad.abs().sum() > 0


def test_reason_traffic_relevance_mask_blocks_static_reason_residuals():
    model = TIDARelationalTrafficFlow(
        dim=16,
        num_actions=4,
        num_reasons=7,
        heads=4,
        reason_traffic_indices=(1, 5),
    )
    with torch.no_grad():
        model.reason_output.weight.fill_(0.05)
    output = model(
        torch.randn(2, 4, 16), torch.randn(2, 7, 16), *_trajectory_inputs(batch=2)
    )
    blocked = [0, 2, 3, 4, 6]
    assert torch.equal(
        output["relational_reason_candidate"][:, blocked],
        torch.zeros(2, len(blocked)),
    )
    assert output["relational_reason_candidate"][:, [1, 5]].abs().sum() > 0
