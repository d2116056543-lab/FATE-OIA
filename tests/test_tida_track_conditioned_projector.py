import torch


def _moving_field(batch=2, labels=3, tracks=4, frames=5, dim=16):
    appearance = torch.randn(batch, labels, tracks, frames, dim)
    xy = torch.zeros(batch, labels, tracks, frames, 2)
    xy[..., 0] = torch.linspace(-0.4, 0.4, frames)
    xy[..., 1] = torch.linspace(0.2, -0.2, frames)
    visibility = torch.ones(batch, labels, tracks, frames)
    exclusive = xy[..., 1:, :] - xy[..., :-1, :]
    return appearance, xy, visibility, exclusive


def test_action_specific_tracks_become_target_private_ordered_tokens():
    from fate_oia.models.tida_track_conditioned_projector import (
        TIDATrackConditionedProjector,
    )

    torch.manual_seed(7)
    batch, labels, tracks, frames, dim = 2, 3, 4, 5, 16
    target = torch.randn(batch, labels, dim)
    appearance, xy, visibility, exclusive = _moving_field(
        batch, labels, tracks, frames, dim
    )
    projector = TIDATrackConditionedProjector(dim=dim, num_targets=labels)
    output = projector(
        target,
        appearance,
        xy,
        visibility,
        exclusive,
        action_specific=True,
    )

    assert output["track_conditioned_tokens"].shape == (batch, frames, labels, dim)
    assert output["track_attention"].shape == (batch, labels, frames, tracks)
    assert torch.allclose(output["track_attention"].sum(-1), torch.ones(batch, labels, frames))
    assert output["track_motion_rms"].gt(0).all()
    assert torch.isfinite(output["track_conditioned_tokens"]).all()


def test_shared_semantic_tracks_are_read_differently_by_each_reason():
    from fate_oia.models.tida_track_conditioned_projector import (
        TIDATrackConditionedProjector,
    )

    torch.manual_seed(11)
    batch, reasons, tracks, frames, dim = 2, 5, 6, 4, 16
    target = torch.randn(batch, reasons, dim)
    appearance, xy, visibility, exclusive = _moving_field(
        batch, 1, tracks, frames, dim
    )
    projector = TIDATrackConditionedProjector(dim=dim, num_targets=reasons)
    output = projector(
        target,
        appearance,
        xy,
        visibility,
        exclusive,
        action_specific=False,
    )

    tokens = output["track_conditioned_tokens"]
    assert tokens.shape == (batch, frames, reasons, dim)
    assert not torch.allclose(tokens[:, :, 0], tokens[:, :, 1])
    assert output["track_effective_count"].ge(1).all()


def test_geometry_makes_temporal_order_observable_without_changing_static_case():
    from fate_oia.models.tida_track_conditioned_projector import (
        TIDATrackConditionedProjector,
    )

    torch.manual_seed(19)
    target = torch.randn(1, 2, 16)
    appearance, xy, visibility, exclusive = _moving_field(1, 2, 3, 5, 16)
    projector = TIDATrackConditionedProjector(dim=16, num_targets=2)
    ordered = projector(target, appearance, xy, visibility, exclusive, action_specific=True)
    assert not torch.allclose(
        ordered["track_conditioned_tokens"],
        ordered["track_shuffled_conditioned_tokens"],
    )
    reversed_output = projector(
        target,
        appearance.flip(3),
        xy.flip(3),
        visibility.flip(3),
        -exclusive.flip(3),
        action_specific=True,
    )
    assert not torch.allclose(
        ordered["track_conditioned_tokens"],
        reversed_output["track_conditioned_tokens"],
    )

    static_xy = xy[..., -1:, :].expand_as(xy)
    static_exclusive = torch.zeros_like(exclusive)
    static = projector(
        target,
        appearance[..., -1:, :].expand_as(appearance),
        static_xy,
        visibility,
        static_exclusive,
        action_specific=True,
    )
    assert static["track_motion_rms"].eq(0).all()


def test_cycle_confidence_biases_attention_toward_reliable_tracks():
    from fate_oia.models.tida_track_conditioned_projector import (
        TIDATrackConditionedProjector,
    )

    torch.manual_seed(23)
    target = torch.randn(1, 1, 16)
    appearance = torch.randn(1, 1, 4, 3, 16)
    # Equal appearances isolate the confidence prior from content similarity.
    appearance[:, :, 1:] = appearance[:, :, :1]
    xy = torch.zeros(1, 1, 4, 3, 2)
    visibility = torch.tensor([[[[0.95, 0.95, 0.95], [0.10, 0.10, 0.10],
                                  [0.05, 0.05, 0.05], [0.01, 0.01, 0.01]]]])
    exclusive = torch.zeros(1, 1, 4, 2, 2)
    projector = TIDATrackConditionedProjector(
        dim=16,
        num_targets=1,
        attention_temperature=1.0,
        attention_topk=4,
        confidence_power=1.0,
    )
    output = projector(
        target, appearance, xy, visibility, exclusive, action_specific=True
    )

    attention = output["track_attention"][0, 0]
    assert torch.all(attention[:, 0] > attention[:, 1])
    assert torch.all(attention[:, 1] > attention[:, 2])
    assert torch.all(attention[:, 2] > attention[:, 3])


def test_topk_attention_has_exact_sparse_support_and_finite_gradients():
    from fate_oia.models.tida_track_conditioned_projector import (
        TIDATrackConditionedProjector,
    )

    torch.manual_seed(29)
    target = torch.randn(2, 3, 16, requires_grad=True)
    appearance, xy, visibility, exclusive = _moving_field(2, 3, 8, 5, 16)
    projector = TIDATrackConditionedProjector(
        dim=16,
        num_targets=3,
        attention_temperature=0.25,
        attention_topk=3,
        confidence_power=1.0,
    )
    output = projector(
        target, appearance, xy, visibility, exclusive, action_specific=True
    )

    attention = output["track_attention"]
    assert torch.all((attention > 0).sum(-1) <= 3)
    assert torch.allclose(attention.sum(-1), torch.ones_like(attention.sum(-1)))
    output["track_conditioned_tokens"].square().mean().backward()
    assert target.grad is not None and torch.isfinite(target.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in projector.parameters()
    )
