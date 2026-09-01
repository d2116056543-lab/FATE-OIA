import torch

from fate_oia.models.tida_geometric_flow import (
    TIDAGeometricFlowDecisionHeads,
    TIDAGeometricFlowEncoder,
)


def _clip(frame: torch.Tensor, frames: int = 5) -> torch.Tensor:
    return torch.stack([frame.clone() for _ in range(frames)], dim=1)


def test_static_clip_has_negligible_geometric_motion():
    model = TIDAGeometricFlowEncoder(hidden_dim=64, flow_hw=(16, 24))
    frame = torch.rand(2, 3, 32, 48)
    output = model(_clip(frame), torch.ones(2, 5, dtype=torch.bool))
    assert output["motion_energy"].abs().max() < 1e-5
    assert output["flow_field"].abs().max() < 1e-5


def test_horizontal_translation_has_direction_and_reversal_changes_sign():
    model = TIDAGeometricFlowEncoder(hidden_dim=64, flow_hw=(16, 24))
    base = torch.zeros(1, 3, 32, 48)
    base[:, :, 8:24, 8:20] = 1.0
    frames = torch.stack([torch.roll(base, shifts=2 * step, dims=-1) for step in range(5)], dim=1)
    ordered = model(frames, torch.ones(1, 5, dtype=torch.bool))
    reversed_output = model(frames.flip(1), torch.ones(1, 5, dtype=torch.bool))
    assert ordered["global_horizontal"].mean() > 0
    assert reversed_output["global_horizontal"].mean() < 0


def test_sparse_large_translation_is_recovered_by_multiscale_correlation():
    model = TIDAGeometricFlowEncoder(hidden_dim=64, flow_hw=(24, 32))
    base = torch.zeros(1, 3, 48, 64)
    base[:, :, 10:30, 6:18] = 1.0
    frames = torch.stack([torch.roll(base, shifts=8 * step, dims=-1) for step in range(3)], dim=1)
    output = model(frames, torch.ones(1, 3, dtype=torch.bool))
    assert output["flow_match_confidence"].mean() > 0
    assert output["global_horizontal"].mean() > 0.01


def test_outward_motion_has_positive_expansion():
    model = TIDAGeometricFlowEncoder(hidden_dim=64, flow_hw=(20, 28))
    frames = []
    for size in (4, 6, 8, 10, 12):
        frame = torch.zeros(1, 3, 40, 56)
        cy, cx = 20, 28
        frame[:, :, cy - size : cy + size, cx - size : cx + size] = 1.0
        frames.append(frame)
    output = model(torch.stack(frames, dim=1), torch.ones(1, 5, dtype=torch.bool))
    assert output["global_expansion"].mean() > 0


def test_history_off_is_exact_zero_and_gradients_are_finite():
    model = TIDAGeometricFlowEncoder(hidden_dim=64, flow_hw=(16, 24))
    frames = torch.rand(2, 5, 3, 32, 48, requires_grad=True)
    off = model(frames, torch.zeros(2, 5, dtype=torch.bool))
    assert torch.equal(off["flow_state"], torch.zeros_like(off["flow_state"]))
    on = model(frames, torch.ones(2, 5, dtype=torch.bool))
    on["flow_state"].square().mean().backward()
    assert frames.grad is not None and torch.isfinite(frames.grad).all()


def test_region_outputs_are_real_and_not_broadcast_copies():
    model = TIDAGeometricFlowEncoder(hidden_dim=64, flow_hw=(16, 24))
    base = torch.zeros(1, 3, 32, 48)
    base[:, :, 10:24, :16] = 1.0
    frames = torch.stack([torch.roll(base, shifts=step, dims=-1) for step in range(5)], dim=1)
    output = model(frames, torch.ones(1, 5, dtype=torch.bool))
    assert output["region_motion"].shape == (1, 4, 5, 3)
    assert output["region_motion"].std(dim=2).mean() > 0


def test_flow_grid_tracks_follow_motion_without_an_external_tracker():
    model = TIDAGeometricFlowEncoder(hidden_dim=64, flow_hw=(16, 24))
    base = torch.zeros(1, 3, 32, 48)
    base[:, :, 8:24, 8:24] = 1.0
    frames = torch.stack(
        [torch.roll(base, shifts=2 * step, dims=-1) for step in range(6)], dim=1
    )
    output = model(frames, torch.ones(1, 6, dtype=torch.bool))
    tracks = output["flow_grid_tracks_xy"]
    visibility = output["flow_grid_tracks_visibility"]
    assert tracks.shape == (1, 6, 16, 2)
    assert visibility.shape == (1, 6, 16)
    assert tracks[..., 0].abs().max() <= 1.0
    assert tracks[..., 1].abs().max() <= 1.0
    assert visibility[:, 0].all()
    assert (tracks[:, -1, :, 0] - tracks[:, 0, :, 0]).mean() > 0


def test_encoder_preserves_local_motion_tokens_before_target_pooling():
    model = TIDAGeometricFlowEncoder(
        hidden_dim=64,
        flow_hw=(16, 24),
        motion_token_hw=(4, 6),
    )
    base = torch.zeros(2, 3, 32, 48)
    base[0, :, 8:24, :16] = 1.0
    base[1, :, 8:24, 32:] = 1.0
    frames = torch.stack(
        [torch.roll(base, shifts=step, dims=-1) for step in range(5)], dim=1
    )

    output = model(frames, torch.ones(2, 5, dtype=torch.bool))

    assert output["flow_motion_tokens"].shape == (2, 24, 26)
    assert output["flow_motion_token_xy"].shape == (2, 24, 2)
    assert output["flow_motion_token_mask"].shape == (2, 24)
    assert output["flow_motion_token_mask"].all()
    assert not torch.allclose(
        output["flow_motion_tokens"][0], output["flow_motion_tokens"][1]
    )


def test_target_conditioned_flow_readers_are_zero_init_and_owner_isolated():
    encoder = TIDAGeometricFlowEncoder(
        hidden_dim=64,
        flow_hw=(16, 24),
        motion_token_hw=(4, 6),
    )
    heads = TIDAGeometricFlowDecisionHeads(
        hidden_dim=64,
        motion_feature_dim=encoder.motion_token_dim,
        target_context_dim=32,
        num_actions=4,
        num_reasons=21,
    )
    frames = torch.rand(2, 5, 3, 32, 48)
    measured = encoder(frames, torch.ones(2, 5, dtype=torch.bool))
    action_context = torch.randn(2, 4, 32)
    reason_context = torch.randn(2, 21, 32)
    action_prior = torch.nn.functional.one_hot(
        torch.arange(4), num_classes=24
    ).float()[None].expand(2, -1, -1)
    reason_prior = torch.nn.functional.one_hot(
        torch.arange(21) % 24, num_classes=24
    ).float()[None].expand(2, -1, -1)

    output = heads(
        measured["flow_state"],
        measured["history_available"],
        motion_tokens=measured["flow_motion_tokens"],
        motion_token_mask=measured["flow_motion_token_mask"],
        action_target_context=action_context,
        reason_target_context=reason_context,
        action_target_attention_prior=action_prior,
        reason_target_attention_prior=reason_prior,
    )

    assert torch.equal(
        output["geometric_action_delta"],
        torch.zeros_like(output["geometric_action_delta"]),
    )
    assert torch.equal(
        output["geometric_reason_delta"],
        torch.zeros_like(output["geometric_reason_delta"]),
    )
    assert output["geometric_action_motion_attention"].shape == (2, 4, 24)
    assert output["geometric_reason_motion_attention"].shape == (2, 21, 24)
    assert torch.allclose(
        output["geometric_action_motion_attention"].sum(-1),
        torch.ones(2, 4),
        atol=1e-5,
    )
    assert output["geometric_action_motion_attention"].max(-1).values.min() > 0.45

    with torch.no_grad():
        heads.action_head.output.weight.fill_(0.1)
    action_loss = heads(
        measured["flow_state"].detach(),
        measured["history_available"],
        motion_tokens=measured["flow_motion_tokens"].detach(),
        motion_token_mask=measured["flow_motion_token_mask"],
        action_target_context=action_context,
        reason_target_context=reason_context,
        action_target_attention_prior=action_prior,
        reason_target_attention_prior=reason_prior,
    )["geometric_action_delta"].sum()
    action_loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in heads.action_parameters()
    )
    assert all(parameter.grad is None for parameter in heads.reason_parameters())


def test_affine_ego_motion_is_removed_but_local_object_motion_remains():
    height, width = 12, 20
    y = torch.linspace(-1.0, 1.0, height).view(1, 1, 1, height, 1)
    x = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, 1, width)
    horizontal = 0.20 + 0.08 * x - 0.03 * y
    vertical = -0.10 + 0.02 * x + 0.05 * y
    confidence = torch.ones_like(horizontal)

    residual_x, residual_y = TIDAGeometricFlowEncoder._affine_residual_flow(
        horizontal, vertical, confidence
    )
    assert residual_x.abs().max() < 1e-4
    assert residual_y.abs().max() < 1e-4

    moving_x = horizontal.clone()
    moving_x[..., 4:8, 7:13] += 0.5
    residual_x, _ = TIDAGeometricFlowEncoder._affine_residual_flow(
        moving_x, vertical, confidence
    )
    moving_mass = residual_x[..., 4:8, 7:13].abs().mean()
    background = residual_x.clone()
    background[..., 4:8, 7:13] = 0
    assert moving_mass > 0.25
    assert background.abs().mean() < 0.08


def test_affine_ego_motion_compensation_supports_bfloat16_inputs():
    horizontal = torch.randn(2, 3, 1, 8, 12, dtype=torch.bfloat16)
    vertical = torch.randn_like(horizontal)
    confidence = torch.ones_like(horizontal)

    residual_x, residual_y = TIDAGeometricFlowEncoder._affine_residual_flow(
        horizontal, vertical, confidence
    )

    assert residual_x.dtype == torch.bfloat16
    assert residual_y.dtype == torch.bfloat16
    assert torch.isfinite(residual_x.float()).all()
    assert torch.isfinite(residual_y.float()).all()
