import torch

from fate_oia.models.tida_interaction_event_features import (
    EVENT_NAMES,
    build_action_interaction_events,
)


def _tracks() -> tuple[torch.Tensor, torch.Tensor]:
    tracks = torch.zeros(2, 7, 5, 2)
    tracks[..., 1] = -0.2
    tracks[:, :, 0, 0] = torch.linspace(-0.1, 0.0, 7)
    tracks[:, :, 0, 1] = torch.linspace(-0.4, 0.65, 7)
    tracks[:, :, 1, 0] = torch.linspace(-0.8, -0.2, 7)
    tracks[:, :, 1, 1] = torch.linspace(0.0, 0.45, 7)
    tracks[:, :, 2, 0] = torch.linspace(0.8, 0.2, 7)
    tracks[:, :, 2, 1] = torch.linspace(0.0, 0.45, 7)
    tracks[:, :, 3, 0] = -0.7
    tracks[:, :, 4, 0] = 0.7
    visibility = torch.ones(2, 7, 5, dtype=torch.bool)
    return tracks, visibility


def test_events_are_finite_target_conditioned_and_named() -> None:
    tracks, visibility = _tracks()
    events = build_action_interaction_events(tracks, visibility)
    assert events.shape == (2, 4, len(EVENT_NAMES))
    assert torch.isfinite(events).all()
    assert (events >= 0).all() and (events <= 1).all()
    assert len(EVENT_NAMES) == len(set(EVENT_NAMES))


def test_events_remove_shared_camera_translation() -> None:
    tracks, visibility = _tracks()
    drift = torch.linspace(0, 0.3, tracks.shape[1])[None, :, None, None]
    shifted = tracks + torch.cat((0.6 * drift, drift), dim=-1)
    torch.testing.assert_close(
        build_action_interaction_events(tracks, visibility),
        build_action_interaction_events(shifted, visibility),
        atol=2e-5,
        rtol=2e-5,
    )


def test_left_right_events_are_mirror_equivariant() -> None:
    tracks, visibility = _tracks()
    mirrored = tracks.clone()
    mirrored[..., 0] = -mirrored[..., 0]
    original = build_action_interaction_events(tracks, visibility)
    reflected = build_action_interaction_events(mirrored, visibility)
    torch.testing.assert_close(original[:, 0], reflected[:, 0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(original[:, 1], reflected[:, 1], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(original[:, 2], reflected[:, 3], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(original[:, 3], reflected[:, 2], atol=2e-5, rtol=2e-5)


def test_time_reversal_changes_directional_event_credit() -> None:
    tracks, visibility = _tracks()
    forward = build_action_interaction_events(tracks, visibility)
    reversed_events = build_action_interaction_events(
        tracks.flip(1), visibility.flip(1)
    )
    assert (forward - reversed_events).abs().amax().item() > 0.05

