import random

import torch

from fate_oia.datasets.bdd_oia_video import (
    jitter_timestamps,
    quadratic_multirate_timestamps,
    timestamps_to_indices,
)


def test_quadratic_multirate_exact_and_deterministic():
    t = quadratic_multirate_timestamps(15, 5.0)
    expected = torch.tensor([-5.0 * (1.0 - i / 14.0) ** 2 for i in range(15)])
    assert torch.allclose(t, expected)
    assert t[-1].item() == 0.0
    assert torch.all(t[1:] > t[:-1])
    idx = timestamps_to_indices(t, fps=30.0, target_frame_index=150)
    assert idx[-1].item() == 150
    assert torch.all(idx[1:] > idx[:-1])


def test_low_fps_sampling_projects_to_unique_feasible_indices():
    timestamps = quadratic_multirate_timestamps(15, 5.0)

    indices = timestamps_to_indices(timestamps, fps=2.0, target_frame_index=30)

    assert indices[0] >= 0
    assert indices[-1].item() == 30
    assert torch.all(indices[1:] > indices[:-1])


def test_short_high_fps_clip_projects_clamped_history_without_negative_indices():
    timestamps = quadratic_multirate_timestamps(15, 5.0)

    indices = timestamps_to_indices(
        timestamps, fps=30.23057216054654, target_frame_index=101
    )

    assert indices[0] >= 0
    assert indices[-1].item() == 101
    assert torch.all(indices[1:] > indices[:-1])


def test_jittered_sampling_remains_feasible_across_clip_rates_and_lengths():
    base = quadratic_multirate_timestamps(15, 5.0)
    for fps, target in ((1.0, 14), (2.0, 20), (30.23, 101), (60.0, 150)):
        for seed in range(20):
            timestamps = jitter_timestamps(base, random.Random(seed))
            indices = timestamps_to_indices(timestamps, fps=fps, target_frame_index=target)
            assert indices[0] >= 0
            assert indices[-1].item() == target
            assert torch.all(indices[1:] > indices[:-1])
