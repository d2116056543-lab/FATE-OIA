import torch

from fate_oia.datasets.bdd_oia_video import quadratic_multirate_timestamps, timestamps_to_indices


def test_quadratic_multirate_exact_and_deterministic():
    t = quadratic_multirate_timestamps(15, 5.0)
    expected = torch.tensor([-5.0 * (1.0 - i / 14.0) ** 2 for i in range(15)])
    assert torch.allclose(t, expected)
    assert t[-1].item() == 0.0
    assert torch.all(t[1:] > t[:-1])
    idx = timestamps_to_indices(t, fps=30.0, target_frame_index=150)
    assert idx[-1].item() == 150
    assert torch.all(idx[1:] > idx[:-1])
