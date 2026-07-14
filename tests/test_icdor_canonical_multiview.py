from __future__ import annotations

import torch

from fate_oia.datasets.mosaic_multiview import MOSAICWeakMultiView


def test_first_view_is_canonical_and_second_view_restores_all_label_spaces() -> None:
    transform = MOSAICWeakMultiView(
        ("left_lane", "right_lane", "front"),
        flip_probability=1.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        seed=7,
    )
    output = transform(torch.arange(24, dtype=torch.float32).reshape(3, 2, 4))
    first, second = output["metadata"]
    assert first["horizontal_flip"] is False
    assert second["horizontal_flip"] is True

    action = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.equal(transform.invert_action_values(action, second), torch.tensor([1.0, 2.0, 4.0, 3.0]))

    reason = torch.arange(21, dtype=torch.float32)
    restored_reason = transform.invert_reason_values(reason, second)
    for left, right in ((9, 15), (10, 16), (11, 17), (12, 18), (13, 19), (14, 20)):
        assert restored_reason[left] == reason[right]
        assert restored_reason[right] == reason[left]

    geometry = torch.tensor([[0.25, 0.5], [-0.75, -0.5]])
    restored_geometry = transform.invert_geometry_coordinates(geometry, second)
    assert torch.equal(restored_geometry[:, 0], -geometry[:, 0])

