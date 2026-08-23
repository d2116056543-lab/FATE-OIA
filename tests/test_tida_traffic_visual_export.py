import torch

from fate_oia.engine.export_tida_traffic_action_visuals import (
    attention_to_time_action_grid,
    render_patch_motion_vectors,
)


def test_attention_export_preserves_time_and_source_action_axes():
    attention = torch.arange(40.0).reshape(4, 10)
    grid = attention_to_time_action_grid(attention, num_actions=2)
    assert grid.shape == (4, 5, 2)
    torch.testing.assert_close(grid[0, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(grid[3, -1], torch.tensor([38.0, 39.0]))


def test_patch_motion_vector_export_is_created(tmp_path):
    raw = torch.tensor([[[0.1, 0.0], [0.0, 0.1]], [[0.2, 0.0], [0.0, 0.2]]])
    common = torch.tensor([[0.05, 0.0], [0.05, 0.0]])
    exclusive = raw - common[:, None]
    output = tmp_path / "vectors.png"
    render_patch_motion_vectors(raw, common, exclusive, output)
    assert output.exists() and output.stat().st_size > 0
