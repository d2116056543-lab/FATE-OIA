import torch

from fate_oia.engine.export_tida_traffic_flow_visuals import _ego_compensated_paths


def test_ego_compensated_paths_remove_uniform_camera_translation():
    static = torch.zeros(5, 3, 2)
    translated = static.clone()
    translated[:, :, 0] = torch.arange(5)[:, None]
    visible = torch.ones(5, 3, dtype=torch.bool)

    base = _ego_compensated_paths(static, visible)
    shifted = _ego_compensated_paths(translated, visible)

    torch.testing.assert_close(base, shifted)
