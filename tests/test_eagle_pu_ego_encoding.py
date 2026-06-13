from __future__ import annotations

import torch
from fate_oia.models.eagle_pu_ego_encoding import EaglePUEgoEncoding

def test_ego_encoding_uses_fixed_corridor_formulas():
    enc = EaglePUEgoEncoding(grid_hw=(45, 80), dim=16)
    features, stats = enc.features(device=torch.device("cpu"), dtype=torch.float32)
    assert features.shape == (3600, 8)
    assert enc.feature_names == ["x_norm", "y_norm", "center_abs_x", "bottomness", "front_center", "left_corridor", "right_corridor", "upper_control_region"]
    left = features[:, 5].view(45, 80)
    right = features[:, 6].view(45, 80)
    upper = features[:, 7].view(45, 80)
    assert right[:, -10:].mean() > right[:, :10].mean()
    assert left[:, :10].mean() > left[:, -10:].mean()
    assert upper[:10].mean() > upper[-10:].mean()
    for key in ["left_corridor_mass", "right_corridor_mass", "front_center_mass", "upper_region_mass"]:
        assert key in stats and stats[key] > 0
