import torch

from fate_oia.models.acpr_ego_regions import ACPREgoRegionEncoder


def test_acpr_ego_region_shapes_and_mass():
    enc = ACPREgoRegionEncoder()
    tokens = torch.zeros(2, 3600, 384)
    out, feats, masks, stats = enc(tokens)
    assert out.shape == tokens.shape
    assert feats.shape == (3600, 8)
    assert {"front_center", "left_corridor", "right_corridor", "upper_traffic_region", "bottom_drivable_region"} <= set(masks)
    assert stats["right_corridor_mass"] > 0
