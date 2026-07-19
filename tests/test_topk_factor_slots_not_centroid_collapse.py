from __future__ import annotations

import torch

from fate_oia.models.mosaic_factor_seeded_rereader import MOSAICFactorSeededRereader
from _mosaic_v5_helpers import typed_inputs


def test_topk_factor_slots_not_centroid_collapse() -> None:
    feature_map, queries, coordinates, features, attention = typed_inputs()
    coordinates[:, 0, ...] = -0.75
    coordinates[:, 1, ...] = 0.75
    weights = torch.zeros(2, 3, 4)
    weights[:, 0] = 0.6
    weights[:, 1] = 0.4
    output = MOSAICFactorSeededRereader(dim=8, target_count=4, slot_count=2)(
        feature_map, queries, coordinates, features, attention, weights
    )
    assert output["topk_factor_ids"].shape == (2, 4, 2)
    assert not torch.allclose(output["slot_coordinates"][:, :, 0], output["slot_coordinates"][:, :, 1])
