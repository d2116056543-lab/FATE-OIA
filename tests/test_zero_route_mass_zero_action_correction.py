from __future__ import annotations

import torch

from fate_oia.models.mosaic_factor_seeded_rereader import MOSAICFactorSeededRereader
from _mosaic_v5_helpers import typed_inputs


def test_zero_route_mass_zero_action_correction() -> None:
    feature_map, queries, coordinates, features, attention = typed_inputs()
    rereader = MOSAICFactorSeededRereader(dim=8, target_count=4)
    output = rereader(feature_map, queries, coordinates, features, attention, torch.zeros(2, 3, 4))
    assert torch.equal(output["route_mass"], torch.zeros_like(output["route_mass"]))
    assert torch.equal(output["support_logits"], torch.zeros_like(output["support_logits"]))
    assert torch.equal(output["veto_logits"], torch.zeros_like(output["veto_logits"]))
