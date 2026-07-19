from __future__ import annotations

import torch

from fate_oia.models.mosaic_factor_seeded_rereader import MOSAICFactorSeededRereader
from _mosaic_v5_helpers import typed_inputs


def test_typed_reason_cannot_bypass_credibility() -> None:
    feature_map, _, coordinates, features, attention = typed_inputs()
    queries = torch.randn(2, 21, 8)
    rereader = MOSAICFactorSeededRereader(dim=8, target_count=21)
    output = rereader(feature_map, queries, coordinates, features, attention, torch.zeros(2, 3, 21))
    assert torch.equal(output["target_nodes"], torch.zeros_like(output["target_nodes"]))
