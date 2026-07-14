from __future__ import annotations

import importlib

import torch


def test_matched_controls_have_four_real_nonoverlap_arms_with_mass_tolerance() -> None:
    module = importlib.import_module("fate_oia.engine.mosaic_icdor_audit_collectors")
    selected = torch.zeros(1, 8, 8)
    selected[:, 2:4, 2:4] = 1.0
    controls = module.build_matched_factor_controls(
        selected,
        selected_factor_type="object",
        selected_region="front",
        identity_masks=torch.stack([selected.roll(3, -1), selected.roll(-3, -1)]),
        identity_types=("object", "object"),
        identity_regions=("front", "front"),
        min_controls=4,
    )
    assert len(controls) >= 4
    kinds = {row["control_type"] for row in controls}
    assert {"spatial_roll", "same_type_identity"} <= kinds
    selected_mass = selected.sum().item()
    for row in controls:
        assert row["overlap"] == 0.0
        assert abs(row["mask_sum"] - selected_mass) / selected_mass <= 0.05
        assert row["factor_type"] == "object"
        assert row["region"] == "front"

