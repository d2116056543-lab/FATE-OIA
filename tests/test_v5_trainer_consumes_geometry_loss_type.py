from __future__ import annotations

import inspect

from fate_oia.engine import train_acpr_mosaic_trust_icdor as trainer


def test_factor_geometry_loss_uses_v5_ontology_contract_not_legacy_type() -> None:
    """The V5 YAML field must drive the real loss routing, not be decorative."""
    source = inspect.getsource(trainer.compute_icdor_training_losses)

    assert "geometry_loss_type" in source
    assert "curve_distance" in source
    assert "object_region_dice" in source
    assert 'item.get("type", "object")' not in source
