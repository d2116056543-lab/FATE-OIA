from __future__ import annotations

import torch

from fate_oia.engine.train_acpr_mosaic_trust_icdor import per_label_pu_margin_from_hidden_rows


def test_pu_aggregation_is_conservative_per_reason_not_global() -> None:
    rows = [
        {"per_label": [
            {"available": True, "margin": 0.04},
            {"available": True, "margin": -0.01},
        ] + [{"available": False, "margin": None} for _ in range(19)]},
        {"per_label": [
            {"available": True, "margin": 0.02},
            {"available": True, "margin": 0.03},
        ] + [{"available": False, "margin": None} for _ in range(19)]},
    ]
    margins = per_label_pu_margin_from_hidden_rows(rows)
    assert margins.shape == (21,)
    assert torch.allclose(margins[:2], torch.tensor([0.02, -0.01]))
    assert torch.isneginf(margins[2:]).all()
