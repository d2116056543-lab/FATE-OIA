from __future__ import annotations

import torch

from fate_oia.datasets.mosaic_icdor_factor_supervision import build_factor_supervision


def test_factor_supervision_keeps_reliable_and_weak_negative_provenance_distinct() -> None:
    observations = {
        "presence_target": torch.tensor([[0.0, 0.0]]),
        "presence_known_mask": torch.tensor([[1.0, 0.0]]),
        "geometry_known_mask": torch.zeros(1, 2),
        "weak_negative_mask": torch.tensor([[0.0, 1.0]]),
    }
    result = build_factor_supervision(
        observations,
        reason_targets=None,
        factors=(
            {"name": "observable", "positive_reason_anchors": []},
            {"name": "weak", "positive_reason_anchors": []},
        ),
        split="train",
        allow_reason_anchors=False,
    )

    assert result["reliable_negative_mask"].tolist() == [[True, False]]
    assert result["weak_negative_mask"].tolist() == [[False, True]]
    assert result["supervision_code"][0, 0].item() != result["supervision_code"][0, 1].item()
