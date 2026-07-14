from __future__ import annotations

import torch


def test_factor_audit_missingness_modes_are_honest_and_non_crashing() -> None:
    from fate_oia.engine import mosaic_icdor_audit_collectors as collectors

    assert hasattr(collectors, "summarize_factor_supervision"), "missing missingness-safe factor summary"
    cases = (
        (torch.tensor([1, 1, 0, 0], dtype=torch.bool), torch.tensor([0, 0, 1, 1], dtype=torch.bool), torch.zeros(4, dtype=torch.bool), "binary_confirmed", True),
        (torch.tensor([1, 1, 0, 0], dtype=torch.bool), torch.zeros(4, dtype=torch.bool), torch.tensor([0, 0, 1, 1], dtype=torch.bool), "positive_vs_weak_negative", True),
        (torch.tensor([1, 1], dtype=torch.bool), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), "positive_only", False),
        (torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), "unavailable", False),
    )
    for positive, reliable_negative, weak_negative, mode, available in cases:
        row = collectors.summarize_factor_supervision(
            torch.linspace(0.1, 0.9, positive.numel()), positive, reliable_negative, weak_negative,
            geometry_valid_mask=positive.clone(),
        )
        assert row["evaluation_mode"] == mode
        assert row["metric_available"] is available
        if available:
            assert row["presence_auprc"] is not None
        else:
            assert row["presence_auprc"] is None
            assert row["unavailable_reason"]
            assert row["certificate_ceiling"] == "Abstained"
