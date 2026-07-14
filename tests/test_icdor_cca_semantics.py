from __future__ import annotations

import torch


def test_cca_is_directional_correctness_rate_not_calibration_difference() -> None:
    from fate_oia.engine import mosaic_target_transfer_metrics as metrics

    assert hasattr(metrics, "counterfactual_credit_alignment"), "CCA still has calibration-difference semantics"
    on = torch.tensor([0.9, 0.8, 0.2, 0.1])
    off = torch.tensor([0.4, 0.3, 0.7, 0.6])
    support_rows = torch.tensor([True, True, False, False])
    veto_rows = ~support_rows
    cca = metrics.counterfactual_credit_alignment(on, off, support_rows=support_rows, veto_rows=veto_rows)
    assert cca.item() == 1.0
    reversed_cca = metrics.counterfactual_credit_alignment(off, on, support_rows=support_rows, veto_rows=veto_rows)
    assert reversed_cca.item() == 0.0
