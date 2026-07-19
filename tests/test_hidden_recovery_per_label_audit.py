from __future__ import annotations

import torch

from fate_oia.engine.mosaic_icdor_hidden_recovery_audit import audit_hidden_recovery_scores


def test_hidden_recovery_audit_exports_per_label_margin_for_pu_admission() -> None:
    posterior = torch.zeros(4, 21)
    baseline = torch.zeros_like(posterior)
    posterior[:, 0] = torch.tensor([0.9, 0.8, 0.2, 0.1])
    baseline[:, 0] = torch.tensor([0.1, 0.2, 0.8, 0.9])
    observed = torch.zeros_like(posterior)
    hidden = torch.zeros_like(posterior, dtype=torch.bool)
    hidden[:2, 0] = True
    audit = audit_hidden_recovery_scores(
        posterior, baseline, observed, hidden, mode="mcar", hide_fraction=0.10
    )
    assert len(audit["per_label"]) == 21
    assert audit["per_label"][0]["available"] is True
    assert audit["per_label"][0]["margin"] > 0.0
    assert audit["per_label"][1]["available"] is False
