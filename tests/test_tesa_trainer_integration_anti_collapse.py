from __future__ import annotations

import torch

from fate_oia.engine.audit_meter_oia_v3_heca import _dynamic_checks
from fate_oia.losses.meter_pu_losses import meter_hidden_positive_audit


def test_hidden_positive_audit_reports_the_same_product_score_used_by_training() -> None:
    private = torch.tensor(
        [[0.9], [0.8], [0.7], [0.6], [0.5], [0.4], [0.3], [0.2]]
    )
    state = torch.tensor(
        [[0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]]
    )
    targets = torch.tensor([[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
    report = meter_hidden_positive_audit(
        private,
        state,
        targets,
        hidden_fraction=0.5,
        min_positive_count=1,
        seed=7,
    )
    assert report["score_semantics"] == "global_x_state_x_reliability_x_observability"
    assert report["labels"][0]["pu_score_mean"] == torch.mean(private * state).item()
def test_heca_dynamic_audit_covers_live_state_path() -> None:
    report = _dynamic_checks()
    assert report["checks"]["action_progress_zero_equivalence"] is True
    assert report["checks"]["reason_progress_zero_equivalence"] is True
    assert report["checks"]["state_uniform_recomputes_values"] is True
    assert report["checks"]["one_dino_call"] is True
    assert report["pass"]
