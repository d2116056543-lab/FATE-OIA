from __future__ import annotations

import inspect

import torch

from fate_oia.engine.audit_acpr_meter_oia import _dynamic_checks
from fate_oia.engine.train_acpr_meter_oia import _compute_losses
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


def test_trainer_passes_yaml_grounding_weights_to_the_loss() -> None:
    source = inspect.getsource(_compute_losses)
    assert 'weights=config["loss_weights"]' in source


def test_dynamic_audit_covers_grounding_mirror_and_full_trainer_gradients() -> None:
    report = _dynamic_checks(torch.device("cpu"))
    assert "grounding_gradient_ownership" in report
    assert "mirror_gradient_ownership" in report
    assert set(report["mirror_gradient_ownership"]["components"]) == {
        "anchor",
        "state",
        "action",
        "reason",
    }
    assert all(
        component["pass"]
        for component in report["mirror_gradient_ownership"]["components"].values()
    )
    assert "trainer_total_gradient_ownership" in report
    assert report["pass"]
