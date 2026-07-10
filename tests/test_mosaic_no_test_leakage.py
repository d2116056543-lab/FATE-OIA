from __future__ import annotations

import inspect

import torch
from torch import nn

from fate_oia.engine.eval_acpr_mosaic_ad import evaluate_mosaic
from fate_oia.models.acpr_mosaic_ad_model import MOSAICADModel
from fate_oia.models.mosaic_group_threshold import MOSAICGroupThresholdHead


class _EvaluationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.schema_bundle = {
            "factors": [{"name": "factor_a"}, {"name": "factor_b"}],
            "states": {"state_a": {}, "state_b": {}},
        }

    def forward(self, images, *, return_masks=False, prior_mode="full"):
        batch = images.shape[0]
        action = images.mean((1, 2, 3), keepdim=False).unsqueeze(-1).expand(batch, 4)
        reason = action[:, :1].expand(batch, 21)
        factor_prob = torch.full((batch, 2), 0.6, device=images.device)
        state_prob = torch.full((batch, 2), 0.4, device=images.device)
        return {
            "action_logits_visual": action,
            "action_logits_state": action * 0.1,
            "action_logits_raw": action,
            "reason_logits_latent": reason,
            "factor_presence_prob": factor_prob,
            "factor_visibility_prob": factor_prob,
            "factor_positive_evidence": factor_prob.square(),
            "factor_negative_evidence": factor_prob * (1-factor_prob),
            "factor_uncertainty": torch.full_like(factor_prob, 0.5),
            "factor_soft_masks": torch.full((batch, 2, 45, 80), 0.1, device=images.device),
            "decision_state_prob": state_prob,
            "decision_state_uncertainty": torch.full_like(state_prob, 0.5),
            "measurement_stats": {},
        }


def _loader():
    return [
        {
            "image": torch.randn(3, 3, 16, 16),
            "action": torch.randint(0, 2, (3, 4)).float(),
            "reason": torch.randint(0, 2, (3, 21)).float(),
            "file_name": ["a.jpg", "b.jpg", "c.jpg"],
            "split": ["test", "test", "test"],
        }
    ]


def test_formal_forward_accepts_no_labels_geometry_or_threshold_metadata() -> None:
    parameters = inspect.signature(MOSAICADModel.forward).parameters
    assert list(parameters) == ["self", "images", "prior_mode", "return_masks"]
    forbidden = {"action", "reason", "labels", "geometry", "threshold", "metadata"}
    assert not forbidden & set(parameters)


def test_test_oracle_diagnostic_never_mutates_calibrator() -> None:
    model = _EvaluationModel()
    threshold = MOSAICGroupThresholdHead()
    before = {name: value.clone() for name, value in threshold.state_dict().items()}
    result = evaluate_mosaic(model, threshold, _loader(), torch.device("cpu"), epoch=0)
    assert result["metrics_summary"]["test_oracle_diagnostic"]["writeback_allowed"] is False
    for name, value in threshold.state_dict().items():
        assert torch.equal(value, before[name])


def test_non_test_batch_fails_closed() -> None:
    batch = _loader()[0]
    batch["split"] = ["val", "val", "val"]
    try:
        evaluate_mosaic(_EvaluationModel(), MOSAICGroupThresholdHead(), [batch], torch.device("cpu"), epoch=0)
    except ValueError as error:
        assert "test batches only" in str(error)
    else:
        raise AssertionError("validation data was accepted by formal test-only evaluation")
