from __future__ import annotations

import torch

from fate_oia.models.mosaic_selective_observation import MOSAICSelectiveObservationModel


def _model() -> MOSAICSelectiveObservationModel:
    factors = ("visible", "lane")
    mapping = {
        reason: {
            "group": ("traffic_control", "obstacle", "lane", "other")[reason % 4],
            "support_factors": [factors[reason % 2]],
            "visibility_factors": [factors[reason % 2]],
            "false_positive_max": 0.05,
        }
        for reason in range(21)
    }
    return MOSAICSelectiveObservationModel(factors, mapping)


def test_inference_surface_is_label_free_and_posterior_is_stop_gradient() -> None:
    model = _model()
    latent = torch.randn(3, 21, requires_grad=True)
    visibility = torch.rand(3, 2, requires_grad=True)
    uncertainty = torch.rand(3, 2, requires_grad=True)
    output = model.forward_inference(latent, visibility, uncertainty)
    assert output["reason_propensity"].shape == (3, 21)
    assert 0.20 <= float(output["reason_propensity"].min())
    assert float(output["reason_propensity"].max()) <= 0.95
    posterior = model.posterior_from_observed_targets(
        latent, torch.zeros(3, 21), output
    )["reason_latent_posterior"]
    assert posterior.shape == (3, 21)
    assert posterior.requires_grad is False
    assert visibility.grad is None and uncertainty.grad is None
