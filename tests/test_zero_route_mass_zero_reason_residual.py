from __future__ import annotations

import torch

from fate_oia.models.mosaic_icdor_dual_reason_decoder import MOSAICICDORObservedReasonMixer


def test_zero_route_mass_zero_reason_residual() -> None:
    visual = torch.randn(2, 21)
    annotation = torch.randn(2, 21)
    mixer = MOSAICICDORObservedReasonMixer(init_mix=0.05)
    output = mixer(visual, annotation, latent_enabled=True, route_mass=torch.zeros(2, 21))
    assert torch.equal(output["reason_observed_logits"], visual)
