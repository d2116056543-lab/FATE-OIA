from __future__ import annotations

import inspect
from pathlib import Path

import torch

from fate_oia.models.mosaic_icdor_dual_reason_decoder import (
    MOSAICICDORLatentReasonDecoder,
    MOSAICICDORObservedReasonMixer,
    MOSAICICDORVisualReasonDecoder,
)
from fate_oia.models.mosaic_icdor_observation_head import MOSAICICDORObservationHead
from fate_oia.models.mosaic_native_semantics import load_icdor_ontology
from fate_oia.losses.mosaic_icdor_reason_losses import reason_observed_losses


def _pyramid(batch_size: int, dim: int) -> dict[str, torch.Tensor]:
    return {
        "F_hi": torch.randn(batch_size, dim, 45, 80),
        "F_mid": torch.randn(batch_size, dim, 23, 40),
        "F_ctx": torch.randn(batch_size, dim, 12, 20),
    }


def test_icdor_dual_reason_paths_and_observation_posterior_are_separated() -> None:
    ontology = load_icdor_ontology(Path("configs"))
    factor_count = len(ontology["factors"])
    visual = MOSAICICDORVisualReasonDecoder(dim=32, highres_topk=64, midres_topk=32)
    latent = MOSAICICDORLatentReasonDecoder(ontology, dim=32, highres_topk=64, midres_topk=32)
    observation = MOSAICICDORObservationHead(ontology, pi_min=0.20, pi_max=0.95)
    mixer = MOSAICICDORObservedReasonMixer()

    reason_pyramid = _pyramid(batch_size=2, dim=32)
    visual_out = visual(reason_pyramid)
    factors = torch.randn(2, factor_count, 32, requires_grad=True)
    masks = torch.rand(2, factor_count, 45, 80, requires_grad=True)
    latent_out = latent(
        reason_pyramid,
        factors,
        masks,
        torch.ones(factor_count, dtype=torch.bool),
    )
    observation_out = observation(
        latent_out["reason_logits_latent"],
        torch.rand(2, factor_count),
        torch.rand(2, factor_count),
    )
    mixed = mixer(
        visual_out["reason_visual_observed_logits"],
        observation_out["reason_observation_logits"],
        latent_enabled=True,
    )

    assert visual_out["reason_visual_observed_logits"].shape == (2, 21)
    assert latent_out["reason_logits_latent"].shape == (2, 21)
    assert latent_out["reason_factor_router_weights"].shape == (2, 21, factor_count)
    assert torch.count_nonzero(
        latent_out["reason_factor_router_weights"] * (~latent.reason_factor_allow_mask).unsqueeze(0)
    ) == 0
    assert mixed["reason_observed_logits"].shape == (2, 21)

    targets = torch.randint(0, 2, (2, 21), dtype=torch.float32)
    posterior = observation.posterior_from_observed_targets(
        latent_out["reason_logits_latent"], targets, observation_out
    )
    assert posterior["reason_latent_posterior"].requires_grad is False
    assert torch.all((posterior["reason_latent_posterior"] >= 0) & (posterior["reason_latent_posterior"] <= 1))
    assert "observed_reason_targets" not in inspect.signature(observation.forward).parameters


def test_direct_observed_reason_asl_is_balanced_within_each_label() -> None:
    """Duplicating known negatives must not dilute a rare observed positive."""
    logits = torch.zeros(2, 21)
    logits[1, 0] = 2.0
    targets = torch.zeros(2, 21)
    targets[0, 0] = 1.0
    valid = torch.zeros(2, 21, dtype=torch.bool)
    valid[:, 0] = True
    base = reason_observed_losses(logits, logits, targets, observed_valid_mask=valid)

    duplicated_logits = torch.cat((logits[:1], logits[1:].repeat(4, 1)), dim=0)
    duplicated_targets = torch.cat((targets[:1], targets[1:].repeat(4, 1)), dim=0)
    duplicated_valid = torch.cat((valid[:1], valid[1:].repeat(4, 1)), dim=0)
    duplicated = reason_observed_losses(
        duplicated_logits,
        duplicated_logits,
        duplicated_targets,
        observed_valid_mask=duplicated_valid,
    )

    assert torch.allclose(
        base["loss_reason_visual_observed_asl"],
        duplicated["loss_reason_visual_observed_asl"],
    )
