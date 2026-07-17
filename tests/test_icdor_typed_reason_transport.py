from __future__ import annotations

from pathlib import Path

import torch

from fate_oia.models.mosaic_icdor_dual_reason_decoder import MOSAICICDORLatentReasonDecoder
from fate_oia.models.mosaic_native_semantics import load_icdor_ontology


def test_latent_reason_decoder_reads_typed_factor_samples() -> None:
    """Fine typed samples must alter the latent reason reread, not only masks."""
    torch.manual_seed(7)
    ontology = load_icdor_ontology(Path("configs"))
    decoder = MOSAICICDORLatentReasonDecoder(
        ontology,
        dim=32,
        decoder_layers=1,
        self_attention_heads=4,
        highres_topk=8,
        midres_topk=4,
    ).eval()
    factor_count = len(ontology["factors"])
    reason_pyramid = {
        "F_hi": torch.randn(1, 32, 45, 80),
        "F_mid": torch.randn(1, 32, 23, 40),
        "F_ctx": torch.randn(1, 32, 12, 20),
    }
    factor_features = torch.randn(1, factor_count, 32)
    factor_masks = torch.rand(1, factor_count, 45, 80)
    coordinates = torch.rand(1, factor_count, 2, 2, 3, 2).mul_(2.0).sub_(1.0)
    sampled_features = torch.randn(1, factor_count, 2, 2, 3, 32)
    sample_attention = torch.rand(1, factor_count, 2, 2, 3)

    with torch.no_grad():
        full = decoder(
            reason_pyramid,
            factor_features,
            factor_masks,
            torch.ones(factor_count, dtype=torch.bool),
            sampling_coordinates=coordinates,
            sampled_features=sampled_features,
            sample_attention=sample_attention,
        )
        changed = decoder(
            reason_pyramid,
            factor_features,
            factor_masks,
            torch.ones(factor_count, dtype=torch.bool),
            sampling_coordinates=coordinates,
            sampled_features=sampled_features.roll(1, dims=1),
            sample_attention=sample_attention,
        )

    assert full["reason_typed_transport_nodes"].shape == (1, 21, 32)
    assert not torch.allclose(full["reason_logits_latent"], changed["reason_logits_latent"])
