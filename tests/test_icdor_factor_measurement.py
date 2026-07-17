from __future__ import annotations

from pathlib import Path

import torch

from fate_oia.models.mosaic_native_semantics import load_icdor_ontology
from fate_oia.models.mosaic_observable_predicates import MOSAICObservablePredicateLayer


def _pyramid(batch_size: int, dim: int) -> dict[str, torch.Tensor]:
    return {
        "F_hi": torch.randn(batch_size, dim, 45, 80),
        "F_mid": torch.randn(batch_size, dim, 23, 40),
        "F_ctx": torch.randn(batch_size, dim, 12, 20),
    }


def test_icdor_factor_measurement_uses_observable_schema_and_independent_prototypes() -> None:
    ontology = load_icdor_ontology(Path("configs"))
    layer = MOSAICObservablePredicateLayer(
        ontology["factors"],
        dim=32,
        point_samples=4,
        curve_samples=16,
        region_samples=12,
    )

    outputs = layer(_pyramid(batch_size=2, dim=32))
    factor_count = len(ontology["factors"])
    assert outputs["factor_presence_prob"].shape == (2, factor_count)
    assert outputs["factor_visibility_prob"].shape == (2, factor_count)
    assert outputs["factor_positive_evidence"].shape == (2, factor_count)
    assert outputs["factor_negative_evidence"].shape == (2, factor_count)
    assert outputs["factor_soft_masks"].shape == (2, factor_count, 45, 80)
    assert outputs["prototype_weights"].shape[:2] == (2, factor_count)
    assert outputs["prototype_scores"].shape[:3] == (2, factor_count, layer.prototype_bank.max_prototypes)
    fine_support = outputs["measurement_stats"]["fine_prototype_support"]
    mid_support = outputs["measurement_stats"]["mid_prototype_support"]
    assert fine_support.shape == (2, factor_count, layer.prototype_bank.max_prototypes)
    assert mid_support.shape == (2, factor_count, layer.prototype_bank.max_prototypes)
    valid = layer.prototype_bank.prototype_valid_mask
    assert torch.all(fine_support[:, valid] > 0)
    assert torch.all(mid_support[:, valid] > 0)
    assert torch.allclose(
        outputs["factor_positive_evidence"] + outputs["factor_negative_evidence"],
        outputs["factor_visibility_prob"],
        atol=1e-6,
    )
    assert layer.typed_attention.curve_sequence_encoder.num_layers == 2
    assert layer.typed_attention.point_samples == 4
    assert layer.typed_attention.curve_samples == 16


def test_icdor_invisible_factor_has_near_zero_present_and_absent_evidence() -> None:
    ontology = load_icdor_ontology(Path("configs"))
    layer = MOSAICObservablePredicateLayer(
        ontology["factors"],
        dim=32,
        point_samples=4,
        curve_samples=16,
        region_samples=12,
    )
    with torch.no_grad():
        layer.visibility_head.weight.zero_()
        layer.visibility_head.bias.fill_(-20.0)

    outputs = layer(_pyramid(batch_size=1, dim=32))
    assert float(outputs["factor_positive_evidence"].max()) < 1e-6
    assert float(outputs["factor_negative_evidence"].max()) < 1e-6


def test_query_shuffle_remeasures_with_permuted_factor_queries() -> None:
    torch.manual_seed(7)
    ontology = load_icdor_ontology(Path("configs"))
    layer = MOSAICObservablePredicateLayer(
        ontology["factors"],
        dim=32,
        point_samples=4,
        curve_samples=16,
        region_samples=12,
    ).eval()
    pyramid = _pyramid(batch_size=2, dim=32)
    permutation = torch.roll(torch.arange(len(ontology["factors"])), shifts=1)

    with torch.no_grad():
        full = layer(pyramid)
        shuffled = layer(pyramid, query_permutation=permutation)

    assert not torch.allclose(full["factor_presence_prob"], shuffled["factor_presence_prob"])
    assert not torch.equal(shuffled["factor_presence_prob"], full["factor_presence_prob"].roll(1, 1))
