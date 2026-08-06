import torch

from fate_oia.models.aie_predicate_naming import AIEPredicateNaming, spatial_soft_iou


def test_name_requires_confidence_margin_and_presence():
    module = AIEPredicateNaming(dim=32, num_predicates=32)
    out = module(torch.randn(1, 4, 4, 32), torch.full((1, 4, 4, 20), 0.05), torch.full((1, 32, 20), 0.05), torch.zeros(1, 32))
    assert bool((out["name_id"] == -1).all())


def test_spatial_soft_iou_uses_salient_support_not_probability_mass():
    localized = torch.tensor([[0.01, 0.01, 0.97, 0.01]])
    disjoint = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
    uniform = torch.full((1, 4), 0.25)
    torch.testing.assert_close(spatial_soft_iou(localized, localized), torch.ones(1))
    torch.testing.assert_close(spatial_soft_iou(localized, disjoint), torch.zeros(1))
    torch.testing.assert_close(spatial_soft_iou(uniform, uniform), torch.zeros(1))


def test_name_accepts_one_clear_localized_predicate_match():
    module = AIEPredicateNaming(dim=8, num_predicates=3)
    with torch.no_grad():
        module.evidence_projection.weight.zero_()
        module.evidence_projection.bias.zero_()
        module.predicate_keys.zero_()
    evidence_map = torch.tensor([[[[0.01, 0.01, 0.97, 0.01]]]])
    predicate_map = torch.tensor(
        [[[0.01, 0.01, 0.97, 0.01], [0.97, 0.01, 0.01, 0.01], [0.01, 0.97, 0.01, 0.01]]]
    )
    out = module(
        torch.zeros(1, 1, 1, 8),
        evidence_map,
        predicate_map,
        torch.tensor([[1.0, 0.1, 0.1]]),
    )
    assert out["name_id"].item() == 0
    assert out["name_confidence"].item() >= 0.45


def test_compatibility_uses_the_planned_unscaled_inner_product():
    module = AIEPredicateNaming(dim=4, num_predicates=2)
    with torch.no_grad():
        module.evidence_projection.weight.zero_()
        module.evidence_projection.bias.fill_(0.125)
        module.predicate_keys.fill_(0.125)
    out = module(
        torch.zeros(1, 1, 1, 4),
        torch.tensor([[[[0.0, 1.0]]]]),
        torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
        torch.ones(1, 2),
    )
    # 64 projected dimensions * 0.125 * 0.125 = 1.0.
    torch.testing.assert_close(
        out["name_compatibility"][0, 0, 0, 0],
        torch.sigmoid(torch.tensor(1.0)),
        atol=1e-6,
        rtol=0,
    )
