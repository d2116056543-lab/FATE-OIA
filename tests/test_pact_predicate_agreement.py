import torch

from fate_oia.models.pact_predicate_agreement import PACTPredicateAgreement


def test_agreement_is_bounded_and_penalizes_mismatched_maps():
    module = PACTPredicateAgreement(0.25)
    visual = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    matched = module(visual, visual, torch.ones(1, 2))
    mismatched = module(visual, visual.flip(-1), torch.ones(1, 2))
    assert matched["predicate_visual_agreement"].mean() > mismatched["predicate_visual_agreement"].mean()
    assert matched["predicate_agreement_strength"].min() >= 0
    assert matched["predicate_agreement_strength"].max() <= 0.25
