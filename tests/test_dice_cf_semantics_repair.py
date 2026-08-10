import torch
from torch import nn

from fate_oia.models.dice_oia_model import DICEOIAModel
from fate_oia.utils.dice_counterfactual_engine import counterfactual_logit_drop


class _Atoms(nn.Module):
    def forward(self, evidence, conditioned, attention, probabilities, masks):
        token = conditioned.mean((1, 2))[:, None, None, :]
        return {
            "centered_token": token,
            "coherent_map": conditioned.new_ones(conditioned.shape[0], 1, 1, 1),
            "predicate_agreement": conditioned.new_ones(conditioned.shape[0], 1, 1),
            "predicate_confidence": conditioned.new_ones(conditioned.shape[0], 1, 1),
        }


class _Direction(nn.Module):
    def forward(self, token, base, legacy, **kwargs):
        delta = token[:, 0, 0, :4]
        return {"dice_action_delta": delta, "dice_action_logits": base + delta}


def _rerun_base():
    return {
        "action_logits_base": torch.full((1, 4), 10.0),
        "action_logits_final": torch.full((1, 4), 100.0),
        "bounded_contribution": torch.zeros(1, 4, 1),
        "evidence_token": torch.zeros(1, 1, 4),
        "predicate_attention": torch.ones(1, 1, 1),
        "predicate_probs": torch.ones(1, 1),
        "ego_region_masks": {},
    }


def test_dice_cf_rerun_uses_explicit_action_logits_base():
    model = DICEOIAModel.__new__(DICEOIAModel)
    nn.Module.__init__(model)
    model.atom_reconstructor = _Atoms()
    model.directional_head = _Direction()
    conditioned = torch.ones(1, 1, 1, 4)
    rerun = model.rerun_dice_from_conditioned(_rerun_base(), conditioned)
    assert torch.equal(rerun["dice_action_logits"], torch.full((1, 4), 11.0))


def test_dice_cf_zero_intervention_identity_and_exact_delta_difference():
    original = torch.tensor([0.2, -0.1])
    modified = original.clone()
    sign = torch.tensor([1.0, -1.0])
    assert torch.equal(counterfactual_logit_drop(original, modified, sign), torch.zeros(2))
    modified = torch.tensor([-0.1, 0.3])
    expected = sign * (original - modified)
    assert torch.equal(counterfactual_logit_drop(original, modified, sign), expected)


def test_dice_cf_delta_difference_is_invariant_to_base_shift():
    base_a = torch.tensor([2.0])
    base_b = torch.tensor([200.0])
    original, modified, sign = torch.tensor([0.2]), torch.tensor([-0.1]), torch.tensor([1.0])
    direct = counterfactual_logit_drop(original, modified, sign)
    assert torch.allclose(direct, sign * ((base_a + original) - (base_a + modified)), atol=1e-6)
    assert torch.allclose(direct, sign * ((base_b + original) - (base_b + modified)), atol=1e-5)
