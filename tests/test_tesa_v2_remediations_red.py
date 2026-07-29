import inspect

import torch

from fate_oia.engine import train_acpr_meter_oia as trainer
from fate_oia.losses.meter_action_losses import meter_action_loss
from fate_oia.models.meter_oia_model import cross_sample_swap_typed_evidence
from fate_oia.models.meter_semantic_action import FactorSpecificActionTransport


def test_action_factor_contributions_are_bounded_and_exactly_additive() -> None:
    torch.manual_seed(17)
    module = FactorSpecificActionTransport(dim=8, rank=4)
    out = module(
        torch.randn(2, 4),
        torch.randn(2, 4, 8),
        torch.randn(2, 21, 8),
        torch.ones(2, 21),
        torch.ones(21),
        progress=1.0,
    )

    contribution = out["action_factor_contributions"]
    kappa = out["action_correction_kappa"].view(1, 4, 1)
    assert torch.all(contribution.abs() <= kappa + 1e-6)
    torch.testing.assert_close(
        out["action_logits_final"],
        out["action_logits_visual"] + contribution.sum(-1),
        atol=1e-6,
        rtol=1e-6,
    )
    deleted = out["action_logits_factor_deleted"]
    for factor_id in (0, 7, 20):
        torch.testing.assert_close(
            deleted[:, :, factor_id],
            out["action_logits_final"] - contribution[:, :, factor_id],
            atol=1e-6,
            rtol=1e-6,
        )


def test_dense_specificity_is_not_added_by_action_loss() -> None:
    base = {
        "action_logits_final": torch.zeros(2, 4),
        "action_logits_visual": torch.zeros(2, 4),
        "dense_identity_loss": torch.zeros(()),
    }
    weights = {
        "action_final": 0.0,
        "action_visual": 0.0,
        "action_correction": 0.0,
        "action_two_way": 0.0,
        "action_soft_f1": 0.0,
        "action_cardinality": 0.0,
        "action_specificity": 1.0,
        "action_identity": 0.0,
    }
    target = torch.zeros(2, 4)
    low = meter_action_loss({**base, "dense_specificity_loss": torch.tensor(0.0)}, target, weights)
    high = meter_action_loss({**base, "dense_specificity_loss": torch.tensor(3.0)}, target, weights)
    torch.testing.assert_close(low["total"], high["total"])


def test_cross_sample_swap_moves_complete_typed_evidence() -> None:
    evidence = {
        "factor_typed_token": torch.arange(3 * 2 * 4).view(3, 2, 4).float(),
        "factor_state_prob": torch.arange(3 * 2 * 3).view(3, 2, 3).float(),
        "factor_reliability": torch.arange(3 * 2).view(3, 2).float(),
        "factor_observability": (torch.arange(3 * 2).view(3, 2).float() + 10),
    }
    swapped = cross_sample_swap_typed_evidence(evidence)
    for key, value in evidence.items():
        torch.testing.assert_close(swapped[key], torch.roll(value, 1, 0))


def test_training_identity_intervention_includes_reason_corruption() -> None:
    source = inspect.getsource(trainer._compute_losses)
    assert "reason_identity_corruption_loss" in source
    assert "reason_logits_final" in source
    assert '("schema", "cross_sample", "state")' in source
    assert "reason_identity_terms" in source


def test_training_adds_dense_necessity_and_specificity_exactly_once() -> None:
    source = inspect.getsource(trainer._compute_losses)
    assert 'dense_weight * mechanism_ramp * dense["total"]' in source
    assert 'dense_weight * mechanism_ramp * dense["necessity"]' not in source


def test_training_uses_one_encode_call_for_paired_mirror_constraint() -> None:
    source = inspect.getsource(trainer._forward_training_batch)
    assert source.count("model.encode_images(") == 1
    assert "torch.flip(images[:1]" in source
    assert "mirror_output" in source
