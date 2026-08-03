import math

import pytest
import torch
import torch.nn.functional as F

from fate_oia.losses import save_grounding_losses as grounding_losses
from fate_oia.losses.save_grounding_losses import (
    DEFAULT_SAVE_GROUNDING_WEIGHTS,
    predicate_anchor_loss,
    predicate_matched_background_loss,
    predicate_mirror_loss,
    predicate_null_loss,
    predicate_state_loss,
    save_grounding_loss,
)
from fate_oia.models.save_predicate_measurement import SAVEPredicateMeasurement


def test_save_grounding_outputs_do_not_backpropagate_into_foundation_inputs() -> None:
    torch.manual_seed(11)
    measurement = SAVEPredicateMeasurement(dim=8)
    factor_nodes = torch.randn(2, 21, 8, requires_grad=True)
    patches = torch.randn(2, 3, 12, 8, requires_grad=True)

    output = measurement(factor_nodes, patches, progress=1.0)
    loss = output["predicate_map"].square().mean()
    loss = loss + output["predicate_state_prob"].square().mean()
    loss.backward()

    assert factor_nodes.grad is None or torch.count_nonzero(factor_nodes.grad) == 0
    assert patches.grad is None or torch.count_nonzero(patches.grad) == 0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in measurement.parameters()
    )


def _grounding_fixture():
    predicate_map = torch.tensor(
        [
            [[0.20, 0.50, 0.30], [0.60, 0.20, 0.20], [0.10, 0.20, 0.70]],
            [[0.40, 0.40, 0.20], [0.20, 0.30, 0.50], [0.30, 0.30, 0.40]],
        ]
    )
    state_logits = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    output: dict[str, torch.Tensor | tuple] = {
        "predicate_map_raw": predicate_map,
        "predicate_map": predicate_map,
        "predicate_null_mass": torch.tensor(
            [[0.20, 0.80, 0.50], [0.40, 0.30, 0.60]]
        ),
        "predicate_state_logits": state_logits,
        "predicate_state_prob_raw": torch.softmax(state_logits, dim=-1),
        "predicate_state_prob": torch.softmax(state_logits, dim=-1),
        "predicate_state_valid_mask": torch.ones(3, 3, dtype=torch.bool),
        "predicate_groundable_mask": torch.tensor([1.0, 1.0, 0.0]),
        "predicate_named_mask": torch.tensor([1.0, 0.5, 0.0]),
        "predicate_mirror_pairs": ((0, 1),),
    }
    anchor = torch.zeros(2, 3, 3)
    anchor[0, 0, 0] = 1.0
    anchor[0, 1, 2] = 1.0
    anchor[0, 2, 0] = 1.0
    anchor[1, 0, 0] = 1.0
    anchor[1, 1, 2] = 1.0
    anchor[1, 2, 0] = 1.0
    targets = {
        "predicate_anchor_map": anchor,
        "predicate_anchor_valid": torch.ones(2, 3, dtype=torch.bool),
        "predicate_state_target": torch.tensor([[0, 1, 0], [-1, 1, 0]]),
        "predicate_state_valid": torch.ones(2, 3, dtype=torch.bool),
        "predicate_present_valid": torch.tensor(
            [[True, False, True], [False, True, True]]
        ),
        "predicate_absent_valid": torch.tensor(
            [[False, True, False], [True, False, False]]
        ),
        "predicate_source_weight": torch.tensor(
            [[1.0, 0.5, 1.0], [0.25, 1.0, 1.0]]
        ),
        "predicate_provenance_valid": torch.tensor(
            [[True, True, True], [True, False, True]]
        ),
    }
    return output, targets


def test_save_grounding_components_match_exact_weighted_formulas() -> None:
    output, targets = _grounding_fixture()
    source_weights = torch.tensor([1.0, 0.5, 0.25])

    anchor_nll, anchor_dice = predicate_anchor_loss(output, targets)
    expected_nll = torch.tensor(
        [-math.log(0.20), -math.log(0.20), -math.log(0.40)]
    ) / math.log(3.0)
    expected_dice = torch.tensor([0.80, 0.80, 0.60])
    torch.testing.assert_close(
        anchor_nll,
        (expected_nll * source_weights).sum() / source_weights.sum(),
    )
    torch.testing.assert_close(
        anchor_dice,
        (expected_dice * source_weights).sum() / source_weights.sum(),
    )

    state = predicate_state_loss(output, targets)
    expected_state = torch.stack(
        [
            F.cross_entropy(
                output["predicate_state_logits"][0, 0].unsqueeze(0),
                torch.tensor([0]),
            ),
            F.cross_entropy(
                output["predicate_state_logits"][0, 1].unsqueeze(0),
                torch.tensor([1]),
            ),
        ]
    )
    torch.testing.assert_close(
        state,
        (expected_state * torch.tensor([1.0, 0.5])).sum() / 1.5,
    )

    null = predicate_null_loss(output, targets)
    expected_null = torch.tensor(
        [
            -math.log(1.0 - 0.20),
            -math.log(0.80),
            -math.log(0.40),
        ]
    )
    torch.testing.assert_close(
        null,
        (expected_null * source_weights).sum() / source_weights.sum(),
    )

    matched_background = predicate_matched_background_loss(output, targets)
    expected_background = torch.tensor([0.22, 0.22, 0.0])
    background_weights = torch.tensor([1.0, 0.25, 0.25])
    torch.testing.assert_close(
        matched_background,
        (expected_background * background_weights).sum()
        / background_weights.sum(),
    )


def test_save_grounding_requires_explicit_bdd100k_train_provenance() -> None:
    output, targets = _grounding_fixture()

    with pytest.raises(TypeError):
        save_grounding_loss(output, targets)
    with pytest.raises(ValueError, match="BDD100K"):
        save_grounding_loss(
            output,
            targets,
            split="train",
            supervision_source="bdd_oia",
        )
    with pytest.raises(ValueError, match="train"):
        save_grounding_loss(
            output,
            targets,
            split="test",
            supervision_source=grounding_losses.BDD100K_SUPERVISION_SOURCE,
        )
    missing_provenance = dict(targets)
    del missing_provenance["predicate_provenance_valid"]
    with pytest.raises(KeyError, match="provenance"):
        save_grounding_loss(
            output,
            missing_provenance,
            split="train",
            supervision_source=grounding_losses.BDD100K_SUPERVISION_SOURCE,
        )


def test_save_grounding_total_uses_exact_default_weights_once() -> None:
    output, targets = _grounding_fixture()
    result = save_grounding_loss(
        output,
        targets,
        split="train",
        supervision_source=grounding_losses.BDD100K_SUPERVISION_SOURCE,
    )
    assert DEFAULT_SAVE_GROUNDING_WEIGHTS == {
        "anchor": 0.05,
        "state": 0.08,
        "null": 0.02,
        "matched_background": 0.03,
        "mirror": 0.02,
        "identity": 0.02,
    }
    expected = sum(
        DEFAULT_SAVE_GROUNDING_WEIGHTS[name] * result[name]
        for name in DEFAULT_SAVE_GROUNDING_WEIGHTS
    )
    torch.testing.assert_close(result["total"], expected)


def test_save_mirror_loss_respects_validity_and_validates_pairs() -> None:
    original_map = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [10.0] * 6, [20.0] * 6]]
    )
    original_state = torch.tensor(
        [[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]]
    )
    original = {
        "predicate_map_raw": original_map,
        "predicate_state_prob_raw": original_state,
    }
    mirrored_map = torch.zeros_like(original_map)
    mirrored_state = torch.zeros_like(original_state)
    # Independent row-major 2x3 horizontal mirror.  A flat reversal would
    # incorrectly exchange the two rows and produce [6, 5, 4, 3, 2, 1].
    mirrored_map[:, 1] = torch.tensor([3.0, 2.0, 1.0, 6.0, 5.0, 4.0])
    mirrored_state[:, 1] = original_state[:, 0]
    mirrored = {
        "predicate_map_raw": mirrored_map,
        "predicate_state_prob_raw": mirrored_state,
    }

    valid_first_only = torch.tensor([[1.0, 0.0, 0.0]])
    assert predicate_mirror_loss(
        original,
        mirrored,
        mirror_pairs=((0, 1),),
        valid=valid_first_only,
        grid_shape=(2, 3),
    ) == 0
    assert predicate_mirror_loss(
        original,
        mirrored,
        mirror_pairs=((0, 1),),
        valid=torch.tensor([[1.0, 1.0, 0.0]]),
        grid_shape=(2, 3),
    ) > 0
    with pytest.raises(ValueError, match="mirror pair"):
        predicate_mirror_loss(
            original,
            mirrored,
            mirror_pairs=((0, 3),),
            valid=valid_first_only,
            grid_shape=(2, 3),
        )
    with pytest.raises(ValueError, match="grid"):
        predicate_mirror_loss(
            original,
            mirrored,
            mirror_pairs=((0, 1),),
            valid=valid_first_only,
            grid_shape=(2, 2),
        )
