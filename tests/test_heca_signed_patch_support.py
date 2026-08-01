import torch

from fate_oia.engine.tesa_diagnostics import (
    select_coverage_stratified_supporting_factors,
    select_target_supporting_factors,
)


def test_patch_audit_only_deletes_positive_target_support() -> None:
    contribution = torch.tensor([0.03, -0.40, 0.02, -0.01])
    eligible = torch.tensor([True, True, False, True])

    selected = select_target_supporting_factors(
        contribution, eligible, factors_per_action=2
    )

    # A negative contribution suppresses the predicted target and is not
    # evidence supporting the target in a deletion-faithfulness test.
    assert selected == [0]


def test_patch_audit_orders_support_by_signed_effect() -> None:
    contribution = torch.tensor([0.02, 0.09, 0.03, -0.20])
    eligible = torch.ones(4, dtype=torch.bool)

    assert select_target_supporting_factors(
        contribution, eligible, factors_per_action=2
    ) == [1, 2]


def test_patch_audit_reserves_secondary_slot_for_uncovered_positive_factor() -> None:
    contribution = torch.tensor([0.90, 0.80, 0.30, -0.50])
    eligible = torch.ones(4, dtype=torch.bool)

    selected = select_coverage_stratified_supporting_factors(
        contribution,
        eligible,
        factors_per_action=2,
        execution_counts={0: 9, 1: 7, 2: 0},
    )

    # Factor 0 remains the true model-top evidence. The secondary audit slot
    # covers the least-observed positive factor, never a negative contribution.
    assert selected == [0, 2]
