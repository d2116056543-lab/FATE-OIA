import torch
import torch.nn.functional as F

from fate_oia.losses.save_faithfulness_losses import (
    dense_utility_auxiliary_loss,
    save_faithfulness_losses,
    utility_counterfactual_loss,
    utility_teacher_target,
)
from fate_oia.models.save_utility_bridge import (
    build_sparse_counterfactual_teacher,
    build_sparse_teacher_plan,
    select_sparse_teacher_targets,
)


def test_sparse_teacher_uses_at_most_two_samples_one_action_and_two_candidates() -> None:
    base_logits = torch.tensor(
        [[0.10, -0.80, 0.20], [0.05, 0.02, -0.01], [-0.90, 0.40, 0.30]]
    )
    labels = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]
    )
    candidates = torch.tensor(
        [
            [[0.1, 0.6, 0.2, 0.1], [0.2, 0.1, 0.6, 0.1], [0.1, 0.2, 0.5, 0.2]],
            [[0.1, 0.5, 0.3, 0.1], [0.1, 0.2, 0.6, 0.1], [0.2, 0.2, 0.5, 0.1]],
            [[0.1, 0.4, 0.4, 0.1], [0.1, 0.2, 0.6, 0.1], [0.1, 0.3, 0.5, 0.1]],
        ]
    )
    reliability = torch.ones(3, 3)
    overlap = torch.tensor(
        [
            [[0.9, 0.8, 0.1], [0.1, 0.2, 0.9], [0.2, 0.3, 0.8]],
            [[0.2, 0.9, 0.8], [0.1, 0.2, 0.9], [0.3, 0.4, 0.8]],
            [[0.8, 0.7, 0.1], [0.1, 0.2, 0.9], [0.2, 0.3, 0.8]],
        ]
    )

    targets = select_sparse_teacher_targets(base_logits, labels, max_samples=2)
    plan_a = build_sparse_teacher_plan(
        base_logits,
        labels,
        candidates,
        reliability,
        overlap,
        utility_logit=torch.randn(3, 3, 3),
        max_samples=2,
    )
    plan_b = build_sparse_teacher_plan(
        base_logits,
        labels,
        candidates,
        reliability,
        overlap,
        utility_logit=torch.randn(3, 3, 3) * 100.0,
        max_samples=2,
    )

    assert targets["sample_indices"].numel() <= 2
    assert plan_a["selected_sample_indices"].numel() <= 2
    assert torch.equal(plan_a["sample_indices"], plan_b["sample_indices"])
    assert torch.equal(plan_a["action_indices"], plan_b["action_indices"])
    assert torch.equal(plan_a["factor_indices"], plan_b["factor_indices"])
    assert plan_a["sample_indices"].ndim == 1
    assert plan_a["sample_indices"].shape == plan_a["action_indices"].shape
    assert plan_a["sample_indices"].shape == plan_a["factor_indices"].shape
    assert plan_a["sample_indices"].numel() <= 4
    assert plan_a["selection_source"] == "candidate_weight*reliability*base_overlap"
    torch.testing.assert_close(
        plan_a["candidate_scores"],
        plan_a["selected_candidate_weight"]
        * plan_a["selected_reliability"]
        * plan_a["selected_base_overlap"],
    )
    assert plan_a["utility_predictor_used_for_selection"] is False


def test_exact_teacher_formula_sparse_gather_dense_auxiliary_and_detach() -> None:
    control_margin = torch.tensor([0.30, -0.10])
    selected_margin = torch.tensor([-0.20, -0.40])
    expected_target = torch.sigmoid((control_margin - selected_margin) / 0.10)
    torch.testing.assert_close(
        utility_teacher_target(control_margin, selected_margin),
        expected_target,
    )

    utility = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    utility.requires_grad_()
    teacher_target = torch.tensor([0.25, 0.75], requires_grad=True)
    sample = torch.tensor([0, 1])
    action = torch.tensor([2, 0])
    factor = torch.tensor([3, 1])
    gathered = utility[sample, action, factor]
    expected_cf = F.smooth_l1_loss(gathered, teacher_target.detach())
    cf = utility_counterfactual_loss(
        utility,
        teacher_target,
        sample_indices=sample,
        action_indices=action,
        factor_indices=factor,
    )
    torch.testing.assert_close(cf, expected_cf)
    cf.backward()
    assert teacher_target.grad is None
    assert torch.count_nonzero(utility.grad) == 2

    dense_logit = torch.tensor([[[0.1, -0.2], [0.3, -0.4]]], requires_grad=True)
    named = torch.tensor([[[0.5, -0.1], [0.2, 0.6]]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0]])
    dense_target = (labels.unsqueeze(-1) * 2.0 - 1.0) * named.detach()
    expected_dense = 0.02 * F.smooth_l1_loss(dense_logit, dense_target)
    dense = dense_utility_auxiliary_loss(dense_logit, named, labels)
    torch.testing.assert_close(dense, expected_dense)
    dense.backward()
    assert named.grad is None

    plan = {
        "sample_indices": sample,
        "action_indices": action,
        "factor_indices": factor,
        "utility_teacher_target": teacher_target.detach(),
    }
    output = {
        "utility_logit": utility.detach().clone().requires_grad_(),
        "action_named_contribution": torch.zeros_like(utility),
        "teacher_plan": plan,
    }
    losses = save_faithfulness_losses(output, torch.zeros(2, 3))
    assert set(losses) == {
        "loss_utility_cf",
        "loss_utility_dense",
        "loss_utility",
        "total",
        "utility_teacher_prediction",
        "utility_teacher_target",
        "utility_dense_target",
    }
    torch.testing.assert_close(
        losses["utility_teacher_prediction"],
        output["utility_logit"][sample, action, factor],
    )
    assembled_cf = 0.10 * F.smooth_l1_loss(
        output["utility_logit"][sample, action, factor],
        teacher_target.detach(),
    )
    assembled_dense = 0.02 * F.smooth_l1_loss(
        output["utility_logit"],
        torch.zeros_like(output["utility_logit"]),
    )
    torch.testing.assert_close(losses["loss_utility_cf"], assembled_cf)
    torch.testing.assert_close(losses["loss_utility_dense"], assembled_dense)
    torch.testing.assert_close(losses["total"], assembled_cf + assembled_dense)


def _matched_teacher_inputs():
    grid_hw = (10, 30)
    patches = grid_hw[0] * grid_hw[1]
    detail = torch.zeros(1, patches, 4)
    detail[..., 0] = 3.0
    predicate_map = torch.zeros(1, 2, patches)
    contribution = torch.zeros(1, 2, patches)

    factor_regions = []
    for factor, (rows, columns) in enumerate(
        (((2, 3), (12, 13, 14, 15)), ((4, 5), (22, 23, 24, 25)))
    ):
        selected = torch.tensor([row * 30 + column for row in rows for column in columns])
        control_columns = (10, 11, 16, 17) if factor == 0 else (20, 21, 26, 27)
        control = torch.tensor([row * 30 + column for row in rows for column in control_columns])
        detail[:, selected, 0] = 1.0
        detail[:, control, 0] = 1.0
        detail[:, selected, 1] = 0.05
        detail[:, control, 1] = 0.05
        predicate_map[:, factor, selected] = 1.0
        predicate_map[:, 1 - factor, control] = 0.2
        contribution[:, 0, selected] = 1.0
        contribution[:, 0, control] = 0.02
        factor_regions.append((selected, control))
    return grid_hw, detail, predicate_map, contribution, factor_regions


def test_full_teacher_uses_same_decoder_for_matched_selected_and_control_sets() -> None:
    grid_hw, detail, predicate_map, contribution, _ = _matched_teacher_inputs()
    base_logits = torch.tensor([[0.10, -0.20]])
    labels = torch.tensor([[1.0, 0.0]])
    candidate = torch.tensor([[[0.80, 0.10, 0.10], [0.40, 0.50, 0.10]]])
    reliability = torch.ones(1, 2)
    overlap = torch.tensor([[[1.0, 0.5], [0.5, 1.0]]])
    calls = []

    def decoder(field, deleted_patches, *, sample_index, action_index, factor_index, variant):
        calls.append((variant, factor_index, deleted_patches.detach().clone(), field.data_ptr()))
        logits = base_logits[sample_index : sample_index + 1].clone()
        if variant == "selected":
            logits[:, action_index] = -0.40 - 0.10 * factor_index
        elif variant == "control":
            logits[:, action_index] = 0.20 + 0.10 * factor_index
        else:
            raise AssertionError("unexpected decoder variant")
        return {"action_logits": logits}

    plan = build_sparse_counterfactual_teacher(
        base_logits,
        labels,
        candidate,
        reliability,
        overlap,
        detail_field=detail,
        predicate_map=predicate_map,
        action_contribution=contribution,
        grid_hw=grid_hw,
        utility_logit=torch.randn(1, 2, 2),
        teacher_decoder=decoder,
    )

    assert len(calls) == 2 * plan["factor_indices"].numel()
    assert {call[0] for call in calls} == {"selected", "control"}
    assert len({call[3] for call in calls}) == 1
    for record in plan["records"]:
        assert not bool(torch.isin(record["selected_patches"], record["control_patches"]).any())
        assert record["control_metadata"]["selection_method"] == "deterministic_matched_control"
        assert record["control_metadata"]["same_sector"] is True
    expected = torch.sigmoid(
        (plan["control_margin"] - plan["selected_deletion_margin"]) / 0.10
    )
    torch.testing.assert_close(plan["utility_teacher_target"], expected)
    assert torch.equal(plan["sample_indices"], torch.zeros_like(plan["sample_indices"]))
    assert torch.equal(plan["action_indices"], torch.zeros_like(plan["action_indices"]))


def test_teacher_fails_closed_without_decoder_logits() -> None:
    grid_hw, detail, predicate_map, contribution, _ = _matched_teacher_inputs()
    args = (
        torch.tensor([[0.10, -0.20]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[[0.80, 0.10, 0.10], [0.40, 0.50, 0.10]]]),
        torch.ones(1, 2),
        torch.ones(1, 2, 2),
    )
    for decoder in (None, lambda *args, **kwargs: {}):
        try:
            build_sparse_counterfactual_teacher(
                *args,
                detail_field=detail,
                predicate_map=predicate_map,
                action_contribution=contribution,
                grid_hw=grid_hw,
                teacher_decoder=decoder,
            )
        except RuntimeError as error:
            assert "teacher" in str(error).lower() or "logits" in str(error).lower()
        else:
            raise AssertionError("teacher must fail closed without selected/control logits")
