import torch
import torch.nn.functional as F

from fate_oia.losses.save_action_losses import (
    SAVE_ACTION_LOSS_WEIGHTS,
    asymmetric_multilabel_elements,
    asymmetric_multilabel_loss,
    counterfactual_utility_loss,
    dense_utility_loss,
    easy_sample_nonregression_loss,
    save_action_loss,
    save_action_loss_per_sample,
    soft_f1_loss,
)
from fate_oia.models.save_action_evidence import SAVEActionEvidence


def test_save_progress_zero_keeps_base_output_but_trains_evidence_auxiliary() -> None:
    torch.manual_seed(13)
    module = SAVEActionEvidence(dim=8, action_dim=2, num_heads=2)
    action_nodes = torch.randn(2, 2, 8)
    global_field = torch.randn(2, 3600, 8)
    detail_field = torch.randn(2, 3600, 8)
    base_logits = torch.randn(2, 2)
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    output = module(
        action_nodes,
        global_field,
        detail_field,
        action_logits_base=base_logits,
        progress=0.0,
    )

    torch.testing.assert_close(output["action_logits_final"], base_logits, atol=0, rtol=0)
    assert not torch.equal(output["action_logits_evidence_aux"], base_logits)

    asymmetric_multilabel_loss(output["action_logits_evidence_aux"], target).backward()

    gradients = [
        parameter.grad
        for name, parameter in module.named_parameters()
        if "detail_" in name or "patch_value" in name
    ]
    assert gradients
    assert any(gradient is not None and torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_evidence_auxiliary_firewalls_foundation_but_trains_both_inquiries() -> None:
    torch.manual_seed(31)
    module = SAVEActionEvidence(dim=8, action_dim=2, num_heads=2)
    action_nodes = torch.randn(2, 2, 8, requires_grad=True)
    global_field = torch.randn(2, 3600, 8, requires_grad=True)
    detail_field = torch.randn(2, 3600, 8, requires_grad=True)
    base_logits = torch.randn(2, 2, requires_grad=True)
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    output = module(
        action_nodes,
        global_field,
        detail_field,
        base_logits,
        progress=0.0,
    )
    asymmetric_multilabel_loss(output["action_logits_evidence_aux"], target).backward()

    for foundation_input in (action_nodes, global_field, detail_field, base_logits):
        assert foundation_input.grad is None or torch.count_nonzero(foundation_input.grad) == 0
    for parameter in (
        module.global_inquiry.in_proj_weight,
        module.detail_query.weight,
        module.detail_key.weight,
        module.detail_value.weight,
        module.detail_output.weight,
        module.patch_action_value.weight,
        module.patch_value.weight,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0

    inquiry_nodes = action_nodes.detach().clone().requires_grad_(True)
    inquiry_global = global_field.detach().clone().requires_grad_(True)
    inquiry_detail = detail_field.detach().clone().requires_grad_(True)
    inquiry = module(
        inquiry_nodes,
        inquiry_global,
        inquiry_detail,
        base_logits.detach(),
        progress=1.0,
    )
    inquiry_loss = (
        inquiry["action_global_token"].square().mean()
        + inquiry["action_detail_token"].square().mean()
    )
    inquiry_gradients = torch.autograd.grad(
        inquiry_loss, (inquiry_nodes, inquiry_global, inquiry_detail)
    )
    assert all(gradient.abs().sum() > 0 for gradient in inquiry_gradients)


def _loss_output() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    output = {
        "action_logits_final": torch.tensor([[1.2, -0.8], [-0.7, 1.1]], requires_grad=True),
        "action_logits_base": torch.tensor([[1.5, -1.4], [-1.3, 1.6]], requires_grad=True),
        "action_logits_evidence_aux": torch.tensor([[0.9, -0.6], [-0.5, 0.8]], requires_grad=True),
        "utility_loss_cf": torch.tensor([0.2, 0.4], requires_grad=True),
        "utility_loss_dense": torch.tensor([0.3, 0.5], requires_grad=True),
        "action_sufficiency_loss": torch.tensor([0.6, 0.8], requires_grad=True),
        "action_necessity_loss": torch.tensor([0.7, 0.9], requires_grad=True),
        "action_control_loss": torch.tensor([0.1, 0.3], requires_grad=True),
        "action_preserve_loss": torch.tensor([0.4, 0.6], requires_grad=True),
    }
    return output, target


def test_save_action_loss_has_exact_12_weights_keys_and_total_formula() -> None:
    expected_weights = {
        "action_final": 1.00,
        "action_base": 0.35,
        "action_evidence_aux": 0.20,
        "action_utility_cf": 0.10,
        "action_utility_dense": 0.02,
        "action_sufficiency": 0.08,
        "action_necessity": 0.08,
        "action_control": 0.04,
        "action_preserve": 0.02,
        "action_soft_f1": 0.03,
        "action_cardinality": 0.02,
        "action_easy": 0.03,
    }
    assert SAVE_ACTION_LOSS_WEIGHTS == expected_weights

    output, target = _loss_output()
    losses = save_action_loss(output, target)
    assert set(losses) == {
        "final",
        "base",
        "evidence_aux",
        "utility_cf",
        "utility_dense",
        "sufficiency",
        "necessity",
        "control",
        "preserve",
        "soft_f1",
        "cardinality",
        "easy",
        "total",
    }
    expected_terms = {
        "final": asymmetric_multilabel_loss(output["action_logits_final"], target),
        "base": asymmetric_multilabel_loss(output["action_logits_base"], target),
        "evidence_aux": asymmetric_multilabel_loss(output["action_logits_evidence_aux"], target),
        "utility_cf": output["utility_loss_cf"].mean(),
        "utility_dense": output["utility_loss_dense"].mean(),
        "sufficiency": output["action_sufficiency_loss"].mean(),
        "necessity": output["action_necessity_loss"].mean(),
        "control": output["action_control_loss"].mean(),
        "preserve": output["action_preserve_loss"].mean(),
        "soft_f1": soft_f1_loss(output["action_logits_final"], target),
        "cardinality": F.smooth_l1_loss(
            torch.sigmoid(output["action_logits_final"].float()).sum(-1),
            target.sum(-1),
        ),
        "easy": easy_sample_nonregression_loss(
            output["action_logits_final"], output["action_logits_base"], target
        ),
    }
    for key, expected in expected_terms.items():
        torch.testing.assert_close(losses[key], expected, atol=1e-7, rtol=0)
    expected_total = sum(
        expected_weights[weight_key] * expected_terms[term_key]
        for term_key, weight_key in (
            ("final", "action_final"),
            ("base", "action_base"),
            ("evidence_aux", "action_evidence_aux"),
            ("utility_cf", "action_utility_cf"),
            ("utility_dense", "action_utility_dense"),
            ("sufficiency", "action_sufficiency"),
            ("necessity", "action_necessity"),
            ("control", "action_control"),
            ("preserve", "action_preserve"),
            ("soft_f1", "action_soft_f1"),
            ("cardinality", "action_cardinality"),
            ("easy", "action_easy"),
        )
    )
    torch.testing.assert_close(losses["total"], expected_total, atol=1e-7, rtol=0)


def test_save_action_loss_optional_terms_detach_teachers_and_absent_terms_are_zero() -> None:
    output, target = _loss_output()
    optional_names = (
        "utility_loss_cf",
        "utility_loss_dense",
        "action_sufficiency_loss",
        "action_necessity_loss",
        "action_control_loss",
        "action_preserve_loss",
    )
    absent = {key: value for key, value in output.items() if key not in optional_names}
    absent_losses = save_action_loss(absent, target)
    for key in ("utility_cf", "utility_dense", "sufficiency", "necessity", "control", "preserve"):
        assert absent_losses[key].item() == 0.0

    utility = torch.randn(2, 2, 3, requires_grad=True)
    contribution = torch.randn(2, 2, 3, requires_grad=True)
    dense_utility_loss(utility, contribution, target).backward()
    assert utility.grad is not None and utility.grad.abs().sum() > 0
    assert contribution.grad is None or torch.count_nonzero(contribution.grad) == 0

    prediction = torch.randn(4, requires_grad=True)
    teacher = torch.randn(4, requires_grad=True)
    counterfactual_utility_loss(prediction, teacher).backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0
    assert teacher.grad is None or torch.count_nonzero(teacher.grad) == 0

    final = torch.tensor([[2.0, -0.6]], requires_grad=True)
    base = torch.tensor([[2.0, -2.0]], requires_grad=True)
    easy_sample_nonregression_loss(final, base, torch.tensor([[1.0, 0.0]])).backward()
    assert final.grad is not None and final.grad.abs().sum() > 0
    assert base.grad is None or torch.count_nonzero(base.grad) == 0


def test_save_action_loss_per_sample_is_exact_final_base_aux_formula() -> None:
    output, target = _loss_output()
    actual = save_action_loss_per_sample(output, target)
    expected = (
        1.00 * asymmetric_multilabel_elements(output["action_logits_final"], target).mean(-1)
        + 0.35 * asymmetric_multilabel_elements(output["action_logits_base"], target).mean(-1)
        + 0.20
        * asymmetric_multilabel_elements(output["action_logits_evidence_aux"], target).mean(-1)
    )
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)
