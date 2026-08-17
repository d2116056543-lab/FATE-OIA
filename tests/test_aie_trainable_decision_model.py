from pathlib import Path

import torch
from torch import nn

from fate_oia.models.aie_trainable_decision_model import AIETrainableDecisionModel
from fate_oia.engine.train_aie_trainable_decision import (
    contradiction_weighted_reason_calibration_loss,
    decision_loss,
    global_soft_f1_linearized_loss,
    global_soft_f1_stats,
    positive_boundary_hinge_loss,
    reason_label_pair_ranking_loss,
    reason_tail_mask,
    soft_macro_f1_loss,
)


class _DifferentiableFakeAIE(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_weight = nn.Parameter(torch.ones(1))

    def encode_images(self, images):
        return {"images": images}

    def decode_from_field(self, field, *, action_scale, reason_scale, reason_action_scale):
        batch = field["images"].shape[0]
        scale = torch.as_tensor(action_scale, device=field["images"].device).view(1, 4)
        primary = self.base_weight * torch.tensor([[0.2, -0.1, 0.3, -0.2]], device=scale.device)
        action = primary.expand(batch, -1) + scale
        reason = self.base_weight * torch.ones(batch, 21, device=scale.device) * reason_scale
        return {
            "action_logits_primary": primary.expand(batch, -1),
            "action_logits_final": action,
            "reason_logits_final": reason,
            "named_coverage": torch.tensor(0.0, device=scale.device),
            "action_delta": action - primary,
        }


def test_trainable_decision_model_updates_only_scales_and_boundaries():
    base = _DifferentiableFakeAIE()
    model = AIETrainableDecisionModel(base, reason_scale=0.6, reason_action_scale=0.0)
    output = model(torch.randn(3, 3, 8, 8))

    loss = output["action_logits_decision"].square().mean() + output["reason_logits_decision"].square().mean()
    loss.backward()

    assert model.action_scale_raw.grad is not None
    assert float(model.action_scale_raw.grad.abs().sum()) > 0
    assert model.threshold_raw.grad is not None
    assert float(model.threshold_raw.grad.abs().sum()) > 0
    assert base.base_weight.requires_grad is False
    assert base.base_weight.grad is None
    assert output["action_scales"].shape == (4,)
    assert bool(((output["action_scales"] > 0) & (output["action_scales"] < 1)).all())
    assert output["threshold_prob"].shape == (25,)


def test_trainable_decision_model_parameters_round_trip_in_state_dict():
    model = AIETrainableDecisionModel(_DifferentiableFakeAIE(), reason_scale=0.6)
    state = model.state_dict()
    assert "action_scale_raw" in state
    assert "threshold_raw" in state
    assert "reason_scale_raw" in state
    assert "base_model.base_weight" in state


def test_trainable_reason_scales_receive_gradient_without_changing_action_branch():
    model = AIETrainableDecisionModel(_DifferentiableFakeAIE(), reason_scale=0.6)
    images = torch.randn(3, 3, 8, 8)
    before = model(images)
    with torch.no_grad():
        model.reason_scale_raw.add_(0.5)
    after = model(images)
    torch.testing.assert_close(
        before["action_logits_decision"], after["action_logits_decision"], atol=0.0, rtol=0.0
    )
    assert not torch.equal(before["reason_logits_decision"], after["reason_logits_decision"])
    loss = after["reason_logits_decision"].square().mean()
    loss.backward()
    assert model.reason_scale_raw.grad is not None
    assert float(model.reason_scale_raw.grad.abs().sum()) > 0.0
    assert bool(((after["reason_scales"] >= 0.0) & (after["reason_scales"] <= 1.0)).all())


def test_sparse_reason_objective_recovers_positives_instead_of_raising_every_boundary():
    action_logits = torch.zeros(8, 4, requires_grad=True)
    reason_logits = torch.zeros(8, 21, requires_grad=True)
    action_target = torch.zeros(8, 4)
    action_target[:4] = 1.0
    reason_target = torch.zeros(8, 21)
    reason_target[0] = 1.0
    losses = decision_loss(
        {
            "action_logits_decision": action_logits,
            "reason_logits_decision": reason_logits,
            "action_delta": torch.zeros(8, 4),
            "action_logits_primary": torch.ones(8, 4),
        },
        action_target,
        reason_target,
    )
    losses["total"].backward()
    assert torch.isfinite(reason_logits.grad).all()
    # Negative logit gradient means optimization raises sparse positive scores;
    # because deploy=base-threshold, the corresponding threshold moves down.
    assert float(reason_logits.grad.mean()) < 0.0


def test_global_soft_f1_linearization_matches_full_batch_gradient():
    torch.manual_seed(7)
    logits = torch.randn(11, 5, requires_grad=True)
    target = (torch.rand(11, 5) > 0.72).float()
    target[0] = 1.0
    exact = soft_macro_f1_loss(logits, target, temperature=0.3)
    exact_grad = torch.autograd.grad(exact, logits)[0]

    split_logits = logits.detach().clone().requires_grad_(True)
    stats = global_soft_f1_stats(split_logits.detach(), target, temperature=0.3)
    linearized = sum(
        global_soft_f1_linearized_loss(split_logits[start : start + 4], target[start : start + 4], stats)
        for start in range(0, len(target), 4)
    )
    linearized_grad = torch.autograd.grad(linearized, split_logits)[0]
    torch.testing.assert_close(linearized_grad, exact_grad, atol=1e-7, rtol=1e-6)


def test_positive_boundary_hinge_equalizes_tail_label_gradient_mass():
    logits = torch.zeros(4, 2, requires_grad=True)
    target = torch.tensor([[1.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    counts = target.sum(0)
    loss = positive_boundary_hinge_loss(logits, target, counts, margin=0.2)
    loss.backward()
    per_label_gradient_mass = (logits.grad * target).abs().sum(0)
    torch.testing.assert_close(
        per_label_gradient_mass[0], per_label_gradient_mass[1], atol=1e-7, rtol=1e-6
    )


def test_reason_tail_mask_uses_train_positive_counts_only():
    counts = torch.tensor([0.0, 5.0, 20.0, 21.0, 100.0])
    assert torch.equal(
        reason_tail_mask(counts, 20),
        torch.tensor([False, True, True, False, False]),
    )


def test_fit_all_mode_uses_fixed_cv_epoch_without_train_audit_selection():
    source = Path("fate_oia/engine/train_aie_trainable_decision.py").read_text(encoding="utf-8")
    assert '"--fit-all-decision-samples"' in source
    assert "fit-all-decision-samples and cv-fold-index are mutually exclusive" in source
    assert '"checkpoint_selection_split": "fixed_cv_epoch"' in source
    assert '"selection_source": "five_fold_train_only_curve"' in source


def test_training_entrypoint_can_freeze_thresholds_for_scale_ablation():
    source = Path("fate_oia/engine/train_aie_trainable_decision.py").read_text(encoding="utf-8")
    assert '"--freeze-thresholds"' in source
    assert '"freeze_thresholds": args.freeze_thresholds' in source


def test_contradiction_weighted_calibration_ignores_unknown_reason_zero():
    logits = torch.zeros(2, 3, requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    contradiction = torch.tensor([[0.0, 0.9, 0.1], [0.0, 0.0, 0.0]])
    loss = contradiction_weighted_reason_calibration_loss(
        logits, target, contradiction, reliable_negative_min=0.6
    )
    loss.backward()

    assert float(logits.grad[0, 0]) < 0.0
    assert float(logits.grad[0, 1]) > 0.0
    assert float(logits.grad[0, 2]) == 0.0
    assert bool((logits.grad[1] == 0.0).all())


def test_reason_label_pair_ranking_uses_only_positive_vs_reliable_negative():
    logits = torch.tensor([[0.0, 0.4, -0.2]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0]])
    contradiction = torch.tensor([[0.0, 0.9, 0.2]])
    loss = reason_label_pair_ranking_loss(
        logits, target, contradiction, reliable_negative_min=0.6, margin=0.2
    )
    loss.backward()

    assert float(logits.grad[0, 0]) < 0.0
    assert float(logits.grad[0, 1]) > 0.0
    assert float(logits.grad[0, 2]) == 0.0
