import torch

from fate_oia.losses.tfc_losses import compute_tfc_losses
from fate_oia.models.acpr_tfc_model import ACPRTFCModel
from fate_oia.models.tfc_deletion_contrast import TFCDeletionContrast
from fate_oia.models.tfc_pu_state import TFCPUStateBuilder
from fate_oia.models.tfc_target_credit import TFCTargetCredit
from fate_oia.models.tfc_topk_factor_measurement import TFCTopKFactorMeasurement


def test_tfc_model_forward_shapes_and_firewall():
    model = ACPRTFCModel(use_mock_dino=True, factor_topk_tokens=8)
    images = torch.randn(2, 3, 360, 640)
    action = torch.zeros(2, 4); reason = torch.zeros(2, 21); reason[:, 0] = 1
    out = model(images, action, reason, epoch=7, split="train", run_deletion=True)
    for key in [
        "action_visual_logits", "action_tfc_delta", "action_logits_base", "action_logits_deploy",
        "reason_visual_logits", "reason_tfc_delta", "reason_logits_base", "reason_logits_deploy",
        "factor_probs_action", "factor_rho_action", "factor_probs_reason", "factor_rho_reason",
        "credit_action", "credit_reason", "credit_confidence_action", "credit_confidence_reason",
        "action_theta", "reason_theta", "theta_delta_action", "theta_delta_reason", "pu_state",
        "deletion_stats", "deletion_stats_action", "deletion_stats_reason", "artifact_stats", "factor_features_action", "factor_features_reason",
        "factor_prototypes", "factor_queries", "native_similarity", "factor_conflict", "compatibility",
    ]:
        assert key in out
    assert out["action_logits_deploy"].shape == (2, 4)
    assert out["reason_logits_deploy"].shape == (2, 21)
    assert torch.isfinite(out["action_logits_deploy"]).all()
    assert torch.allclose(out["action_logits_deploy"], out["action_logits_base"] - out["action_theta"], atol=1e-6)
    out_no_del = model(images, action, reason, epoch=7, split="train", run_deletion=False)
    assert torch.allclose(out_no_del["action_tfc_delta"], torch.zeros_like(out_no_del["action_tfc_delta"]))
    assert torch.allclose(out_no_del["reason_tfc_delta"], torch.zeros_like(out_no_del["reason_tfc_delta"]))
    losses = compute_tfc_losses(out, action, reason, {"prototype_consistency": 0.05, "rate_cardinality": 0.05})
    assert "prototype" in losses
    assert "cardinality" in losses
    assert torch.isfinite(losses["total"])


def test_deletion_replacement_uses_same_region_background():
    deletion = TFCDeletionContrast(ema_momentum=0.0)
    patch = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1)
    patched = deletion._replace(patch, torch.tensor([[0, 1]]), torch.tensor([[4, 5]]))
    assert torch.allclose(patched[0, 0, 0, 0], torch.tensor(4.5))
    assert torch.allclose(patched[0, 0, 1, 0], torch.tensor(4.5))
    assert torch.allclose(deletion.ema_background.view(-1), torch.tensor([4.5]))


def test_deletion_max_factors_schedule_limits_early_epochs():
    model = ACPRTFCModel(use_mock_dino=True, factor_topk_tokens=8, max_deletion_factors_per_sample=4)
    assert model.deletion_max_factors_for_epoch(0) == 2
    assert model.deletion_max_factors_for_epoch(5) == 2
    assert model.deletion_max_factors_for_epoch(6) == 4


def test_random_deletion_indices_are_equal_area_unique():
    measurement = TFCTopKFactorMeasurement(dim=4, topk=4, grid_hw=(2, 3))
    patches = torch.randn(2, 1, 6, 4)
    queries = torch.randn(3, 4)
    out = measurement(patches, queries, ["front_center", "left_corridor", "right_corridor"])
    random_indices = out["random_indices"]
    assert random_indices.shape == (2, 3, 4)
    for sample in range(random_indices.shape[0]):
        for factor in range(random_indices.shape[1]):
            assert random_indices[sample, factor].unique().numel() == random_indices.shape[-1]


def test_tfc_target_credit_cannot_create_unknown_native_credit():
    module = TFCTargetCredit(num_factors=2, action_dim=2, reason_dim=3, dim=4)
    factor_probs = torch.ones(1, 2)
    factor_rho = torch.ones(1, 2)
    factor_features = torch.randn(1, 2, 4)
    compatibility = {
        "factor_to_action_support": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        "factor_to_action_inhibit": torch.zeros(2, 2),
        "factor_to_reason_support": torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        "factor_to_reason_inhibit": torch.zeros(2, 3),
    }
    out = module(factor_probs, factor_rho, factor_features, compatibility)
    assert out["credit_action"][0, 0, 0].abs() > 0
    assert torch.allclose(out["credit_action"][0, :, 1], torch.zeros(2), atol=1e-7)
    assert torch.allclose(out["credit_action"][0, 1], torch.zeros(2), atol=1e-7)
    assert out["credit_reason"][0, 0, 1].abs() > 0
    assert torch.allclose(out["credit_reason"][0, :, 0], torch.zeros(2), atol=1e-7)


def test_tfc_pu_hard_negative_requires_deletion_gate():
    builder = TFCPUStateBuilder(max_hard_negative_rate=1.0)
    reason_targets = torch.zeros(1, 3)
    credit_reason = torch.tensor([[[-0.5, -0.6, -0.7]]])
    factor_probs = torch.ones(1, 1)
    factor_rho = torch.ones(1, 1)
    no_gate = builder(reason_targets, credit_reason, factor_probs, factor_rho, epoch=7)
    assert no_gate["hard_negative_mask"].sum() == 0
    with_gate = builder(
        reason_targets,
        credit_reason,
        factor_probs,
        factor_rho,
        epoch=7,
        deletion_gate_reason=torch.ones(1, 3, dtype=torch.bool),
    )
    assert with_gate["hard_negative_mask"].sum() > 0
