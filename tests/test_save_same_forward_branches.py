from __future__ import annotations

import torch

from fate_oia.models.meter_schema import METERFactorSchema
from fate_oia.models.save_oia_model import SAVEOIAModel


def test_save_decode_exposes_required_branches_from_one_field() -> None:
    model = SAVEOIAModel(use_mock_dino=True)
    field = model.encode_images(torch.randn(1, 3, 360, 640))
    output = model.decode_from_field(field, progress=0.25)

    required = {
        "action_logits_base",
        "action_logits_final",
        "action_logits_evidence_aux",
        "reason_logits_calalign",
        "reason_logits_clean",
        "reason_logits_benchmark",
        "reason_logits_private_direct",
        "reason_logits_pu_private",
        "action_global_attention",
        "action_detail_attention",
        "action_patch_contribution",
        "action_named_contribution",
        "action_unnamed_contribution",
        "predicate_map",
        "predicate_reliability",
        "utility_logit",
        "utility_prob",
        "branch_logits",
        "audit_outputs",
    }
    assert required.issubset(output)
    assert {"base", "final", "evidence_aux"}.issubset(output["branch_logits"])
    assert output["audit_outputs"]["field_reused"] is True
    assert output["utility_teacher_plan"] is None
    assert output["action_named_contribution"].shape == (
        *output["action_evidence_raw"].shape,
        model.reason_dim,
    )
    assert output["action_unnamed_contribution"].shape == output["action_evidence_raw"].shape[:2]
    reconstructed = (
        output["action_named_contribution"].float().sum(-1)
        + output["action_unnamed_contribution"].float()
    )
    assert torch.equal(reconstructed, output["action_evidence_raw"].float())
    assert torch.equal(
        output["action_responsibility_sum"].float(),
        torch.ones_like(output["action_responsibility_sum"].float()),
    )
    assert torch.equal(
        output["action_conservation_error"].float(),
        torch.zeros_like(output["action_conservation_error"].float()),
    )


def test_save_evidence_auxiliary_gradient_is_firewalled_from_foundation() -> None:
    model = SAVEOIAModel(use_mock_dino=True)
    output = model.decode_from_field(
        model.encode_images(torch.randn(1, 3, 360, 640)), progress=0.5
    )

    output["action_logits_evidence_aux"].sum().backward()

    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.action_evidence.parameters()
    )
    assert all(parameter.grad is None for parameter in model.foundation.dino.parameters())


def test_save_action_loss_activates_clean_reason_to_action_and_private_is_firewalled() -> None:
    model = SAVEOIAModel(use_mock_dino=True)
    output = model.decode_from_field(
        model.encode_images(torch.randn(1, 3, 360, 640)), progress=1.0
    )

    output["action_logits_final"].square().mean().backward()

    bridge = model.reason_decoder.clean_reason.reason_to_action
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in bridge.parameters()
    )
    assert all(parameter.grad is None for parameter in model.reason_decoder.private_reason.parameters())

    model.zero_grad(set_to_none=True)
    output = model.decode_from_field(
        model.encode_images(torch.randn(1, 3, 360, 640)), progress=1.0
    )
    output["reason_logits_pu_private"].square().mean().backward()

    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.reason_decoder.private_reason.parameters()
    )
    assert all(parameter.grad is None for parameter in model.reason_decoder.clean_reason.parameters())
    assert all(parameter.grad is None for parameter in model.foundation.parameters())


def test_save_clean_reason_action_path_is_bounded_and_zero_at_progress_zero() -> None:
    model = SAVEOIAModel(use_mock_dino=True)
    field = model.encode_images(torch.randn(1, 3, 360, 640))
    zero = model.decode_from_field(field, progress=0.0)
    full = model.decode_from_field(field, progress=1.0)

    torch.testing.assert_close(
        zero["action_logits_final"],
        zero["action_logits_base"],
        atol=0.0,
        rtol=0.0,
    )
    assert float(full["action_clean_reason_ramp"]) == 1.0
    assert torch.all(
        full["action_clean_reason_delta"].abs()
        <= full["action_clean_reason_kappa"].view(1, -1) + 1e-7
    )


def test_save_training_forward_runs_due_teacher_on_the_same_encoded_field() -> None:
    model = SAVEOIAModel(use_mock_dino=True)
    model.train()
    output = model(
        torch.randn(1, 3, 360, 640),
        progress=0.0,
        action_targets=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        optimizer_update=4,
        run_teacher=True,
    )

    plan = output["utility_teacher_plan"]
    assert output["utility_teacher_due"] is True
    assert plan is not None
    assert plan["factor_indices"].numel() > 0
    assert plan["selected_control_calls"] == 2 * plan["factor_indices"].numel()
    assert output["utility_teacher_prediction"].shape == output["utility_teacher_target"].shape
    assert all(
        record["selected_patches"].numel() == record["control_patches"].numel()
        and not bool(torch.isin(record["selected_patches"], record["control_patches"]).any())
        for record in plan["records"]
    )
    assert model.encode_call_count == 1
    assert model.foundation.ordinary_dino_calls == 1


def test_save_due_teacher_faithfulness_trains_utility_without_foundation_gradient() -> None:
    model = SAVEOIAModel(use_mock_dino=True)
    model.train()
    output = model(
        torch.randn(1, 3, 360, 640),
        progress=0.0,
        action_targets=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        optimizer_update=4,
        run_teacher=True,
    )

    output["losses"]["loss_utility_cf"].backward()

    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.utility_bridge.utility_predictor.parameters()
    )
    assert all(parameter.grad is None for parameter in model.foundation.parameters())


def test_save_predicate_measurement_loads_save_schema_metadata_by_default() -> None:
    model = SAVEOIAModel(use_mock_dino=True)
    schema_path = model.predicate_measurement.schema_path
    schema = METERFactorSchema(schema_path)

    assert schema_path.name == "save_factor_schema.yaml"
    assert model.predicate_measurement.schema_sha256 == schema.sha256
    assert model.predicate_measurement.typed_head.schema_sha256 == schema.sha256
    assert model.predicate_measurement.predicate_mirror_pairs == schema.mirror_pairs
