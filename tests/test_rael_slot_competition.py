"""P5 behavioral and boundary contracts for the competitive RAEL ledger."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from fate_oia.models.rael_multilayer_field import RAELMultiLayerField


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "configs" / "rael_reason_semantics.yaml"


def _module():
    spec = importlib.util.find_spec("fate_oia.models.rael_slot_ledger")
    assert spec is not None, "P5 requires fate_oia.models.rael_slot_ledger"
    return importlib.import_module("fate_oia.models.rael_slot_ledger")


@pytest.fixture()
def ledger():
    torch.manual_seed(3)
    return _module().RAELSlotLedger(dim=8, num_layers=4)


@pytest.fixture()
def prepared_field():
    torch.manual_seed(17)
    field = RAELMultiLayerField(dim=8, num_layers=4, formal_grid_hw=(3, 4))
    field.eval()
    with torch.no_grad():
        return field.precompute(
            torch.randn(2, 4, 12, 8),
            torch.randn(2, 4, 8),
            grid_hw=(3, 4),
        )


def _audit(ledger, output):
    return ledger.audit_diagnostics(output["public_evidence"])


def _storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def test_p5_module_imports_after_green() -> None:
    assert _module().RAELSlotLedger is not None


def test_internal_slot_schema_has_fixed_road_and_separate_background(ledger) -> None:
    module = _module()
    specs = ledger.slot_specs
    assert len(specs) == 21
    assert sum(spec.family == "entity" for spec in specs) == 12
    assert tuple(spec.name for spec in specs[12:17]) == module.RAELSlotLedger.ROAD_SLOT_NAMES
    assert all(spec.fixed_identity for spec in specs[12:17])
    assert sum(spec.family == "latent" for spec in specs) == 3
    assert specs[20].name == "background_sink"
    assert specs[20].public is False
    assert ledger.formal_metadata()["public_slot_count"] == 20


def test_competition_normalizes_over_slots_and_wrong_axis_mutation_fails(ledger) -> None:
    logits = torch.zeros(1, 21, 3)
    logits[:, 0, 0] = 4.0
    logits[:, 1, 1] = 3.0
    assignment = ledger._slot_competition(logits)
    assert torch.allclose(assignment.sum(dim=1), torch.ones(1, 3), atol=1e-6)
    wrong_axis = torch.softmax(logits, dim=-1)
    with pytest.raises(AssertionError):
        assert torch.allclose(wrong_axis.sum(dim=1), torch.ones(1, 3), atol=1e-6)


def test_two_iterations_have_mask_bias_and_one_or_three_round_mutations_fail(ledger, prepared_field) -> None:
    with patch.object(ledger.slot_gru, "forward", wraps=ledger.slot_gru.forward) as forward:
        output = ledger(prepared_field)
    diagnostics = _audit(ledger, output)
    assert output["iterations"] == 2
    assert forward.call_count == 2
    assert torch.allclose(
        diagnostics.iteration1_assignment,
        torch.softmax(diagnostics.iteration1_visual_logits, dim=1),
        atol=1e-6,
    )
    expected_two = diagnostics.iteration2_visual_logits + 0.5 * torch.log(
        diagnostics.iteration1_assignment.clamp_min(ledger.eps)
    )
    assert torch.allclose(diagnostics.iteration2_logits, expected_two, atol=1e-6)
    with pytest.raises(AssertionError):
        assert forward.call_count == 1
    with pytest.raises(AssertionError):
        assert forward.call_count == 3
    with pytest.raises(AssertionError):
        assert torch.allclose(diagnostics.iteration2_logits, diagnostics.iteration2_visual_logits)


def test_public_default_api_hides_internal21_and_background_is_separate(ledger, prepared_field) -> None:
    output = ledger(prepared_field)
    view = output["public_evidence"]
    diagnostics = _audit(ledger, output)
    assert all(not key.startswith("internal") for key in output)
    assert "diagnostics" not in output
    assert isinstance(view, _module().PublicEvidenceView)
    assert output["slot_tokens"].shape == (2, 20, 8)
    assert output["slot_masks"].shape == (2, 20, 3, 4)
    assert output["slot_activity"].shape == (2, 20)
    assert output["slot_area"].shape == (2, 20)
    assert output["slot_centroid"].shape == (2, 20, 2)
    assert output["slot_scale"].shape == (2, 20)
    assert torch.isfinite(output["slot_masks"]).all()
    assert bool((output["slot_masks"] >= 0.0).all())
    assert bool((output["slot_masks"] <= 1.0).all())
    assert bool((output["slot_activity"] >= 0.0).all())
    assert bool((output["slot_area"] >= 0.0).all())
    assert torch.allclose(output["slot_tokens"], diagnostics.slot_tokens[:, :20])
    assert torch.allclose(output["slot_masks"], diagnostics.slot_masks[:, :20])
    assert output["background_token"].shape == (2, 8)
    assert output["background_mask"].shape == (2, 1, 3, 4)
    assert torch.allclose(output["background_mask"], diagnostics.slot_masks[:, 20:21])
    assert diagnostics.allow_contribution is False
    assert diagnostics.allow_cf is False
    assert output["background_contract"] == {
        "allow_contribution": False,
        "allow_cf": False,
        "allow_explanation": False,
    }


def test_internal_diagnostics_are_audit_only_and_reject_nonissued_inputs(ledger, prepared_field) -> None:
    output = ledger(prepared_field)
    diagnostics = _audit(ledger, output)
    assert isinstance(diagnostics, _module().InternalLedgerDiagnostics)
    with pytest.raises(TypeError, match="PublicEvidenceView"):
        ledger.audit_diagnostics(output)
    with pytest.raises(TypeError, match="only available"):
        _module().InternalLedgerDiagnostics(_issuer=object(), provenance=object())


def test_p4_and_contribution_adapters_accept_only_issued_public_views(ledger, prepared_field) -> None:
    output = ledger(prepared_field)
    view = output["public_evidence"]
    bundle = ledger.to_evidence_read_bundle(view)
    contribution = ledger.public_contribution_view(view)
    assert bundle.tokens.shape == (2, 20, 8)
    assert torch.allclose(bundle.tokens, output["slot_tokens"])
    assert contribution is view
    assert contribution.masks.shape == (2, 20, 3, 4)
    with pytest.raises(TypeError, match="PublicEvidenceView"):
        ledger.to_evidence_read_bundle(output)
    with pytest.raises(TypeError, match="PublicEvidenceView"):
        ledger.public_contribution_view(_audit(ledger, output))


def test_forged_or_background_replaced_public_view_is_rejected(ledger, prepared_field) -> None:
    output = ledger(prepared_field)
    with pytest.raises(TypeError, match="only be issued"):
        _module().PublicEvidenceView(
            _issuer=object(),
            tokens=output["slot_tokens"],
            masks=output["slot_masks"],
            valid_mask=output["public_slot_valid"],
            slot_indices=tuple(range(20)),
            provenance=object(),
        )
    fake = object.__new__(_module().PublicEvidenceView)
    with pytest.raises(ValueError, match="not issued"):
        ledger.public_contribution_view(fake)

    altered_output = ledger(prepared_field)
    altered_view = altered_output["public_evidence"]
    background = altered_output["background_token"].unsqueeze(1).expand(-1, 20, -1)
    object.__setattr__(altered_view, "_PublicEvidenceView__tokens", background)
    with pytest.raises(ValueError, match="provenance"):
        ledger.to_evidence_read_bundle(altered_view)


def test_p5_public_view_final_isolation_rejects_replacement_and_inplace_mutation(ledger, prepared_field) -> None:
    output = ledger(prepared_field)
    view = output["public_evidence"]
    diagnostics = _audit(ledger, output)

    assert not hasattr(view, "__dict__")
    with pytest.raises(AttributeError, match="sealed"):
        view._tokens = output["background_token"]
    with pytest.raises(AttributeError, match="sealed"):
        view.tokens = output["slot_tokens"]
    assert view.tokens.requires_grad
    assert view.tokens.data_ptr() != output["slot_tokens"].data_ptr()
    assert _storage_ptr(view.tokens) != _storage_ptr(output["slot_tokens"])
    assert _storage_ptr(view.tokens) != _storage_ptr(diagnostics.slot_tokens)
    assert _storage_ptr(view.masks) != _storage_ptr(output["slot_masks"])
    assert _storage_ptr(view.masks) != _storage_ptr(diagnostics.slot_masks)
    assert _storage_ptr(view.tokens) != _storage_ptr(output["background_token"])

    ledger.zero_grad(set_to_none=True)
    view.tokens.square().mean().backward()
    assert ledger.slot_queries.grad is not None
    assert float(ledger.slot_queries.grad.abs().sum()) > 0.0

    with torch.no_grad():
        view.tokens.add_(0.01)
    for consumer in (
        ledger.public_contribution_view,
        ledger.to_evidence_read_bundle,
        ledger.latent_training_view,
        ledger.audit_diagnostics,
    ):
        with pytest.raises(ValueError, match="version"):
            consumer(view)

    for tensor_name, mutation in (
        ("masks", lambda tensor: tensor.mul_(0.9)),
        ("valid_mask", lambda tensor: tensor.logical_not_()),
    ):
        fresh = ledger(prepared_field)["public_evidence"]
        with torch.no_grad():
            mutation(getattr(fresh, tensor_name))
        with pytest.raises(ValueError, match="version"):
            ledger.to_evidence_read_bundle(fresh)


def test_p5_integrity_snapshots_reject_data_copy_background_bypass(ledger, prepared_field) -> None:
    token_view = ledger(prepared_field)["public_evidence"]
    token_background = ledger(prepared_field)["background_token"].unsqueeze(1).expand(-1, 20, -1)
    token_view.tokens.data.copy_(token_background)
    with pytest.raises(ValueError, match="integrity"):
        ledger.to_evidence_read_bundle(token_view)

    mask_output = ledger(prepared_field)
    mask_view = mask_output["public_evidence"]
    mask_background = mask_output["background_mask"].expand(-1, 20, -1, -1)
    mask_view.masks.data.copy_(mask_background)
    with pytest.raises(ValueError, match="integrity"):
        ledger.to_evidence_read_bundle(mask_view)

    valid_view = ledger(prepared_field)["public_evidence"]
    valid_view.valid_mask.data.copy_(torch.zeros_like(valid_view.valid_mask))
    with pytest.raises(ValueError, match="integrity"):
        ledger.to_evidence_read_bundle(valid_view)

    private_names = (
        "_PublicEvidenceView__token_integrity_snapshot",
        "_PublicEvidenceView__mask_integrity_snapshot",
        "_PublicEvidenceView__valid_mask_integrity_snapshot",
    )
    for private_name, public_tensor in zip(
        private_names,
        (valid_view.tokens, valid_view.masks, valid_view.valid_mask),
    ):
        snapshot = object.__getattribute__(valid_view, private_name)
        assert snapshot.requires_grad is False
        assert snapshot.data_ptr() != public_tensor.data_ptr()
        assert _storage_ptr(snapshot) != _storage_ptr(public_tensor)
    assert not hasattr(valid_view, "token_integrity_snapshot")


def test_p5_diagnostic_snapshots_are_detached_nonaliased_and_mutable_without_public_effect(ledger, prepared_field) -> None:
    output = ledger(prepared_field)
    view = output["public_evidence"]
    diagnostics = _audit(ledger, output)
    assert diagnostics.slot_tokens.requires_grad is False
    assert diagnostics.slot_masks.requires_grad is False
    assert diagnostics.slot_tokens.data_ptr() != view.tokens.data_ptr()
    assert diagnostics.slot_masks.data_ptr() != view.masks.data_ptr()
    assert _storage_ptr(diagnostics.slot_tokens) != _storage_ptr(view.tokens)
    assert _storage_ptr(diagnostics.slot_tokens) != _storage_ptr(output["slot_tokens"])
    assert _storage_ptr(diagnostics.slot_masks) != _storage_ptr(view.masks)
    assert _storage_ptr(diagnostics.slot_masks) != _storage_ptr(output["slot_masks"])
    assert _storage_ptr(diagnostics.slot_tokens) != _storage_ptr(output["background_token"])

    public_tokens_before = output["slot_tokens"].detach().clone()
    public_masks_before = output["slot_masks"].detach().clone()
    view_tokens_before = view.tokens.detach().clone()
    view_masks_before = view.masks.detach().clone()
    with torch.no_grad():
        diagnostics.slot_tokens.add_(1.0)
        diagnostics.slot_masks.mul_(0.0)
    assert torch.allclose(output["slot_tokens"], public_tokens_before)
    assert torch.allclose(output["slot_masks"], public_masks_before)
    assert torch.allclose(view.tokens, view_tokens_before)
    assert torch.allclose(view.masks, view_masks_before)
    assert ledger.to_evidence_read_bundle(view).tokens.shape == (2, 20, 8)

    with torch.no_grad():
        output["slot_tokens"].add_(2.0)
        output["slot_masks"].add_(0.25)
    assert torch.allclose(view.tokens, view_tokens_before)
    assert torch.allclose(view.masks, view_masks_before)
    assert not torch.allclose(diagnostics.slot_tokens[:, :20], output["slot_tokens"])
    assert not torch.allclose(diagnostics.slot_masks[:, :20], output["slot_masks"])


def test_latent_task_view_diversity_paths_are_public_and_have_gradients(ledger, prepared_field) -> None:
    output = ledger(prepared_field)
    latent = ledger.latent_training_view(output["public_evidence"])
    assert set(latent) == {"task_tokens", "view_masks", "diversity_activity"}
    assert latent["task_tokens"].shape == (2, 3, 8)
    assert latent["view_masks"].shape == (2, 3, 3, 4)
    assert latent["diversity_activity"].shape == (2, 3)
    for name in ("task_tokens", "view_masks", "diversity_activity"):
        ledger.zero_grad(set_to_none=True)
        current = ledger(prepared_field)
        current_latent = ledger.latent_training_view(current["public_evidence"])
        current_latent[name].square().mean().backward()
        assert ledger.slot_queries.grad is not None, name
        assert float(ledger.slot_queries.grad.abs().sum()) > 0.0, name


def test_ledger_reads_p3_shared_kv_and_remains_field_dependent(ledger, prepared_field) -> None:
    output_a = ledger(prepared_field)
    perturbed = dict(prepared_field)
    perturbed["keys_by_layer"] = prepared_field["keys_by_layer"].clone()
    perturbed["values_by_layer"] = prepared_field["values_by_layer"].clone()
    perturbed["keys_by_layer"][:, 0, 0, :] += 2.0
    perturbed["values_by_layer"][:, 0, 0, :] -= 1.0
    output_b = ledger(perturbed)
    assert not torch.allclose(output_a["slot_tokens"], output_b["slot_tokens"])
    source = inspect.getsource(ledger._visual_logits)
    assert "for layer_index in range(self.num_layers)" in source
    assert "torch.stack" not in source


def test_mirror_geometry_audit_interface_requires_issued_views_and_applies_road_swap(ledger, prepared_field) -> None:
    canonical = ledger(prepared_field)
    mirrored = ledger(prepared_field)
    canonical_view = canonical["public_evidence"]
    mirrored_view = mirrored["public_evidence"]
    canonical_diagnostics = _audit(ledger, canonical)
    mirrored_diagnostics = _audit(ledger, mirrored)
    object.__setattr__(
        mirrored_diagnostics,
        "_InternalLedgerDiagnostics__slot_masks",
        canonical_diagnostics.slot_masks.index_select(1, ledger.mirror_slot_permutation).flip(dims=(-1,)),
    )
    consistency = ledger.mirror_geometry_consistency(canonical_view, mirrored_view)
    assert consistency["finite"].all()
    assert float(consistency["mask_l1"].max()) == pytest.approx(0.0, abs=1e-7)
    with pytest.raises(TypeError, match="PublicEvidenceView"):
        ledger.mirror_geometry_consistency(canonical, mirrored_view)


def test_missing_road_permutation_mutation_has_nonzero_mirror_error(ledger) -> None:
    prepared = {
        "keys_by_layer": torch.zeros(1, 4, 8, 8),
        "values_by_layer": torch.zeros(1, 4, 8, 8),
        "layer_global_tokens": torch.zeros(1, 4, 8),
        "grid_hw": (2, 4),
    }
    first, second = ledger(prepared), ledger(prepared)
    crafted = torch.zeros_like(_audit(ledger, first).slot_masks)
    crafted[:, 12, :, 0] = 1.0
    crafted[:, 14, :, 3] = 1.0
    object.__setattr__(_audit(ledger, first), "_InternalLedgerDiagnostics__slot_masks", crafted)
    object.__setattr__(_audit(ledger, second), "_InternalLedgerDiagnostics__slot_masks", crafted.flip(dims=(-1,)))
    wrong = ledger.mirror_geometry_consistency(first["public_evidence"], second["public_evidence"])
    assert float(wrong["mask_l1"].max()) > 0.1


def test_mask_geometry_rejects_nonfinite_negative_and_above_one_values(ledger) -> None:
    for invalid in (
        torch.full((1, 21, 2, 2), -0.1),
        torch.full((1, 21, 2, 2), 1.1),
        torch.full((1, 21, 2, 2), float("inf")),
    ):
        with pytest.raises(ValueError):
            ledger.geometry_from_masks(invalid)


def test_rejects_missing_or_invalid_p3_contract_without_using_p4_schema(ledger, prepared_field) -> None:
    invalid = dict(prepared_field)
    invalid.pop("values_by_layer")
    with pytest.raises(KeyError, match="values_by_layer"):
        ledger(invalid)
    invalid = dict(prepared_field)
    invalid["grid_hw"] = (2, 5)
    with pytest.raises(ValueError, match="shape"):
        ledger(invalid)
    assert SCHEMA_PATH.exists()
