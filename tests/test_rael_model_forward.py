from __future__ import annotations

import inspect
import os
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch


REQUIRED_KEYS = {
    "action_logits_global",
    "action_logits_visual_global",
    "action_logits_final",
    "reason_logits_global",
    "reason_logits_semantic_global",
    "reason_logits_final",
    "reason_logits_semantic",
    "semantic_reason_tokens",
    "reason_tokens",
    "reason_private_delta",
    "action_tokens",
    "action_semantic_delta",
    "evidence_slots",
    "grounding_outputs",
    "entity_slots",
    "road_slots",
    "latent_slots",
    "latent_feature_view_one",
    "latent_feature_view_two",
    "background_slot",
    "slot_masks",
    "background_mask",
    "slot_area",
    "slot_centroid",
    "slot_scale",
    "slot_activity",
    "slot_presence",
    "slot_observability",
    "slot_type_probs",
    "slot_state_probs",
    "slot_sector_probs",
    "slot_reliability",
    "slot_q_ground",
    "slot_q_view",
    "slot_feature_dropout_consistency",
    "slot_q_state",
    "road_rho_clear",
    "action_slot_weights",
    "reason_slot_weights",
    "action_unary_contributions",
    "reason_unary_contributions",
    "action_pairwise_contributions",
    "reason_pairwise_contributions",
    "action_pairwise_incident_contributions",
    "reason_pairwise_incident_contributions",
    "action_analytical_deletion",
    "reason_analytical_deletion",
    "action_unary_contributions_raw",
    "reason_unary_contributions_raw",
    "action_pairwise_contributions_raw",
    "reason_pairwise_contributions_raw",
    "action_pair_indices",
    "reason_pair_indices",
    "action_global_contribution",
    "reason_global_contribution",
    "named_contribution_ratio",
    "latent_contribution_ratio",
    "positive_contribution",
    "negative_contribution",
    "null_mass",
    "layer_weights_action",
    "layer_weights_reason",
    "layer_weights_slots",
    "clear_left",
    "clear_center",
    "clear_right",
    "occupied_left",
    "occupied_center",
    "occupied_right",
    "pu_scores",
    "pu_active_labels",
    "branch_logits",
    "diagnostics",
}

BRANCH_NAMES = {
    "global_only",
    "global_plus_semantic_bridge",
    "unary_only",
    "pairwise_only",
    "full",
    "no_semantic_reason",
    "semantic_reason_shuffled",
    "reason_private_shuffled",
    "named_slots_only",
    "latent_slots_only",
    "global_context_only",
    "evidence_shuffled",
    "pairwise_off",
    "pu_off",
}


def _model_class() -> type[torch.nn.Module]:
    try:
        from fate_oia.models.rael_oia_model import RAELOIAModel
    except ModuleNotFoundError as exc:
        pytest.fail(f"P16 production model is missing: {exc}", pytrace=False)
    return RAELOIAModel


def _assert_required_contract(output: dict[str, Any]) -> None:
    missing = REQUIRED_KEYS.difference(output)
    assert not missing, f"missing required model outputs: {sorted(missing)}"


def _iter_tensors(value: Any, path: str = "output") -> Sequence[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        return ((path, value),)
    if isinstance(value, Mapping):
        return tuple(
            item
            for key, nested in value.items()
            for item in _iter_tensors(nested, f"{path}.{key}")
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            item
            for index, nested in enumerate(value)
            for item in _iter_tensors(nested, f"{path}[{index}]")
        )
    return ()


def _all_zero_or_none(grads: Sequence[torch.Tensor | None]) -> bool:
    return all(grad is None or torch.count_nonzero(grad).item() == 0 for grad in grads)


class _WrongContractModel:
    def __call__(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"action_logits_final": torch.zeros(images.shape[0], 4)}


@pytest.mark.skipif(
    os.environ.get("RAEL_P16_RED_PROBE") != "1",
    reason="TDD-only wrong-contract probe",
)
def test_red_probe_rejects_collectable_wrong_contract() -> None:
    _assert_required_contract(_WrongContractModel()(torch.zeros(1, 3, 360, 640)))


class FakeDinoExtractor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def forward(
        self, images: torch.Tensor
    ) -> dict[str, torch.Tensor | tuple[int, int] | int]:
        self.call_count += 1
        batch = images.shape[0]
        generator = torch.Generator(device=images.device).manual_seed(41)
        patch = torch.randn(
            batch,
            4,
            45 * 80,
            384,
            generator=generator,
            dtype=images.dtype,
            device=images.device,
        )
        patch = patch + images.mean(dim=(1, 2, 3), keepdim=True).reshape(batch, 1, 1, 1)
        return {
            "patch_tokens_by_layer": patch,
            "cls_tokens_by_layer": patch.mean(dim=2),
            "grid_hw": (45, 80),
            "original_tokens": 3601,
            "dino_call_count": 1,
            "lifetime_dino_call_count": self.call_count,
        }


def _build_model() -> tuple[torch.nn.Module, FakeDinoExtractor]:
    extractor = FakeDinoExtractor()
    return _model_class()(dino_extractor=extractor), extractor


def test_model_api_is_images_only_and_keyword_diagnostics() -> None:
    cls = _model_class()
    assert list(inspect.signature(cls.forward).parameters) == [
        "self",
        "images",
        "diagnostic_modes",
    ]
    assert inspect.signature(cls.forward).parameters["diagnostic_modes"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert list(inspect.signature(cls.decode_from_field).parameters) == [
        "self",
        "field",
        "diagnostic_modes",
    ]


def test_forward_emits_full_finite_schema_with_exact_shapes() -> None:
    model, extractor = _build_model()
    output = model(torch.zeros(1, 3, 360, 640))
    _assert_required_contract(output)
    assert extractor.call_count == 1

    shapes = {
        "action_logits_global": (1, 4),
        "action_logits_visual_global": (1, 4),
        "action_logits_final": (1, 4),
        "reason_logits_global": (1, 21),
        "reason_logits_semantic_global": (1, 21),
        "reason_logits_final": (1, 21),
        "reason_logits_semantic": (1, 21),
        "semantic_reason_tokens": (1, 21, 384),
        "reason_tokens": (1, 21, 384),
        "reason_private_delta": (1, 21, 384),
        "action_tokens": (1, 4, 384),
        "action_semantic_delta": (1, 4, 384),
        "entity_slots": (1, 12, 384),
        "road_slots": (1, 5, 384),
        "latent_slots": (1, 3, 384),
        "latent_feature_view_one": (1, 3, 384),
        "latent_feature_view_two": (1, 3, 384),
        "background_slot": (1, 1, 384),
        "slot_masks": (1, 20, 45, 80),
        "background_mask": (1, 1, 45, 80),
        "slot_area": (1, 20),
        "slot_centroid": (1, 20, 2),
        "slot_scale": (1, 20),
        "slot_reliability": (1, 20),
        "slot_q_ground": (1, 20),
        "slot_q_view": (1, 20),
        "slot_feature_dropout_consistency": (1, 20),
        "slot_q_state": (1, 20),
        "road_rho_clear": (1, 3),
        "action_slot_weights": (1, 4, 21),
        "reason_slot_weights": (1, 21, 21),
        "action_unary_contributions": (1, 4, 20),
        "reason_unary_contributions": (1, 21, 20),
        "action_pairwise_contributions": (1, 4, 190),
        "reason_pairwise_contributions": (1, 21, 190),
        "action_pair_indices": (190, 2),
        "reason_pair_indices": (190, 2),
        "layer_weights_action": (1, 4, 4),
        "layer_weights_reason": (1, 21, 4),
        "layer_weights_slots": (1, 21, 4),
        "pu_scores": (1, 21),
        "pu_active_labels": (21,),
    }
    for key, shape in shapes.items():
        assert tuple(output[key].shape) == shape, key
    assert tuple(output["slot_sector_probs"]["horizontal"].shape) == (1, 20, 3)
    assert tuple(output["slot_sector_probs"]["depth"].shape) == (1, 20, 3)
    tensors = _iter_tensors(output)
    assert tensors, "the formal output must expose tensor diagnostics"
    for path, value in tensors:
        assert torch.isfinite(value).all(), path


def test_reliability_redecode_uses_one_encoded_field_and_recomputes_all_relations() -> None:
    model, extractor = _build_model()
    with torch.no_grad():
        model.action_unary.gamma_unary_raw.fill_(1.0)
        model.reason_unary.gamma_unary_raw.fill_(1.0)
        model.action_pairwise.gamma_pair_raw.fill_(1.0)
        model.reason_pairwise.gamma_pair_raw.fill_(1.0)
    images = torch.randn(1, 3, 360, 640)
    field = model.encode_images(images)
    provisional = model.decode_from_field_provisional(field)
    zero = torch.zeros(1, 20)
    refined = model.decode_from_field_with_reliability(
        field,
        q_ground=zero,
        q_view=zero,
        q_view_sector=torch.zeros(1, 3),
    )

    assert extractor.call_count == 1
    _assert_required_contract(refined)
    assert not refined["slot_reliability"].requires_grad
    assert torch.count_nonzero(refined["slot_reliability"]).item() == 0
    torch.testing.assert_close(
        refined["action_logits_final"], refined["action_logits_global"]
    )
    torch.testing.assert_close(
        refined["reason_logits_final"], refined["reason_logits_global"]
    )
    assert not torch.allclose(
        provisional["action_logits_final"], refined["action_logits_final"]
    )


def test_background_is_not_public_evidence_or_contribution() -> None:
    model, _ = _build_model()
    output = model(torch.zeros(1, 3, 360, 640))
    public = torch.cat(
        [output["entity_slots"], output["road_slots"], output["latent_slots"]], dim=1
    )
    assert public.shape[1] == 20
    assert output["background_slot"].shape[1] == 1
    assert public.untyped_storage().data_ptr() != output[
        "background_slot"
    ].untyped_storage().data_ptr()
    assert output["action_unary_contributions"].shape[-1] == 20
    assert output["reason_pairwise_contributions"].shape[-1] == 190


def test_zero_init_and_contribution_reconstruction_are_exact() -> None:
    model, _ = _build_model()
    output = model(torch.zeros(1, 3, 360, 640))
    action_reconstructed = (
        output["action_global_contribution"]
        + output["action_unary_contributions"].sum(-1)
        + output["action_pairwise_contributions"].sum(-1)
    )
    reason_reconstructed = (
        output["reason_global_contribution"]
        + output["reason_unary_contributions"].sum(-1)
        + output["reason_pairwise_contributions"].sum(-1)
    )
    torch.testing.assert_close(output["action_logits_final"], action_reconstructed)
    torch.testing.assert_close(output["reason_logits_final"], reason_reconstructed)
    torch.testing.assert_close(
        output["action_logits_final"], output["action_logits_global"]
    )
    torch.testing.assert_close(
        output["reason_logits_final"], output["reason_logits_global"]
    )
    assert output["diagnostics"]["action_reconstruction_max_error"] < 1e-6
    assert output["diagnostics"]["reason_reconstruction_max_error"] < 1e-6


def test_default_training_forward_finalizes_layer_collapse_once() -> None:
    model, _ = _build_model()
    model.train()
    output = model(torch.zeros(1, 3, 360, 640))
    collapse = output["diagnostics"]["collapse"]
    assert collapse["collapse_state_updated"] is True
    assert collapse["collapse_batch_token"] == 0
    assert len(model.multilayer_field._finalized_collapse_tokens) == 1


def test_reason_private_has_no_action_gradient_path() -> None:
    model, _ = _build_model()
    output = model(torch.zeros(1, 3, 360, 640))
    grad = torch.autograd.grad(
        output["action_logits_final"].sum(),
        output["reason_private_delta"],
        allow_unused=True,
        retain_graph=True,
    )[0]
    assert grad is None or torch.count_nonzero(grad).item() == 0


def test_pu_score_uses_independent_private_head_and_exact_p12_equation() -> None:
    model, _ = _build_model()
    output = model(torch.zeros(1, 3, 360, 640))
    assert hasattr(model, "pu_private_head")
    assert model.pu_private_head is not model.reason_private.reason_global_head
    assert list(model.pu_private_head.parameters())
    pu = output["diagnostics"]["pu"]
    required = {
        "p_evidence",
        "p_private",
        "p_private_view_one",
        "p_private_view_two",
        "private_logits_view_one",
        "private_logits_view_two",
        "private_probs_view_one",
        "private_probs_view_two",
        "c_view",
        "c_obs",
        "score",
    }
    assert required.issubset(pu)
    assert not torch.equal(pu["private_probs_view_one"], pu["private_probs_view_two"])
    assert pu["private_logits_view_one"].requires_grad
    assert pu["private_logits_view_two"].requires_grad
    assert pu["private_probs_view_one"].requires_grad
    assert pu["private_probs_view_two"].requires_grad
    torch.testing.assert_close(
        pu["p_private_view_one"], pu["private_probs_view_one"].detach()
    )
    torch.testing.assert_close(
        pu["p_private_view_two"], pu["private_probs_view_two"].detach()
    )
    torch.testing.assert_close(
        pu["c_view"],
        (1.0 - (pu["p_private_view_one"] - pu["p_private_view_two"]).abs()).clamp(0.0, 1.0),
    )
    torch.testing.assert_close(
        pu["p_private"],
        (pu["p_private_view_one"] * pu["p_private_view_two"]).sqrt(),
    )
    expected = (
        (pu["p_evidence"] * pu["p_private"]).sqrt()
        * pu["c_view"]
        * pu["c_obs"]
    ).clamp(0.0, 1.0)
    torch.testing.assert_close(output["pu_scores"], expected)
    torch.testing.assert_close(pu["score"], expected)
    for value in (
        output["pu_scores"],
        pu["p_evidence"],
        pu["p_private"],
        pu["p_private_view_one"],
        pu["p_private_view_two"],
        pu["c_view"],
        pu["c_obs"],
        pu["score"],
    ):
        assert not value.requires_grad


def test_pu_private_bce_proxy_updates_only_the_private_owner() -> None:
    model, _ = _build_model()
    output = model(torch.ones(1, 3, 360, 640))
    pu = output["diagnostics"]["pu"]
    logits_one = pu["private_logits_view_one"]
    logits_two = pu["private_logits_view_two"]
    target = torch.zeros_like(logits_one)
    target[:, ::2] = 1.0
    proxy_loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(logits_one, target)
        + torch.nn.functional.binary_cross_entropy_with_logits(logits_two, target)
    )
    private_delta_grad = torch.autograd.grad(
        proxy_loss,
        output["reason_private_delta"],
        allow_unused=True,
        retain_graph=True,
    )[0]
    assert private_delta_grad is None
    protected_params = (
        tuple(model.reason_private.parameters())
        + tuple(model.semantic_reason.parameters())
        + tuple(model.action_category.parameters())
        + tuple(model.action_reason_bridge.parameters())
    )
    optimizer = torch.optim.SGD(model.pu_private_head.parameters(), lr=0.1)
    before = tuple(parameter.detach().clone() for parameter in model.pu_private_head.parameters())
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    proxy_loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.pu_private_head.parameters()
    )
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for parameter in protected_params
    )
    optimizer.step()
    assert any(
        not torch.equal(previous, parameter.detach())
        for previous, parameter in zip(before, model.pu_private_head.parameters())
    )
    expected_score = (
        (pu["p_evidence"] * pu["p_private"]).sqrt() * pu["c_view"] * pu["c_obs"]
    ).clamp(0.0, 1.0)
    torch.testing.assert_close(pu["score"], expected_score)
    assert not pu["score"].requires_grad


def test_integrated_firewall_is_parameter_level_and_keeps_shared_action_path() -> None:
    model, _ = _build_model()
    with torch.no_grad():
        model.action_reason_bridge.gamma_as_raw.fill_(1.0)
    output = model(torch.ones(1, 3, 360, 640))
    private_params = tuple(model.reason_private.parameters()) + tuple(
        model.pu_private_head.parameters()
    )
    action_only_params = tuple(model.action_category.parameters()) + tuple(
        model.action_reason_bridge.parameters()
    )
    shared_params = tuple(model.semantic_reason.parameters())
    action_to_private = torch.autograd.grad(
        output["action_logits_final"].sum(),
        private_params,
        allow_unused=True,
        retain_graph=True,
    )
    reason_to_action = torch.autograd.grad(
        output["reason_logits_final"].sum(),
        action_only_params,
        allow_unused=True,
        retain_graph=True,
    )
    action_to_shared = torch.autograd.grad(
        output["action_logits_final"].sum(),
        shared_params,
        allow_unused=True,
        retain_graph=True,
    )
    assert _all_zero_or_none(action_to_private)
    assert _all_zero_or_none(reason_to_action)
    assert any(grad is not None and torch.count_nonzero(grad).item() > 0 for grad in action_to_shared)


def test_diagnostic_modes_fail_closed_and_branches_are_deterministic() -> None:
    model, extractor = _build_model()
    images = torch.zeros(1, 3, 360, 640)
    counted_modules = {
        "slot_ledger": model.slot_ledger,
        "slot_attribute_heads": model.slot_attribute_heads,
        "semantic_reason": model.semantic_reason,
        "action_category": model.action_category,
        "action_reason_bridge": model.action_reason_bridge,
        "reason_private": model.reason_private,
        "action_unary": model.action_unary,
        "reason_unary": model.reason_unary,
        "action_pairwise": model.action_pairwise,
        "reason_pairwise": model.reason_pairwise,
        "pu_private_head": model.pu_private_head,
        "action_global_projector": model.action_category.global_head,
        "reason_global_head": model.reason_private.reason_global_head,
    }
    observed_calls = {name: 0 for name in counted_modules}
    baseline_hooks = {
        name: set(module._forward_hooks) for name, module in counted_modules.items()
    }
    hooks = [
        module.register_forward_hook(
            lambda _module, _inputs, _output, name=name: observed_calls.__setitem__(
                name, observed_calls[name] + 1
            )
        )
        for name, module in counted_modules.items()
    ]
    with pytest.raises((KeyError, ValueError), match="diagnostic"):
        model(images, diagnostic_modes=("unknown_mode",))
    assert extractor.call_count == 0
    core = model(images)
    for hook in hooks:
        hook.remove()
    for name, module in counted_modules.items():
        assert set(module._forward_hooks) == baseline_hooks[name], name
    assert set(core["branch_logits"]) == {"full"}
    summary = core["diagnostics"]["module_call_summary"]
    assert summary["multilayer_field"] == 1
    for name, count in observed_calls.items():
        assert summary[name] == count, name
    first = model(images, diagnostic_modes=tuple(sorted(BRANCH_NAMES)))
    second = model(images, diagnostic_modes=tuple(sorted(BRANCH_NAMES)))
    assert BRANCH_NAMES == set(first["branch_logits"])
    for path, value in _iter_tensors(first):
        assert torch.isfinite(value).all(), path
    for name in BRANCH_NAMES:
        for task, shape in (("action", (1, 4)), ("reason", (1, 21))):
            assert first["branch_logits"][name][task].shape == shape
            torch.testing.assert_close(
                first["branch_logits"][name][task],
                second["branch_logits"][name][task],
            )
    assert first["branch_logits"]["full"] is not first["branch_logits"]["global_only"]
    torch.testing.assert_close(
        first["branch_logits"]["pu_off"]["action"],
        first["branch_logits"]["full"]["action"],
    )
    torch.testing.assert_close(
        first["branch_logits"]["pu_off"]["reason"],
        first["branch_logits"]["full"]["reason"],
    )
    torch.testing.assert_close(
        first["branch_logits"]["pairwise_off"]["action"],
        first["branch_logits"]["unary_only"]["action"],
    )
    torch.testing.assert_close(
        first["branch_logits"]["pairwise_off"]["reason"],
        first["branch_logits"]["unary_only"]["reason"],
    )
    assert not torch.equal(
        first["branch_logits"]["global_context_only"]["action"],
        first["branch_logits"]["global_only"]["action"],
    )


def test_module_summary_matches_real_hooks_for_all_requested_branches() -> None:
    model, _ = _build_model()
    counted_modules = {
        "slot_ledger": model.slot_ledger,
        "slot_attribute_heads": model.slot_attribute_heads,
        "semantic_reason": model.semantic_reason,
        "action_category": model.action_category,
        "action_reason_bridge": model.action_reason_bridge,
        "reason_private": model.reason_private,
        "pu_private_head": model.pu_private_head,
        "action_unary": model.action_unary,
        "reason_unary": model.reason_unary,
        "action_pairwise": model.action_pairwise,
        "reason_pairwise": model.reason_pairwise,
        "action_global_projector": model.action_category.global_head,
        "reason_global_head": model.reason_private.reason_global_head,
    }
    observed = {name: 0 for name in counted_modules}
    hooks = [
        module.register_forward_hook(
            lambda _module, _inputs, _output, name=name: observed.__setitem__(
                name, observed[name] + 1
            )
        )
        for name, module in counted_modules.items()
    ]
    output = model(
        torch.zeros(1, 3, 360, 640),
        diagnostic_modes=tuple(sorted(BRANCH_NAMES)),
    )
    for hook in hooks:
        hook.remove()
    summary = output["diagnostics"]["module_call_summary"]
    assert summary["multilayer_field"] == 1
    assert {name: summary[name] for name in observed} == observed


def test_evidence_shuffle_breaks_content_identity_after_route_activation() -> None:
    model, _ = _build_model()
    with torch.no_grad():
        model.action_unary.gamma_unary_raw.fill_(1.0)
        model.reason_unary.gamma_unary_raw.fill_(1.0)
        model.action_pairwise.gamma_pair_raw.fill_(1.0)
        model.reason_pairwise.gamma_pair_raw.fill_(1.0)
        # P10 formal projections intentionally start at zero.  Activate only
        # this audit instance so pairwise branches have an observable route.
        model.action_pairwise.pair_output.fill_(0.01)
        model.reason_pairwise.pair_output.fill_(0.01)
    output = model(
        torch.zeros(1, 3, 360, 640),
        diagnostic_modes=("evidence_shuffled",),
    )
    assert not torch.equal(
        output["branch_logits"]["evidence_shuffled"]["action"],
        output["branch_logits"]["full"]["action"],
    )
    assert not torch.equal(
        output["branch_logits"]["evidence_shuffled"]["reason"],
        output["branch_logits"]["full"]["reason"],
    )


def test_requested_private_shuffle_reuses_action_relations_without_recomputing_them() -> None:
    model, _ = _build_model()
    call_counts = {"action_unary": 0, "action_pairwise": 0, "reason_unary": 0, "reason_pairwise": 0}
    hooks = [
        module.register_forward_hook(
            lambda _module, _inputs, _output, name=name: call_counts.__setitem__(
                name, call_counts[name] + 1
            )
        )
        for name, module in (
            ("action_unary", model.action_unary),
            ("action_pairwise", model.action_pairwise),
            ("reason_unary", model.reason_unary),
            ("reason_pairwise", model.reason_pairwise),
        )
    ]
    output = model(
        torch.zeros(1, 3, 360, 640),
        diagnostic_modes=("reason_private_shuffled",),
    )
    for hook in hooks:
        hook.remove()
    assert call_counts == {
        "action_unary": 1,
        "action_pairwise": 1,
        "reason_unary": 2,
        "reason_pairwise": 2,
    }
    summary = output["diagnostics"]["module_call_summary"]
    assert summary["action_unary"] == call_counts["action_unary"]
    assert summary["reason_unary"] == call_counts["reason_unary"]


def test_activated_branches_have_their_declared_semantics_without_false_aliases() -> None:
    model, _ = _build_model()
    with torch.no_grad():
        model.action_reason_bridge.gamma_as_raw.fill_(1.0)
        model.reason_private.gamma_ra_raw.fill_(1.0)
        model.action_unary.gamma_unary_raw.fill_(1.0)
        model.reason_unary.gamma_unary_raw.fill_(1.0)
        model.action_pairwise.gamma_pair_raw.fill_(1.0)
        model.reason_pairwise.gamma_pair_raw.fill_(1.0)
        model.action_pairwise.pair_output.fill_(0.01)
        model.reason_pairwise.pair_output.fill_(0.01)
    branches = model(
        torch.ones(1, 3, 360, 640),
        diagnostic_modes=tuple(sorted(BRANCH_NAMES)),
    )["branch_logits"]
    # Only these two aliases are defined by the diagnostic contract.
    for branch_name, reference_name in (("pu_off", "full"), ("pairwise_off", "unary_only")):
        for target in ("action", "reason"):
            torch.testing.assert_close(
                branches[branch_name][target], branches[reference_name][target]
            )
    comparisons = (
        ("global_plus_semantic_bridge", "global_only", "action"),
        ("unary_only", "global_plus_semantic_bridge", "action"),
        ("pairwise_only", "global_plus_semantic_bridge", "action"),
        ("full", "unary_only", "action"),
        ("no_semantic_reason", "full", "reason"),
        ("semantic_reason_shuffled", "full", "reason"),
        ("reason_private_shuffled", "full", "reason"),
        ("named_slots_only", "latent_slots_only", "reason"),
        ("global_context_only", "global_only", "reason"),
        ("evidence_shuffled", "full", "action"),
    )
    for branch_name, reference_name, target in comparisons:
        assert not torch.equal(
            branches[branch_name][target], branches[reference_name][target]
        ), f"{branch_name} unexpectedly aliases {reference_name} for {target}"


def test_deterministic_branches_match_exact_additive_arithmetic() -> None:
    model, _ = _build_model()
    with torch.no_grad():
        model.action_unary.gamma_unary_raw.fill_(1.0)
        model.reason_unary.gamma_unary_raw.fill_(1.0)
        model.action_pairwise.gamma_pair_raw.fill_(1.0)
        model.reason_pairwise.gamma_pair_raw.fill_(1.0)
    output = model(
        torch.ones(1, 3, 360, 640),
        diagnostic_modes=(
            "global_only",
            "global_plus_semantic_bridge",
            "unary_only",
            "pairwise_only",
            "named_slots_only",
            "latent_slots_only",
            "pairwise_off",
            "pu_off",
        ),
    )
    branches = output["branch_logits"]
    action_global = output["action_logits_global"]
    reason_global = output["reason_logits_global"]
    action_unary = output["action_unary_contributions"]
    reason_unary = output["reason_unary_contributions"]
    action_pair = output["action_pairwise_contributions"]
    reason_pair = output["reason_pairwise_contributions"]
    pair_indices = output["action_pair_indices"]
    torch.testing.assert_close(pair_indices, output["reason_pair_indices"])

    named_slots = torch.arange(20, device=pair_indices.device) < 17
    pair_left, pair_right = pair_indices.unbind(dim=-1)
    named_pairs = named_slots[pair_left] & named_slots[pair_right]
    latent_slots = ~named_slots
    latent_pairs = ~named_pairs

    expected = {
        "global_only": {
            "action": output["action_logits_visual_global"],
            "reason": output["reason_logits_semantic_global"],
        },
        "global_plus_semantic_bridge": {
            "action": action_global,
            "reason": output["reason_logits_semantic_global"],
        },
        "unary_only": {
            "action": action_global + action_unary.sum(-1),
            "reason": reason_global + reason_unary.sum(-1),
        },
        "pairwise_only": {
            "action": action_global + action_pair.sum(-1),
            "reason": reason_global + reason_pair.sum(-1),
        },
        "named_slots_only": {
            "action": (
                action_global
                + action_unary[..., named_slots].sum(-1)
                + action_pair[..., named_pairs].sum(-1)
            ),
            "reason": (
                reason_global
                + reason_unary[..., named_slots].sum(-1)
                + reason_pair[..., named_pairs].sum(-1)
            ),
        },
        "latent_slots_only": {
            "action": (
                action_global
                + action_unary[..., latent_slots].sum(-1)
                + action_pair[..., latent_pairs].sum(-1)
            ),
            "reason": (
                reason_global
                + reason_unary[..., latent_slots].sum(-1)
                + reason_pair[..., latent_pairs].sum(-1)
            ),
        },
        "pairwise_off": {
            "action": action_global + action_unary.sum(-1),
            "reason": reason_global + reason_unary.sum(-1),
        },
        "pu_off": {
            "action": output["action_logits_final"],
            "reason": output["reason_logits_final"],
        },
    }
    for branch_name, task_values in expected.items():
        for task, value in task_values.items():
            torch.testing.assert_close(branches[branch_name][task], value)


def test_default_forward_keeps_reconstruction_errors_on_device() -> None:
    model, _ = _build_model()
    output = model(torch.zeros(1, 3, 360, 640))
    diagnostics = output["diagnostics"]
    assert diagnostics["collapse"]["collapse_state_updated"] is True
    for key in ("action_reconstruction_max_error", "reason_reconstruction_max_error"):
        assert torch.is_tensor(diagnostics[key])
        assert diagnostics[key].device == output["action_logits_final"].device
        assert not diagnostics[key].requires_grad


def test_visual_input_perturbation_reaches_logits_and_call_summary() -> None:
    model, extractor = _build_model()
    zero = model(torch.zeros(1, 3, 360, 640))
    one = model(torch.ones(1, 3, 360, 640))
    assert extractor.call_count == 2
    assert not torch.equal(zero["action_logits_global"], one["action_logits_global"])
    summary = one["diagnostics"]["module_call_summary"]
    assert one["diagnostics"]["dino_call_count"] == 1
    for name in (
        "multilayer_field",
        "semantic_reason",
        "slot_ledger",
        "slot_attribute_heads",
        "action_category",
        "action_reason_bridge",
        "action_unary",
        "reason_unary",
        "action_pairwise",
        "reason_pairwise",
        "pu_private_head",
        "reason_private",
    ):
        assert summary[name] >= 1


def test_shared_evidence_boundary_is_the_exact_tensor_consumed_by_final_logits() -> None:
    model, _ = _build_model()
    with torch.no_grad():
        model.action_unary.gamma_unary_raw.fill_(1.0)
        model.reason_unary.gamma_unary_raw.fill_(1.0)
        model.action_pairwise.gamma_pair_raw.fill_(1.0)
        model.reason_pairwise.gamma_pair_raw.fill_(1.0)
        model.action_pairwise.pair_output.fill_(0.01)
        model.reason_pairwise.pair_output.fill_(0.01)

    images = torch.zeros(1, 3, 360, 640)
    blocked = model(images)
    evidence = blocked["evidence_slots"]
    assert evidence.shape == (1, 20, 384)
    assert blocked["entity_slots"].untyped_storage().data_ptr() == evidence.untyped_storage().data_ptr()
    assert blocked["road_slots"].untyped_storage().data_ptr() == evidence.untyped_storage().data_ptr()
    assert blocked["latent_slots"].untyped_storage().data_ptr() == evidence.untyped_storage().data_ptr()

    hook_calls: list[torch.Tensor] = []
    handle = evidence.register_hook(
        lambda gradient: hook_calls.append(gradient.detach().clone())
        or torch.zeros_like(gradient)
    )
    blocked_loss = (
        blocked["action_logits_final"].square().mean()
        + blocked["reason_logits_final"].square().mean()
    )
    blocked_loss.backward()
    handle.remove()
    assert len(hook_calls) == 1
    assert torch.count_nonzero(hook_calls[0]).item() > 0
    blocked_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.slot_ledger.named_parameters()
        if parameter.grad is not None
    }
    assert blocked_grads

    model.zero_grad(set_to_none=True)
    normal = model(images)
    normal_loss = (
        normal["action_logits_final"].square().mean()
        + normal["reason_logits_final"].square().mean()
    )
    normal_loss.backward()
    normal_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.slot_ledger.named_parameters()
        if parameter.grad is not None
    }
    assert normal_grads.keys() == blocked_grads.keys()
    assert any(
        not torch.equal(normal_grads[name], blocked_grads[name])
        for name in normal_grads
    ), "replacing the public evidence boundary must change upstream ledger gradients"


def test_grounding_outputs_are_raw_current_graph_predictions_not_detached_diagnostics() -> None:
    model, _ = _build_model()
    output = model(torch.zeros(1, 3, 360, 640))
    grounding = output["grounding_outputs"]

    assert set(grounding) == {"entity", "road"}
    assert grounding["entity"]["presence_logits"].shape == (1, 12)
    assert grounding["entity"]["entity_type_logits"].shape == (1, 12, 6)
    assert grounding["entity"]["traffic_state_logits"].shape == (1, 12, 4)
    assert grounding["entity"]["entity_reliability"].shape == (1, 12)
    assert grounding["road"]["drivable_logits"].shape == (1, 3, 45, 80)
    assert grounding["road"]["boundary_logits"].shape == (1, 2, 45, 80)
    assert grounding["road"]["boundary_style_logits"].shape == (1, 2, 3)

    grounding_loss = (
        grounding["entity"]["presence_logits"].square().mean()
        + grounding["road"]["drivable_logits"].square().mean()
        + grounding["road"]["boundary_logits"].square().mean()
        + grounding["road"]["boundary_style_logits"].square().mean()
    )
    entity_grad, ledger_grad = torch.autograd.grad(
        grounding_loss,
        (
            model.slot_attribute_heads.presence_head.weight,
            model.slot_ledger.slot_queries,
        ),
    )
    assert torch.count_nonzero(entity_grad).item() > 0
    assert torch.count_nonzero(ledger_grad).item() > 0


@pytest.mark.parametrize(
    ("target_family", "logit_key"),
    (("action", "action_logits_final"), ("reason", "reason_logits_final")),
)
def test_counterfactual_replay_reconstructs_final_logits_without_another_dino_call(
    target_family: str,
    logit_key: str,
) -> None:
    model, extractor = _build_model()
    with torch.no_grad():
        model.action_unary.gamma_unary_raw.fill_(0.5)
        model.reason_unary.gamma_unary_raw.fill_(0.5)
        model.action_pairwise.gamma_pair_raw.fill_(0.5)
        model.reason_pairwise.gamma_pair_raw.fill_(0.5)
        model.action_pairwise.pair_output.fill_(0.01)
        model.reason_pairwise.pair_output.fill_(0.01)

    visual = model.encode_images(torch.zeros(1, 3, 360, 640))
    output = model.decode_from_field(visual)
    replay = model.build_counterfactual_replay(
        visual,
        output,
        target_family=target_family,
    )

    assert extractor.call_count == 1
    assert replay["shared_field"].shape == (1, 4 * 384, 45, 80)
    replayed = (
        replay["public_readout"](replay["shared_field"])
        + replay["public_contribution"](replay["shared_field"])
    )
    torch.testing.assert_close(replayed, output[logit_key], atol=2.0e-6, rtol=0.0)
    assert extractor.call_count == 1


def _counterfactual_test_masks(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Provide one public selected/control pair with matched geometry."""

    masks = torch.zeros(1, 20, 45, 80, device=device, dtype=dtype)
    masks[:, 0, 18:22, 16:20] = 1.0
    masks[:, 1, 18:22, 28:32] = 1.0
    return masks


@pytest.mark.parametrize("target_family", ("action", "reason"))
def test_counterfactual_replay_uses_original_boundaries_without_ledger_or_relation_bypass(
    target_family: str,
) -> None:
    """P14 replay may perturb values, but training gradients stay at P16 boundaries."""

    from fate_oia.losses.rael_counterfactual_losses import run_feature_intervention

    model, extractor = _build_model()
    with torch.no_grad():
        model.action_unary.gamma_unary_raw.fill_(0.8)
        model.reason_unary.gamma_unary_raw.fill_(0.8)
        model.action_pairwise.gamma_pair_raw.fill_(0.8)
        model.reason_pairwise.gamma_pair_raw.fill_(0.8)
        model.action_pairwise.pair_output.fill_(0.03)
        model.reason_pairwise.pair_output.fill_(0.03)
        model.action_reason_bridge.gamma_as_raw.fill_(1.0)

    visual = model.encode_images(torch.zeros(1, 3, 360, 640))
    output = model.decode_from_field(visual)
    masks = _counterfactual_test_masks(
        device=output["evidence_slots"].device,
        dtype=output["evidence_slots"].dtype,
    )
    output["slot_masks"] = masks
    sectors = torch.zeros(1, 20, 3, device=masks.device, dtype=masks.dtype)
    sectors[..., 1] = 1.0
    targets = 4 if target_family == "action" else 21
    analytical = torch.zeros(1, targets, 20, device=masks.device, dtype=masks.dtype)
    analytical[:, 0, 0] = 10.0
    logit_key = f"{target_family}_logits_final"
    replay = model.build_counterfactual_replay(
        visual,
        output,
        target_family=target_family,
    )

    evidence_grads: list[torch.Tensor] = []
    semantic_grads: list[torch.Tensor] = []
    evidence_hook = output["evidence_slots"].register_hook(
        lambda gradient: evidence_grads.append(gradient.detach().clone())
        or gradient
    )
    semantic_hook = output["semantic_reason_tokens"].register_hook(
        lambda gradient: semantic_grads.append(gradient.detach().clone())
        or gradient
    )
    ledger_calls = 0

    def _count_ledger(*_: object) -> None:
        nonlocal ledger_calls
        ledger_calls += 1

    ledger_handle = model.slot_ledger.register_forward_hook(_count_ledger)
    result = run_feature_intervention(
        optimizer_update=8,
        shared_field=replay["shared_field"],
        slot_masks=masks,
        sector_probs=sectors,
        base_logits=output[logit_key],
        analytical_deletion=analytical,
        public_readout=replay["public_readout"],
        public_contribution=replay["public_contribution"],
        case_ids=["cf-boundary.jpg"],
    )
    ledger_handle.remove()

    assert result["available"] is True
    assert result["selected_slot"].item() == 0
    assert result["control_slot"].item() == 1
    assert extractor.call_count == 1
    assert ledger_calls == 0
    assert result["loss"] is not None
    result["loss"].backward()
    evidence_hook.remove()
    semantic_hook.remove()

    assert evidence_grads and torch.count_nonzero(evidence_grads[0]).item() > 0
    if target_family == "reason":
        assert semantic_grads and torch.count_nonzero(semantic_grads[0]).item() > 0
    else:
        assert not semantic_grads or torch.count_nonzero(semantic_grads[0]).item() == 0
    for module in (
        model.action_unary,
        model.reason_unary,
        model.action_pairwise,
        model.reason_pairwise,
    ):
        assert all(parameter.grad is None for parameter in module.parameters())


def test_counterfactual_replay_has_no_direct_ledger_path_when_evidence_boundary_is_blocked() -> None:
    """A zeroed P16 evidence hook must also zero every replay-ledger gradient."""

    from fate_oia.losses.rael_counterfactual_losses import run_feature_intervention

    model, _ = _build_model()
    with torch.no_grad():
        model.action_unary.gamma_unary_raw.fill_(0.8)
        model.action_pairwise.gamma_pair_raw.fill_(0.8)
        model.action_pairwise.pair_output.fill_(0.03)

    visual = model.encode_images(torch.zeros(1, 3, 360, 640))
    output = model.decode_from_field(visual)
    masks = _counterfactual_test_masks(
        device=output["evidence_slots"].device,
        dtype=output["evidence_slots"].dtype,
    )
    output["slot_masks"] = masks
    sectors = torch.zeros(1, 20, 3, device=masks.device, dtype=masks.dtype)
    sectors[..., 1] = 1.0
    analytical = torch.zeros(1, 4, 20, device=masks.device, dtype=masks.dtype)
    analytical[:, 0, 0] = 10.0
    replay = model.build_counterfactual_replay(visual, output, target_family="action")
    handle = output["evidence_slots"].register_hook(torch.zeros_like)
    result = run_feature_intervention(
        optimizer_update=8,
        shared_field=replay["shared_field"],
        slot_masks=masks,
        sector_probs=sectors,
        base_logits=output["action_logits_final"],
        analytical_deletion=analytical,
        public_readout=replay["public_readout"],
        public_contribution=replay["public_contribution"],
        case_ids=["cf-ledger-firewall.jpg"],
    )
    assert result["available"] is True
    result["loss"].backward()
    handle.remove()
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for parameter in model.slot_ledger.parameters()
    )


def test_counterfactual_replay_uses_residual_anchored_second_readout_jacobian_not_ste() -> None:
    """E08-R3 requires I + dRI/dE - dR0/dE, never an identity STE."""

    from fate_oia.losses.rael_counterfactual_losses import (
        neighborhood_background_mean,
        replace_region_with_neighbor_mean,
    )

    model, _ = _build_model()
    with torch.no_grad():
        model.action_unary.gamma_unary_raw.fill_(0.8)
        model.action_pairwise.gamma_pair_raw.fill_(0.8)
        model.action_pairwise.pair_output.fill_(0.03)

    visual = model.encode_images(torch.zeros(1, 3, 360, 640))
    output = model.decode_from_field(visual)
    masks = _counterfactual_test_masks(
        device=output["evidence_slots"].device,
        dtype=output["evidence_slots"].dtype,
    )
    output["slot_masks"] = masks
    replay = model.build_counterfactual_replay(visual, output, target_family="action")

    shared_field = replay["shared_field"]
    selected_mask = masks[:, 0]
    replacement, available = neighborhood_background_mean(shared_field, selected_mask)
    assert bool(available.item())
    intervened_field = replace_region_with_neighbor_mean(
        shared_field,
        selected_mask,
        replacement,
    )

    evidence = output["evidence_slots"]
    probe = torch.linspace(
        -0.75,
        0.75,
        evidence.numel(),
        device=evidence.device,
        dtype=evidence.dtype,
    ).reshape_as(evidence)

    identity_evidence = replay["intervened_evidence"](shared_field)
    torch.testing.assert_close(identity_evidence, evidence, atol=0.0, rtol=0.0)

    r0 = replay["second_readout"](shared_field)
    ri = replay["second_readout"](intervened_field)
    intervened_evidence = replay["intervened_evidence"](intervened_field)

    actual_vjp = torch.autograd.grad(
        (intervened_evidence * probe).sum(),
        evidence,
        retain_graph=True,
    )[0]
    ri_vjp = torch.autograd.grad(
        (ri * probe).sum(),
        evidence,
        retain_graph=True,
    )[0]
    r0_vjp = torch.autograd.grad(
        (r0 * probe).sum(),
        evidence,
        retain_graph=True,
    )[0]
    expected_vjp = probe + ri_vjp - r0_vjp

    torch.testing.assert_close(actual_vjp, expected_vjp, atol=2.0e-6, rtol=2.0e-5)
    assert not torch.allclose(
        actual_vjp,
        probe,
        atol=2.0e-6,
        rtol=2.0e-5,
    ), "an identity-only evidence Jacobian is the forbidden STE path"


def test_analytical_deletion_is_unary_plus_every_incident_pair() -> None:
    model, _ = _build_model()
    with torch.no_grad():
        model.action_unary.gamma_unary_raw.fill_(0.5)
        model.reason_unary.gamma_unary_raw.fill_(0.5)
        model.action_pairwise.gamma_pair_raw.fill_(0.5)
        model.reason_pairwise.gamma_pair_raw.fill_(0.5)
        model.action_pairwise.pair_output.fill_(0.01)
        model.reason_pairwise.pair_output.fill_(0.01)
    output = model(torch.zeros(1, 3, 360, 640))

    for family in ("action", "reason"):
        pair = output[f"{family}_pairwise_contributions"]
        indices = output[f"{family}_pair_indices"]
        manual = pair.new_zeros(pair.shape[0], pair.shape[1], 20)
        left = indices[:, 0].view(1, 1, -1).expand_as(pair)
        right = indices[:, 1].view(1, 1, -1).expand_as(pair)
        manual.scatter_add_(2, left, pair)
        manual.scatter_add_(2, right, pair)
        torch.testing.assert_close(
            output[f"{family}_pairwise_incident_contributions"], manual
        )
        torch.testing.assert_close(
            output[f"{family}_analytical_deletion"],
            output[f"{family}_unary_contributions"] + manual,
        )


def test_latent_feature_dropout_views_are_distinct_and_train_the_shared_ledger() -> None:
    model, _ = _build_model()
    output = model(torch.zeros(1, 3, 360, 640))
    first = output["latent_feature_view_one"]
    second = output["latent_feature_view_two"]
    assert not torch.equal(first, second)
    feature_view_loss = (
        1.0
        - torch.nn.functional.cosine_similarity(first, second, dim=-1)
    ).mean()
    gradient = torch.autograd.grad(
        feature_view_loss, model.slot_ledger.slot_queries
    )[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() > 0


def _ledger_grad_norm(model: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.slot_ledger.parameters()
        if parameter.grad is not None
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mixed BF16 replay requires CUDA")
def test_counterfactual_slot_readout_accepts_fp32_frozen_routes_with_bf16_values() -> None:
    model, _ = _build_model()
    model = model.cuda()
    values = torch.randn(1, 4, 3600, 384, device="cuda", dtype=torch.bfloat16)
    masks = torch.rand(1, 20, 45, 80, device="cuda", dtype=torch.float32)
    layer_weights = torch.softmax(
        torch.randn(1, 20, 4, device="cuda", dtype=torch.float32), dim=-1
    )

    pooled = model._counterfactual_slot_readout(values, masks, layer_weights)

    assert pooled.dtype == torch.bfloat16
    assert pooled.shape == (1, 20, 384)
    assert bool(torch.isfinite(pooled).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mixed BF16 replay requires CUDA")
def test_counterfactual_second_readout_casts_frozen_gru_state_to_bf16_values() -> None:
    model, _ = _build_model()
    model = model.cuda()
    evidence = torch.randn(
        1, 20, 384, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    values = torch.randn(1, 4, 3600, 384, device="cuda", dtype=torch.bfloat16)
    masks = torch.rand(1, 20, 45, 80, device="cuda", dtype=torch.float32)
    layer_weights = torch.softmax(
        torch.randn(1, 20, 4, device="cuda", dtype=torch.float32), dim=-1
    )

    output = model._counterfactual_second_readout(
        evidence, values, masks, layer_weights
    )
    output.float().sum().backward()

    assert output.dtype == torch.bfloat16
    assert bool(torch.isfinite(output).all())
    assert evidence.grad is not None
    assert float(evidence.grad.float().abs().sum()) > 0.0


def _zero_admission(loss: torch.Tensor, evidence: torch.Tensor) -> list[torch.Tensor]:
    observed: list[torch.Tensor] = []
    handle = evidence.register_hook(
        lambda gradient: observed.append(gradient.detach().clone())
        or torch.zeros_like(gradient)
    )
    loss.backward()
    handle.remove()
    return observed


def test_e04_r2_canonical_mask_has_only_evidence_vjp_and_detached_field_keys() -> None:
    model, _ = _build_model()
    field = model.encode_images(torch.zeros(1, 3, 360, 640))
    output = model.decode_from_field(field)
    evidence = output["evidence_slots"].detach().requires_grad_(True)
    field_keys = field.prepared_field["keys_by_layer"]
    masks = model._canonical_slot_masks(evidence, field_keys)

    assert masks.shape == (1, 20, 45, 80)
    assert torch.isfinite(masks).all()
    torch.testing.assert_close(
        masks.sum(dim=1), torch.ones_like(masks[:, 0]), atol=1.0e-6, rtol=0.0
    )
    mask_loss = masks.square().mean()
    evidence_vjp, field_vjp = torch.autograd.grad(
        mask_loss, (evidence, field_keys), allow_unused=True
    )
    assert evidence_vjp is not None and torch.count_nonzero(evidence_vjp).item() > 0
    assert field_vjp is None or torch.count_nonzero(field_vjp).item() == 0


def test_e04_r2_mask_only_grounding_is_admission_dominated_and_updates_ledger_when_admitted() -> None:
    model, _ = _build_model()
    with torch.no_grad():
        # Remove the road-token adjustment: this leaves the real grounding
        # head driven only by the canonical road mask logit.
        model.slot_attribute_heads.drivable_token_head.weight.zero_()
        model.slot_attribute_heads.drivable_token_head.bias.zero_()
    images = torch.zeros(1, 3, 360, 640)
    blocked = model(images)
    assert "ledger_slot_masks_diagnostic" in blocked
    assert blocked["ledger_slot_masks_diagnostic"].requires_grad is False
    mask_loss = blocked["grounding_outputs"]["road"]["drivable_logits"].square().mean()
    observed = _zero_admission(mask_loss, blocked["evidence_slots"])
    assert len(observed) == 1 and torch.count_nonzero(observed[0]).item() > 0
    assert _ledger_grad_norm(model) == 0.0

    model.zero_grad(set_to_none=True)
    admitted = model(images)
    before = [parameter.detach().clone() for parameter in model.slot_ledger.parameters()]
    optimizer = torch.optim.SGD(model.slot_ledger.parameters(), lr=1.0e-3)
    admitted_loss = admitted["grounding_outputs"]["road"]["drivable_logits"].square().mean()
    evidence_grad = torch.autograd.grad(
        admitted_loss, admitted["evidence_slots"], retain_graph=True
    )[0]
    assert torch.count_nonzero(evidence_grad).item() > 0
    admitted_loss.backward()
    assert _ledger_grad_norm(model) > 0.0
    optimizer.step()
    assert any(
        not torch.equal(before_value, parameter)
        for before_value, parameter in zip(before, model.slot_ledger.parameters())
    )


def test_e04_r2_boundary_and_pairwise_geometry_cannot_bypass_evidence_admission() -> None:
    model, _ = _build_model()
    with torch.no_grad():
        model.action_pairwise.gamma_pair_raw.fill_(1.0)
        model.action_pairwise.pair_output.fill_(0.1)
    output = model(torch.zeros(1, 3, 360, 640))

    boundary_loss = output["grounding_outputs"]["road"]["boundary_logits"].square().mean()
