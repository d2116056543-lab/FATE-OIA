from __future__ import annotations

import importlib
import itertools
import copy
from collections import OrderedDict
import multiprocessing
import pickle
import queue
import threading

import pytest
import torch
from torch import nn


def _p13_spawn_pickle_roundtrip(payload: bytes, result_queue) -> None:
    controller = pickle.loads(payload)
    result_queue.put(
        (
            str(controller.evidence_action_ema.dtype),
            str(controller.evidence_action_ema.device),
            int(controller.evidence_ema_updates.item()),
            controller.evidence_action_ema.cpu().tolist(),
            controller.semantic_action_ema.cpu().tolist(),
        )
    )


def _module():
    try:
        return importlib.import_module("fate_oia.optim.rael_gradient_admission")
    except ModuleNotFoundError as error:
        pytest.fail(f"P13 RED: gradient admission module is absent: {error}")


def _project_oracle(gradient: torch.Tensor, anchor: torch.Tensor, eps: float) -> torch.Tensor:
    anchor_batched = anchor.unsqueeze(0).expand_as(gradient)
    dot = (gradient.float() * anchor_batched.float()).sum(dim=-1, keepdim=True)
    denom = anchor_batched.float().square().sum(dim=-1, keepdim=True) + eps
    coefficient = torch.minimum(torch.zeros_like(dot), dot / denom)
    return gradient.float() - coefficient * anchor_batched.float()


def _cap_oracle(gradient: torch.Tensor, anchor: torch.Tensor, ratio: float, eps: float) -> torch.Tensor:
    cap = ratio * anchor.float().norm(dim=-1, keepdim=True).unsqueeze(0)
    norm = gradient.float().norm(dim=-1, keepdim=True)
    scale = torch.minimum(torch.ones_like(norm), cap / (norm + eps))
    return gradient.float() * scale


def test_p13_admission_core_matches_projection_budget_and_zero_anchor() -> None:
    module = _module()
    eps = 1.0e-8
    action = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    reason = torch.tensor([[[-4.0, 3.0], [2.0, -5.0]], [[-2.0, 1.0], [3.0, -4.0]]])
    grounding = torch.tensor([[[-5.0, 2.0], [4.0, -3.0]], [[-1.0, 4.0], [5.0, -2.0]]])
    counterfactual = torch.tensor([[[-7.0, 0.0], [0.0, -7.0]], [[-3.0, 0.0], [0.0, -3.0]]])
    anchor = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    output = module.admission_core(action, reason, grounding, counterfactual, anchor, eps=eps)
    reason_expected = _cap_oracle(_project_oracle(reason, anchor, eps), anchor, 0.25, eps)
    grounding_expected = _cap_oracle(_project_oracle(grounding, anchor, eps), anchor, 0.15, eps)
    cf_expected = _cap_oracle(_project_oracle(counterfactual, anchor, eps), anchor, 0.05, eps)
    assert torch.allclose(output.projected_reason, _project_oracle(reason, anchor, eps), atol=1e-6)
    assert torch.allclose(output.admitted_reason, reason_expected, atol=1e-6)
    assert torch.allclose(output.admitted_grounding, grounding_expected, atol=1e-6)
    assert torch.allclose(output.admitted_counterfactual, cf_expected, atol=1e-6)
    assert torch.allclose(output.admitted, action + reason_expected + grounding_expected + cf_expected, atol=1e-6)
    zero = module.admission_core(action, reason, grounding, counterfactual, torch.zeros_like(anchor), eps=eps)
    assert torch.equal(zero.admitted_reason, torch.zeros_like(reason))
    assert torch.equal(zero.admitted_grounding, torch.zeros_like(grounding))
    assert torch.equal(zero.admitted_counterfactual, torch.zeros_like(counterfactual))
    assert torch.equal(zero.admitted, action)


def test_p13_projection_preserves_the_exact_eps_denominator_contract() -> None:
    """A deliberately visible eps catches normalized-anchor approximations."""

    module = _module()
    gradient = torch.tensor([[[-3.0, 2.0]]])
    anchor = torch.tensor([[0.1, 0.0]])
    eps = 0.25
    expected = _project_oracle(gradient, anchor, eps)
    actual = module.project_against_action_ema(gradient, anchor, eps=eps)
    assert torch.allclose(actual, expected, atol=1.0e-7, rtol=1.0e-7)


def test_p13_projection_preserves_aligned_and_orthogonal_components() -> None:
    module = _module()
    anchor = torch.tensor([[2.0, 0.0]])
    aligned = torch.tensor([[[3.0, 7.0]]])
    orthogonal = torch.tensor([[[0.0, -5.0]]])
    anti_aligned = torch.tensor([[[-3.0, 0.0]]])
    assert torch.equal(module.project_against_action_ema(aligned, anchor), aligned)
    assert torch.equal(module.project_against_action_ema(orthogonal, anchor), orthogonal)
    assert torch.allclose(module.project_against_action_ema(anti_aligned, anchor), torch.zeros_like(anti_aligned))


def test_p13_per_boundary_ema_state_is_independent_and_resume_exact() -> None:
    module = _module()
    controller = module.RAELGradientAdmission(ema_decay=0.95)
    evidence_action = torch.full((2, 20, 3), 4.0)
    semantic_action = torch.full((2, 21, 3), -2.0)
    zero_evidence = torch.zeros_like(evidence_action)
    zero_semantic = torch.zeros_like(semantic_action)
    first = controller.admit_from_gradients(
        evidence_action=evidence_action,
        evidence_reason=zero_evidence,
        evidence_grounding=zero_evidence,
        evidence_counterfactual=zero_evidence,
        semantic_action=semantic_action,
        semantic_reason=zero_semantic,
        semantic_grounding=zero_semantic,
        semantic_counterfactual=zero_semantic,
    )
    assert torch.allclose(controller.evidence_action_ema, torch.full((20, 3), 0.2))
    assert torch.allclose(controller.semantic_action_ema, torch.full((21, 3), -0.1))
    assert first.evidence.active and first.semantic.active
    state = controller.state_dict()
    resumed = module.RAELGradientAdmission(ema_decay=0.95)
    resumed.load_state_dict(state)
    assert torch.equal(resumed.evidence_action_ema, controller.evidence_action_ema)
    assert torch.equal(resumed.semantic_action_ema, controller.semantic_action_ema)
    controller.admit_from_gradients(
        evidence_action=evidence_action,
        evidence_reason=zero_evidence,
        evidence_grounding=zero_evidence,
        evidence_counterfactual=zero_evidence,
        semantic_action=semantic_action,
        semantic_reason=zero_semantic,
        semantic_grounding=zero_semantic,
        semantic_counterfactual=zero_semantic,
    )
    assert torch.allclose(controller.evidence_action_ema, torch.full((20, 3), 0.39))
    assert torch.allclose(controller.semantic_action_ema, torch.full((21, 3), -0.195))


def test_p13_autograd_extraction_hook_cleanup_and_private_gradient_isolation() -> None:
    module = _module()
    torch.manual_seed(13)
    shared_evidence = nn.Parameter(torch.randn(2, 20, 4))
    shared_semantic = nn.Parameter(torch.randn(2, 21, 4))
    action_private = nn.Parameter(torch.randn(4))
    reason_private = nn.Parameter(torch.randn(4))
    grounding_private = nn.Parameter(torch.randn(4))
    evidence_slots = shared_evidence * 1.0
    semantic_tokens = shared_semantic * 1.0
    action_loss = evidence_slots.mean(dim=(1, 2)) + semantic_tokens.mean(dim=(1, 2)) + action_private.square().mean()
    reason_loss = -(evidence_slots.square().mean(dim=(1, 2)) + semantic_tokens.square().mean(dim=(1, 2))) + reason_private.square().mean()
    grounding_loss = evidence_slots.sin().mean(dim=(1, 2)) + semantic_tokens.cos().mean(dim=(1, 2)) + grounding_private.square().mean()
    counterfactual_loss = -(evidence_slots * semantic_tokens[:, :20]).mean(dim=(1, 2))
    controller = module.RAELGradientAdmission()
    admitted = controller.admit_from_losses(
        evidence_slots=evidence_slots,
        semantic_reason_tokens=semantic_tokens,
        action_loss=action_loss,
        reason_loss=reason_loss,
        grounding_loss=grounding_loss,
        counterfactual_loss=counterfactual_loss,
    )
    assert shared_evidence.grad is None and shared_semantic.grad is None
    assert admitted.evidence.active and admitted.semantic.active
    expected_action_private = torch.autograd.grad(action_loss.sum(), action_private, retain_graph=True)[0]
    expected_reason_private = torch.autograd.grad(reason_loss.sum(), reason_private, retain_graph=True)[0]
    expected_grounding_private = torch.autograd.grad(grounding_loss.sum(), grounding_private, retain_graph=True)[0]
    with controller.replace_shared_boundary_gradients(
        evidence_slots=evidence_slots,
        semantic_reason_tokens=semantic_tokens,
        admission=admitted,
    ) as hooks:
        (action_loss.sum() + reason_loss.sum() + grounding_loss.sum() + counterfactual_loss.sum()).backward()
        assert hooks.active
    assert not hooks.active
    assert torch.allclose(shared_evidence.grad, admitted.evidence.admitted, atol=1e-6)
    assert torch.allclose(shared_semantic.grad, admitted.semantic.admitted, atol=1e-6)
    assert torch.allclose(action_private.grad, expected_action_private, atol=1e-6)
    assert torch.allclose(reason_private.grad, expected_reason_private, atol=1e-6)
    assert torch.allclose(grounding_private.grad, expected_grounding_private, atol=1e-6)
    fresh_evidence = (shared_evidence.detach().clone().requires_grad_() * 1.0)
    fresh_semantic = (shared_semantic.detach().clone().requires_grad_() * 1.0)
    with pytest.raises(RuntimeError, match="P13 forced cleanup"):
        with controller.replace_shared_boundary_gradients(
            evidence_slots=fresh_evidence,
            semantic_reason_tokens=fresh_semantic,
            admission=admitted,
        ) as failed_hooks:
            raise RuntimeError("P13 forced cleanup")
    assert not failed_hooks.active


def test_p13_per_sample_autograd_grad_none_and_noncontiguous_contract() -> None:
    module = _module()
    boundary = torch.randn(2, 20, 8, requires_grad=True)[..., ::2]
    active_loss = boundary.square().mean(dim=(1, 2))
    gradient = module.per_sample_boundary_grad(active_loss, boundary)
    assert gradient is not None and gradient.shape == boundary.shape
    unrelated = torch.randn(2, 21, requires_grad=True).mean(dim=1)
    assert module.per_sample_boundary_grad(unrelated, boundary) is None
    with pytest.raises(ValueError, match=r"\[B\]"):
        module.per_sample_boundary_grad(boundary.square(), boundary)


def test_p13_inactive_accumulation_and_fullgraph_core_contract() -> None:
    module = _module()
    controller = module.RAELGradientAdmission()
    inactive = controller.admit_from_gradients(
        evidence_action=None,
        evidence_reason=torch.randn(1, 20, 2),
        evidence_grounding=None,
        evidence_counterfactual=None,
        semantic_action=None,
        semantic_reason=torch.randn(1, 21, 2),
        semantic_grounding=None,
        semantic_counterfactual=None,
    )
    assert not inactive.evidence.active and inactive.evidence.admitted is None
    assert controller.evidence_action_ema.numel() == 0 and controller.evidence_ema_updates.item() == 0
    for _ in range(2):
        evidence = (torch.randn(1, 20, 3, requires_grad=True) * 1.0)
        semantic = (torch.randn(1, 21, 3, requires_grad=True) * 1.0)
        action = evidence.mean(dim=(1, 2)) + semantic.mean(dim=(1, 2))
        reason = -evidence.square().mean(dim=(1, 2)) - semantic.square().mean(dim=(1, 2))
        admitted = controller.admit_from_losses(
            evidence_slots=evidence,
            semantic_reason_tokens=semantic,
            action_loss=action,
            reason_loss=reason,
            grounding_loss=torch.zeros_like(action),
            counterfactual_loss=torch.zeros_like(action),
        )
        with controller.replace_shared_boundary_gradients(
            evidence_slots=evidence,
            semantic_reason_tokens=semantic,
            admission=admitted,
        ):
            (action + reason).sum().backward()
    assert controller.evidence_ema_updates.item() == 2
    assert controller.semantic_ema_updates.item() == 2
    assert all(not tensor.requires_grad for tensor in controller.state_dict().values())
    if hasattr(torch, "compile"):
        compiled = torch.compile(
            lambda a, r, g, c, m: module.admission_core(a, r, g, c, m).admitted,
            backend="eager",
            fullgraph=True,
        )
        inputs = [torch.randn(1, 20, 3) for _ in range(4)]
        output = compiled(*inputs, torch.randn(20, 3))
        assert output.shape == (1, 20, 3) and torch.isfinite(output).all()


def test_p13_hook_context_rejects_reentry_and_only_tracks_two_boundaries() -> None:
    module = _module()
    evidence = torch.zeros(1, 20, 2, requires_grad=True)
    semantic = torch.zeros(1, 21, 2, requires_grad=True)
    controller = module.RAELGradientAdmission()
    admitted = controller.admit_from_gradients(
        evidence_action=torch.ones_like(evidence),
        evidence_reason=torch.zeros_like(evidence),
        evidence_grounding=torch.zeros_like(evidence),
        evidence_counterfactual=torch.zeros_like(evidence),
        semantic_action=torch.ones_like(semantic),
        semantic_reason=torch.zeros_like(semantic),
        semantic_grounding=torch.zeros_like(semantic),
        semantic_counterfactual=torch.zeros_like(semantic),
    )
    hooks = controller.replace_shared_boundary_gradients(
        evidence_slots=evidence,
        semantic_reason_tokens=semantic,
        admission=admitted,
    )
    assert tuple(name for name, _, _ in hooks._entries) == ("evidence_slots", "semantic_reason_tokens")
    with hooks:
        with pytest.raises(RuntimeError, match="cannot be entered twice"):
            hooks.__enter__()


def test_p13_admit_from_losses_treats_none_as_exact_zero_without_ema_update() -> None:
    module = _module()
    torch.manual_seed(1315)
    controller = module.RAELGradientAdmission()
    seed_evidence = torch.ones(2, 20, 3)
    seed_semantic = torch.full((2, 21, 3), 2.0)
    controller.admit_from_gradients(
        evidence_action=seed_evidence,
        evidence_reason=torch.zeros_like(seed_evidence),
        evidence_grounding=torch.zeros_like(seed_evidence),
        evidence_counterfactual=torch.zeros_like(seed_evidence),
        semantic_action=seed_semantic,
        semantic_reason=torch.zeros_like(seed_semantic),
        semantic_grounding=torch.zeros_like(seed_semantic),
        semantic_counterfactual=torch.zeros_like(seed_semantic),
    )
    prior_evidence_ema = controller.evidence_action_ema.clone()
    prior_semantic_ema = controller.semantic_action_ema.clone()
    prior_counts = (controller.evidence_ema_updates.clone(), controller.semantic_ema_updates.clone())
    evidence_leaf = torch.randn(2, 20, 3, requires_grad=True)
    semantic_leaf = torch.randn(2, 21, 3, requires_grad=True)
    evidence = evidence_leaf * 1.0
    semantic = semantic_leaf * 1.0
    reason = -evidence.square().mean((1, 2)) - semantic.square().mean((1, 2))
    output = controller.admit_from_losses(
        evidence_slots=evidence,
        semantic_reason_tokens=semantic,
        action_loss=None,
        reason_loss=reason,
        grounding_loss=None,
        counterfactual_loss=None,
    )
    expected_evidence_reason = module.per_sample_boundary_grad(reason, evidence)
    expected_semantic_reason = module.per_sample_boundary_grad(reason, semantic)
    assert torch.equal(output.evidence.action_gradient, torch.zeros_like(evidence))
    assert torch.equal(output.semantic.action_gradient, torch.zeros_like(semantic))
    assert torch.allclose(output.evidence.reason_gradient, expected_evidence_reason)
    assert torch.allclose(output.semantic.reason_gradient, expected_semantic_reason)
    assert torch.equal(controller.evidence_action_ema, prior_evidence_ema)
    assert torch.equal(controller.semantic_action_ema, prior_semantic_ema)
    assert torch.equal(controller.evidence_ema_updates, prior_counts[0])
    assert torch.equal(controller.semantic_ema_updates, prior_counts[1])
    assert evidence_leaf.grad is None and semantic_leaf.grad is None
    assert not bool(output.evidence.diagnostics["action_loss_active"])
    assert bool(output.evidence.diagnostics["reason_loss_active"])


def test_p13_admit_from_losses_all_none_returns_zero_admission_without_autograd() -> None:
    module = _module()
    controller = module.RAELGradientAdmission()
    evidence = torch.randn(1, 20, 3, requires_grad=True)
    semantic = torch.randn(1, 21, 3, requires_grad=True)
    output = controller.admit_from_losses(
        evidence_slots=evidence,
        semantic_reason_tokens=semantic,
        action_loss=None,
        reason_loss=None,
        grounding_loss=None,
        counterfactual_loss=None,
    )
    for boundary, tensor in ((output.evidence, evidence), (output.semantic, semantic)):
        assert not boundary.active
        assert boundary.inactive_reason == "all_losses_none"
        assert torch.equal(boundary.action_gradient, torch.zeros_like(tensor))
        assert torch.equal(boundary.reason_gradient, torch.zeros_like(tensor))
        assert torch.equal(boundary.grounding_gradient, torch.zeros_like(tensor))
        assert torch.equal(boundary.counterfactual_gradient, torch.zeros_like(tensor))
        assert torch.equal(boundary.admitted, torch.zeros_like(tensor))
        assert not bool(boundary.diagnostics["action_loss_active"])
    assert controller.evidence_action_ema.numel() == 0
    assert controller.semantic_action_ema.numel() == 0
    assert controller.evidence_ema_updates.item() == 0
    assert controller.semantic_ema_updates.item() == 0
    assert evidence.grad is None and semantic.grad is None


def test_p13_boundary_hook_is_one_shot_and_cleanup_is_idempotent() -> None:
    module = _module()
    controller = module.RAELGradientAdmission()
    evidence = nn.Parameter(torch.zeros(1, 20, 2))
    semantic = nn.Parameter(torch.zeros(1, 21, 2))
    admission = controller.admit_from_gradients(
        evidence_action=torch.ones_like(evidence),
        evidence_reason=torch.zeros_like(evidence),
        evidence_grounding=torch.zeros_like(evidence),
        evidence_counterfactual=torch.zeros_like(evidence),
        semantic_action=torch.ones_like(semantic),
        semantic_reason=torch.zeros_like(semantic),
        semantic_grounding=torch.zeros_like(semantic),
        semantic_counterfactual=torch.zeros_like(semantic),
    )
    hooks = controller.replace_shared_boundary_gradients(
        evidence_slots=evidence,
        semantic_reason_tokens=semantic,
        admission=admission,
    )
    with hooks:
        (evidence.sum() + semantic.sum()).backward(retain_graph=True)
        assert torch.equal(evidence.grad, admission.evidence.admitted)
        assert torch.equal(semantic.grad, admission.semantic.admitted)
        evidence.grad.zero_()
        semantic.grad.zero_()
        (3.0 * evidence.sum() + 3.0 * semantic.sum()).backward()
        assert torch.equal(evidence.grad, torch.full_like(evidence.grad, 3.0))
        assert torch.equal(semantic.grad, torch.full_like(semantic.grad, 3.0))
    hooks.close()
    assert not hooks.active
    before = controller.replace_shared_boundary_gradients(
        evidence_slots=evidence,
        semantic_reason_tokens=semantic,
        admission=admission,
    )
    with pytest.raises(RuntimeError, match="before trigger"):
        with before:
            raise RuntimeError("before trigger")
    assert not before.active
    untriggered_evidence = torch.zeros(1, 20, 2, requires_grad=True)
    untriggered_semantic = torch.zeros(1, 21, 2, requires_grad=True)
    untriggered = controller.replace_shared_boundary_gradients(
        evidence_slots=untriggered_evidence,
        semantic_reason_tokens=untriggered_semantic,
        admission=admission,
    )
    with untriggered:
        pass
    (2.0 * untriggered_evidence.sum() + 2.0 * untriggered_semantic.sum()).backward()
    assert torch.equal(untriggered_evidence.grad, torch.full_like(untriggered_evidence.grad, 2.0))
    assert torch.equal(untriggered_semantic.grad, torch.full_like(untriggered_semantic.grad, 2.0))
    after = controller.replace_shared_boundary_gradients(
        evidence_slots=evidence,
        semantic_reason_tokens=semantic,
        admission=admission,
    )
    with pytest.raises(RuntimeError, match="after trigger"):
        with after:
            (evidence.sum() + semantic.sum()).backward(retain_graph=True)
            raise RuntimeError("after trigger")
    assert not after.active


@pytest.mark.parametrize("enabled", list(itertools.product((False, True), repeat=4)))
def test_p13_admit_from_losses_supports_every_none_combination(enabled: tuple[bool, bool, bool, bool]) -> None:
    module = _module()
    evidence = torch.randn(1, 20, 2, requires_grad=True)
    semantic = torch.randn(1, 21, 2, requires_grad=True)
    terms = (
        evidence.mean((1, 2)) + semantic.mean((1, 2)),
        -evidence.square().mean((1, 2)) - semantic.square().mean((1, 2)),
        evidence.sin().mean((1, 2)) + semantic.cos().mean((1, 2)),
        -(evidence * semantic[:, :20]).mean((1, 2)),
    )
    controller = module.RAELGradientAdmission()
    output = controller.admit_from_losses(
        evidence_slots=evidence,
        semantic_reason_tokens=semantic,
        action_loss=terms[0] if enabled[0] else None,
        reason_loss=terms[1] if enabled[1] else None,
        grounding_loss=terms[2] if enabled[2] else None,
        counterfactual_loss=terms[3] if enabled[3] else None,
    )
    for boundary, tensor in ((output.evidence, evidence), (output.semantic, semantic)):
        for name, is_enabled in zip(("action", "reason", "grounding", "counterfactual"), enabled):
            gradient = getattr(boundary, f"{name}_gradient")
            if not is_enabled:
                assert torch.equal(gradient, torch.zeros_like(tensor))
            assert bool(boundary.diagnostics[f"{name}_loss_active"]) is is_enabled
    assert controller.evidence_ema_updates.item() == int(enabled[0])
    assert controller.semantic_ema_updates.item() == int(enabled[0])
    if not any(enabled):
        assert not output.evidence.active and not output.semantic.active


def test_p13_all_none_hook_is_pass_through_and_amp_scale_recovers_admission() -> None:
    module = _module()
    evidence = nn.Parameter(torch.zeros(1, 20, 2))
    semantic = nn.Parameter(torch.zeros(1, 21, 2))
    controller = module.RAELGradientAdmission()
    inactive = controller.admit_from_losses(
        evidence_slots=evidence,
        semantic_reason_tokens=semantic,
        action_loss=None,
        reason_loss=None,
        grounding_loss=None,
        counterfactual_loss=None,
    )
    with controller.replace_shared_boundary_gradients(
        evidence_slots=evidence,
        semantic_reason_tokens=semantic,
        admission=inactive,
    ):
        (evidence.sum() + semantic.sum()).backward()
    assert torch.equal(evidence.grad, torch.ones_like(evidence))
    assert torch.equal(semantic.grad, torch.ones_like(semantic))
    evidence.grad.zero_()
    semantic.grad.zero_()
    active = controller.admit_from_gradients(
        evidence_action=torch.ones_like(evidence),
        evidence_reason=torch.zeros_like(evidence),
        evidence_grounding=torch.zeros_like(evidence),
        evidence_counterfactual=torch.zeros_like(evidence),
        semantic_action=torch.ones_like(semantic),
        semantic_reason=torch.zeros_like(semantic),
        semantic_grounding=torch.zeros_like(semantic),
        semantic_counterfactual=torch.zeros_like(semantic),
    )
    scale = 1024.0
    with controller.replace_shared_boundary_gradients(
        evidence_slots=evidence,
        semantic_reason_tokens=semantic,
        admission=active,
        backward_scale=scale,
    ):
        (scale * (evidence.sum() + semantic.sum())).backward()
    assert torch.allclose(evidence.grad / scale, active.evidence.admitted, atol=1e-6)
    assert torch.allclose(semantic.grad / scale, active.semantic.admitted, atol=1e-6)
    with pytest.raises(ValueError, match="backward_scale"):
        controller.replace_shared_boundary_gradients(
            evidence_slots=evidence,
            semantic_reason_tokens=semantic,
            admission=active,
            backward_scale=0.0,
        )


def test_p13_state_load_respects_destination_dtype_and_device_policy() -> None:
    module = _module()
    source = module.RAELGradientAdmission(state_dtype=torch.float32)
    evidence = torch.full((1, 20, 2), 4.0)
    semantic = torch.full((1, 21, 2), -2.0)
    source.admit_from_gradients(
        evidence_action=evidence,
        evidence_reason=torch.zeros_like(evidence),
        evidence_grounding=torch.zeros_like(evidence),
        evidence_counterfactual=torch.zeros_like(evidence),
        semantic_action=semantic,
        semantic_reason=torch.zeros_like(semantic),
        semantic_grounding=torch.zeros_like(semantic),
        semantic_counterfactual=torch.zeros_like(semantic),
    )
    destination = module.RAELGradientAdmission(state_dtype=torch.float64)
    destination.load_state_dict(source.state_dict())
    assert destination.evidence_action_ema.dtype is torch.float64
    assert destination.semantic_action_ema.dtype is torch.float64
    assert destination.evidence_action_ema.device.type == "cpu"
    assert torch.allclose(destination.evidence_action_ema, source.evidence_action_ema.double())
    initialized = module.RAELGradientAdmission(state_dtype=torch.float64)
    initialized.admit_from_gradients(
        evidence_action=evidence.double(),
        evidence_reason=torch.zeros_like(evidence.double()),
        evidence_grounding=torch.zeros_like(evidence.double()),
        evidence_counterfactual=torch.zeros_like(evidence.double()),
        semantic_action=semantic.double(),
        semantic_reason=torch.zeros_like(semantic.double()),
        semantic_grounding=torch.zeros_like(semantic.double()),
        semantic_counterfactual=torch.zeros_like(semantic.double()),
    )
    initialized.load_state_dict(source.state_dict())
    assert initialized.evidence_action_ema.dtype is torch.float64
    assert torch.allclose(initialized.evidence_action_ema, source.evidence_action_ema.double())
    initialized.to("cpu")
    assert initialized.evidence_action_ema.dtype is torch.float64


def test_p13_rejects_multi_rank_ddp_without_explicit_synchronization(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(module.torch.distributed, "get_world_size", lambda: 2)
    controller = module.RAELGradientAdmission()
    evidence = torch.zeros(1, 20, 2)
    semantic = torch.zeros(1, 21, 2)
    with pytest.raises(RuntimeError, match="single-process"):
        controller.admit_from_gradients(
            evidence_action=torch.ones_like(evidence),
            evidence_reason=torch.zeros_like(evidence),
            evidence_grounding=torch.zeros_like(evidence),
            evidence_counterfactual=torch.zeros_like(evidence),
            semantic_action=torch.ones_like(semantic),
            semantic_reason=torch.zeros_like(semantic),
            semantic_grounding=torch.zeros_like(semantic),
            semantic_counterfactual=torch.zeros_like(semantic),
        )


def test_p13_uses_four_batched_vjps_and_matches_independent_jacobian(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    torch.manual_seed(1316)
    evidence_leaf = torch.randn(2, 20, 2, requires_grad=True)
    semantic_leaf = torch.randn(2, 21, 2, requires_grad=True)
    evidence = evidence_leaf * 1.0
    semantic = semantic_leaf * 1.0
    action = evidence.mean((1, 2)) + semantic.mean((1, 2))
    reason = -evidence.square().mean((1, 2)) - semantic.square().mean((1, 2))
    grounding = evidence.sin().mean((1, 2)) + semantic.cos().mean((1, 2))
    counterfactual = -(evidence * semantic[:, :20]).mean((1, 2))
    expected_evidence = torch.autograd.functional.jacobian(
        lambda value: -value.square().mean((1, 2)), evidence.detach()
    )
    expected_semantic = torch.autograd.functional.jacobian(
        lambda value: -value.square().mean((1, 2)), semantic.detach()
    )
    index = torch.arange(evidence.shape[0])
    original_grad = module.torch.autograd.grad
    calls = 0

    def counted_grad(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(module.torch.autograd, "grad", counted_grad)
    output = module.RAELGradientAdmission().admit_from_losses(
        evidence_slots=evidence,
        semantic_reason_tokens=semantic,
        action_loss=action,
        reason_loss=reason,
        grounding_loss=grounding,
        counterfactual_loss=counterfactual,
    )
    assert calls == 4
    assert torch.allclose(output.evidence.reason_gradient, expected_evidence[index, index], atol=1e-6)
    assert torch.allclose(output.semantic.reason_gradient, expected_semantic[index, index], atol=1e-6)


def test_p13_state_anchor_controls_default_to_dtype_device_checkpoint_load() -> None:
    module = _module()
    source = module.RAELGradientAdmission()
    evidence = torch.full((1, 20, 2), 3.0)
    semantic = torch.full((1, 21, 2), -3.0)
    source.admit_from_gradients(
        evidence_action=evidence, evidence_reason=torch.zeros_like(evidence),
        evidence_grounding=torch.zeros_like(evidence), evidence_counterfactual=torch.zeros_like(evidence),
        semantic_action=semantic, semantic_reason=torch.zeros_like(semantic),
        semantic_grounding=torch.zeros_like(semantic), semantic_counterfactual=torch.zeros_like(semantic),
    )
    checkpoint = source.state_dict()
    assert "_state_anchor" not in checkpoint
    destination = module.RAELGradientAdmission().to(dtype=torch.float64)
    destination.load_state_dict(checkpoint)
    assert destination.evidence_action_ema.dtype is torch.float64
    assert destination.semantic_action_ema.dtype is torch.float64
    assert torch.allclose(destination.evidence_action_ema, source.evidence_action_ema.double())
    if torch.cuda.is_available():
        cuda_destination = module.RAELGradientAdmission().to(device="cuda", dtype=torch.float64)
        cuda_destination.load_state_dict(checkpoint)
        assert cuda_destination.evidence_action_ema.dtype is torch.float64
        assert cuda_destination.evidence_action_ema.device.type == "cuda"
        assert torch.allclose(cuda_destination.evidence_action_ema, source.evidence_action_ema.double().cuda())


def test_p13_lock_survives_pickle_deepcopy_and_windows_spawn_roundtrip() -> None:
    module = _module()
    controller = module.RAELGradientAdmission()
    evidence = torch.full((1, 20, 2), 2.0)
    semantic = torch.full((1, 21, 2), -2.0)
    controller.admit_from_gradients(
        evidence_action=evidence, evidence_reason=torch.zeros_like(evidence),
        evidence_grounding=torch.zeros_like(evidence), evidence_counterfactual=torch.zeros_like(evidence),
        semantic_action=semantic, semantic_reason=torch.zeros_like(semantic),
        semantic_grounding=torch.zeros_like(semantic), semantic_counterfactual=torch.zeros_like(semantic),
    )
    for clone in (pickle.loads(pickle.dumps(controller)), copy.deepcopy(controller)):
        assert torch.equal(clone.evidence_action_ema, controller.evidence_action_ema)
        clone.admit_from_gradients(
            evidence_action=evidence, evidence_reason=torch.zeros_like(evidence),
            evidence_grounding=torch.zeros_like(evidence), evidence_counterfactual=torch.zeros_like(evidence),
            semantic_action=semantic, semantic_reason=torch.zeros_like(semantic),
            semantic_grounding=torch.zeros_like(semantic), semantic_counterfactual=torch.zeros_like(semantic),
        )
        assert clone.evidence_ema_updates.item() == 2
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_p13_spawn_pickle_roundtrip, args=(pickle.dumps(controller), result_queue))
    process.start()
    process.join(timeout=30)
    assert process.exitcode == 0
    try:
        dtype, device, count, evidence_values, semantic_values = result_queue.get(timeout=5)
    except queue.Empty as error:
        raise AssertionError("P13 spawn roundtrip produced no result") from error
    finally:
        result_queue.close()
        result_queue.join_thread()
    assert dtype == str(controller.evidence_action_ema.dtype)
    assert device == str(controller.evidence_action_ema.device)
    assert count == controller.evidence_ema_updates.item()
    assert torch.equal(torch.tensor(evidence_values), controller.evidence_action_ema.cpu())
    assert torch.equal(torch.tensor(semantic_values), controller.semantic_action_ema.cpu())


def test_p13_state_dict_is_atomic_cloned_snapshot_under_update_pressure() -> None:
    module = _module()
    controller = module.RAELGradientAdmission()
    stop = threading.Event()
    errors: list[BaseException] = []

    def updater() -> None:
        try:
            for step in range(1, 81):
                evidence = torch.full((1, 20, 2), float(step))
                semantic = torch.full((1, 21, 2), -float(step))
                zeros_evidence = torch.zeros_like(evidence)
                zeros_semantic = torch.zeros_like(semantic)
                controller.admit_from_gradients(
                    evidence_action=evidence, evidence_reason=zeros_evidence,
                    evidence_grounding=zeros_evidence, evidence_counterfactual=zeros_evidence,
                    semantic_action=semantic, semantic_reason=zeros_semantic,
                    semantic_grounding=zeros_semantic, semantic_counterfactual=zeros_semantic,
                )
        except BaseException as error:
            errors.append(error)
        finally:
            stop.set()

    worker = threading.Thread(target=updater)
    worker.start()
    snapshots = []
    while not stop.is_set() or worker.is_alive():
        snapshot = controller.state_dict()
        snapshots.append(snapshot)
        assert snapshot["evidence_ema_updates"].item() == snapshot["semantic_ema_updates"].item()
        if snapshot["evidence_action_ema"].numel():
            assert torch.allclose(
                snapshot["evidence_action_ema"].mean(),
                -snapshot["semantic_action_ema"].mean(),
            )
    worker.join(timeout=10)
    assert not errors
    if not snapshots:
        snapshots.append(controller.state_dict())
    held = next(snapshot for snapshot in snapshots if snapshot["evidence_action_ema"].numel())
    held_values = {name: value.clone() for name, value in held.items()}
    evidence = torch.ones(1, 20, 2)
    semantic = -torch.ones(1, 21, 2)
    controller.admit_from_gradients(
        evidence_action=evidence, evidence_reason=torch.zeros_like(evidence),
        evidence_grounding=torch.zeros_like(evidence), evidence_counterfactual=torch.zeros_like(evidence),
        semantic_action=semantic, semantic_reason=torch.zeros_like(semantic),
        semantic_grounding=torch.zeros_like(semantic), semantic_counterfactual=torch.zeros_like(semantic),
    )
    for name, value in held_values.items():
        assert torch.equal(held[name], value)


def test_p13_state_dict_preserves_destination_prefix_and_metadata_with_clones() -> None:
    module = _module()
    controller = module.RAELGradientAdmission()
    evidence = torch.ones(1, 20, 2)
    semantic = -torch.ones(1, 21, 2)
    controller.admit_from_gradients(
        evidence_action=evidence, evidence_reason=torch.zeros_like(evidence),
        evidence_grounding=torch.zeros_like(evidence), evidence_counterfactual=torch.zeros_like(evidence),
        semantic_action=semantic, semantic_reason=torch.zeros_like(semantic),
        semantic_grounding=torch.zeros_like(semantic), semantic_counterfactual=torch.zeros_like(semantic),
    )
    destination = OrderedDict()
    destination._metadata = OrderedDict()
    snapshot = controller.state_dict(destination=destination, prefix="p13.")
    assert snapshot is destination
    assert "p13.evidence_action_ema" in snapshot
    assert hasattr(snapshot, "_metadata")
    snapshot["p13.evidence_action_ema"].fill_(77.0)
    assert not torch.equal(snapshot["p13.evidence_action_ema"], controller.evidence_action_ema)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="P13 CUDA probe requires CUDA")
def test_p13_cuda_bf16_batch1_noncontiguous_and_extreme_norms() -> None:
    module = _module()
    device = torch.device("cuda")
    base = torch.randn(1, 20, 8, device=device, dtype=torch.bfloat16, requires_grad=True)
    evidence = base[..., ::2]
    semantic_base = torch.randn(1, 21, 8, device=device, dtype=torch.bfloat16, requires_grad=True)
    semantic = semantic_base[..., ::2]
    controller = module.RAELGradientAdmission().to(device)
    action = evidence.float().mean(dim=(1, 2)) + semantic.float().mean(dim=(1, 2))
    reason = -(evidence.float().square().mean(dim=(1, 2)) + semantic.float().square().mean(dim=(1, 2)))
    admitted = controller.admit_from_losses(
        evidence_slots=evidence,
        semantic_reason_tokens=semantic,
        action_loss=action,
        reason_loss=reason,
        grounding_loss=torch.zeros_like(action),
        counterfactual_loss=torch.zeros_like(action),
    )
    with controller.replace_shared_boundary_gradients(
        evidence_slots=evidence,
        semantic_reason_tokens=semantic,
        admission=admitted,
    ):
        (action + reason).sum().backward()
    for tensor in (base.grad, semantic_base.grad, controller.evidence_action_ema, controller.semantic_action_ema):
        assert tensor is not None and torch.isfinite(tensor).all()
    cuda_destination = module.RAELGradientAdmission(state_dtype=torch.float64).to(device)
    cuda_destination.load_state_dict(controller.state_dict())
    assert cuda_destination.evidence_action_ema.dtype is torch.float64
    assert cuda_destination.semantic_action_ema.dtype is torch.float64
    assert cuda_destination.evidence_action_ema.device == controller.evidence_action_ema.device
    assert torch.allclose(cuda_destination.evidence_action_ema, controller.evidence_action_ema.double())
    for scale in (0.0, 1.0e-20, 1.0e20):
        action_gradient = torch.full((1, 20, 4), scale, device=device, dtype=torch.bfloat16)
        auxiliary = -torch.ones_like(action_gradient)
        output = module.admission_core(action_gradient, auxiliary, auxiliary, auxiliary, action_gradient[0].float())
        assert torch.isfinite(output.admitted).all()
