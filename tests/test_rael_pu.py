from __future__ import annotations

import importlib
import time

import pytest
import torch
from torch.nn import functional as F


def _modules():
    # Import inside tests so RED is a normal assertion/test failure, never a
    # collection failure if the P12 modules do not exist yet.
    task = importlib.import_module("fate_oia.losses.rael_task_losses")
    pu = importlib.import_module("fate_oia.losses.rael_pu_losses")
    return task, pu


def _sample_ids(batch: int) -> list[str]:
    return [f"bdd-oia-train-{index:05d}.jpg" for index in range(batch)]


def _asymmetric_loss_oracle(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma_positive: float,
    gamma_negative: float,
    clip: float,
) -> torch.Tensor:
    """Independent scalar-by-scalar ASL oracle matching repository semantics."""
    probability = logits.float().sigmoid()
    clipped_negative = (1.0 - probability + clip).clamp(max=1.0)
    positive_term = target.float() * probability.clamp_min(1e-8).log()
    negative_term = (1.0 - target.float()) * clipped_negative.clamp_min(1e-8).log()
    point_probability = probability * target + clipped_negative * (1.0 - target)
    gamma = gamma_positive * target + gamma_negative * (1.0 - target)
    focusing = (1.0 - point_probability).clamp_min(1e-8).pow(gamma)
    return -(positive_term * focusing + negative_term * focusing).mean()


def _grouped_auprc_oracle(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Independent tie-grouped AP: a full equal-score group shares one precision."""
    score = scores.float().cpu()
    target = (targets.float().cpu() > 0.5).float()
    positives = target.sum()
    if positives == 0:
        return torch.tensor(0.0)
    order = torch.argsort(score, descending=True, stable=True)
    ordered_score = score[order]
    ordered_target = target[order]
    total = torch.tensor(0.0)
    start = 0
    while start < score.numel():
        end = start + 1
        while end < score.numel() and ordered_score[end] == ordered_score[start]:
            end += 1
        group_positive = ordered_target[start:end].sum()
        cumulative_positive = ordered_target[:end].sum()
        precision_at_group_end = cumulative_positive / float(end)
        total = total + precision_at_group_end * group_positive
        start = end
    return total / positives


def _reason_inputs(*, batch: int = 4, reasons: int = 21, dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu"):
    torch.manual_seed(20260727 + batch)
    labels = torch.randint(0, 2, (batch, reasons), device=device, dtype=dtype)
    pi = torch.softmax(torch.randn(batch, reasons, 20, device=device, dtype=dtype), dim=-1).requires_grad_()
    rho = torch.full((batch, 20), 0.7, device=device, dtype=dtype, requires_grad=True)
    contributions = torch.randn(batch, reasons, 20, device=device, dtype=dtype, requires_grad=True)
    view_one = torch.rand(batch, reasons, device=device, dtype=dtype, requires_grad=True)
    view_two = torch.rand(batch, reasons, device=device, dtype=dtype, requires_grad=True)
    observed = torch.rand(batch, reasons, device=device, dtype=dtype, requires_grad=True)
    return labels, pi, rho, contributions, view_one, view_two, observed


def test_p12_loss_modules_are_discoverable() -> None:
    task, pu = _modules()
    assert hasattr(task, "multilabel_asymmetric_loss")
    assert hasattr(task, "evidence_conditional_loss")
    assert hasattr(pu, "reason_confidence_weights")
    assert hasattr(pu, "labelwise_pu_audit")


def test_task_losses_are_multilabel_fp32_internal_and_validate_shapes_devices_and_reduction() -> None:
    task, _ = _modules()
    logits = torch.tensor([[20.0, -20.0], [-8.0, 8.0]], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    losses = [
        task.multilabel_asymmetric_loss(logits, target),
        task.soft_f1_loss(logits, target),
        task.two_way_consistency_loss(logits, logits.detach().roll(1, dims=-1)),
        task.multilabel_ranking_loss(logits, target),
        task.evidence_conditional_loss(logits, target * 0.8, torch.full_like(target, 0.7), torch.full_like(target, 0.2)),
    ]
    assert all(loss.dtype == torch.float32 and torch.isfinite(loss) for loss in losses)
    sum(losses).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    none = task.evidence_conditional_loss(logits.detach(), target, torch.ones_like(target), torch.ones_like(target), reduction="none")
    assert none.shape == target.shape and none.dtype == torch.float32
    with pytest.raises(ValueError, match="shape"):
        task.multilabel_asymmetric_loss(logits.detach(), target[:, :1])
    with pytest.raises(ValueError, match="reduction"):
        task.soft_f1_loss(logits.detach(), target, reduction="bad")


def test_atomic_12_asl_matches_repository_equivalent_oracle_with_clipped_negative_focusing() -> None:
    task, _ = _modules()
    logits = torch.tensor([[-2.0, -0.1, 0.2, 3.0], [1.1, -4.0, 5.0, -0.7]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    actual = task.multilabel_asymmetric_loss(logits, target, gamma_positive=0.3, gamma_negative=2.5, clip=0.15)
    expected = _asymmetric_loss_oracle(logits, target, gamma_positive=0.3, gamma_negative=2.5, clip=0.15)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)
    actual.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_atomic_12_tie_grouped_auprc_is_row_reorder_invariant_and_handles_degenerate_labels() -> None:
    _, pu = _modules()
    scores = torch.tensor([0.8, 0.8, 0.1, 0.1, -2.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
    expected = _grouped_auprc_oracle(scores, targets)
    actual = pu.average_precision_binary(scores, targets)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)
    permutation = torch.tensor([1, 3, 4, 0, 2])
    assert torch.allclose(actual, pu.average_precision_binary(scores[permutation], targets[permutation]), atol=1e-7, rtol=1e-7)
    assert pu.average_precision_binary(scores, torch.zeros_like(targets)).item() == 0.0
    assert pu.average_precision_binary(torch.ones(4), torch.ones(4)).item() == 1.0


def test_atomic_12_two_way_and_rank_match_independent_observed_label_oracles() -> None:
    task, _ = _modules()
    first = torch.tensor([[1.0, -2.0, 0.5], [-0.4, 1.2, -1.1]], requires_grad=True)
    second = torch.tensor([[-0.2, 1.1, 0.8], [0.9, -0.6, -1.4]], requires_grad=True)
    expected_two_way = 0.5 * (
        F.binary_cross_entropy_with_logits(first, second.sigmoid().detach())
        + F.binary_cross_entropy_with_logits(second, first.sigmoid().detach())
    )
    actual_two_way = task.two_way_consistency_loss(first, second)
    assert torch.allclose(actual_two_way, expected_two_way, atol=1e-7, rtol=1e-7)
    actual_two_way.backward()
    assert first.grad is not None and second.grad is not None
    rank_logits = torch.tensor([[0.2, 1.4, -0.5, 0.8], [0.5, -0.1, 0.2, 0.4], [0.7, -0.3, 1.2, -0.8]])
    observed = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    # Only row 0 has both observed classes: weakest positive=.2, strongest negative=.8.
    expected_none = torch.tensor([0.85, 0.0, 0.0])
    assert torch.allclose(task.multilabel_ranking_loss(rank_logits, observed, margin=0.25, reduction="none"), expected_none)
    assert torch.allclose(task.multilabel_ranking_loss(rank_logits, observed, margin=0.25), expected_none[:1].mean())


def test_reason_confidence_matches_plan_formula_and_is_fully_stop_gradient() -> None:
    _, pu = _modules()
    labels, pi, rho, contribution, view_one, view_two, observed = _reason_inputs(batch=2)
    values = pu.reason_confidence_weights(pi, rho, contribution, view_one, view_two, observed)
    evidence = (pi * rho.unsqueeze(1) * contribution.sigmoid()).sum(dim=-1)
    view = 1.0 - (view_one - view_two).abs()
    confidence = (evidence * view * observed).clamp_min(0.0).pow(1.0 / 3.0)
    assert torch.allclose(values["c_evidence"], evidence.detach(), atol=1e-6)
    assert torch.allclose(values["c_view"], view.detach(), atol=1e-6)
    assert torch.allclose(values["confidence"], confidence.detach(), atol=1e-6)
    assert torch.allclose(values["positive_weight"], 0.4 + 0.6 * confidence.detach(), atol=1e-6)
    assert torch.allclose(values["negative_weight"], 0.1 + 0.3 * observed.detach() * (1.0 - evidence.detach()), atol=1e-6)
    assert all(not value.requires_grad for value in values.values())
    assert torch.all((values["confidence"] >= 0.0) & (values["confidence"] <= 1.0))
    assert torch.all((values["positive_weight"] >= 0.4) & (values["positive_weight"] <= 1.0))
    assert torch.all((values["negative_weight"] >= 0.1) & (values["negative_weight"] <= 0.4))
    assert labels.shape == values["confidence"].shape


def test_pu_soft_targets_are_all_off_at_update_zero_and_observed_positives_stay_exact_one() -> None:
    _, pu = _modules()
    labels, _, _, contribution, view_one, view_two, observed = _reason_inputs(batch=3)
    evidence_prob = contribution.sigmoid().mean(dim=-1)
    private_prob = torch.rand_like(evidence_prob)
    lam = torch.full((21,), 0.2)
    zero = pu.build_pu_soft_targets(labels, evidence_prob, private_prob, 1.0 - (view_one - view_two).abs(), observed, lam, update_index=0)
    assert torch.equal(zero["effective_lambda"], torch.zeros_like(lam))
    assert torch.equal(zero["soft_targets"], labels)
    active = pu.build_pu_soft_targets(labels, evidence_prob, private_prob, 1.0 - (view_one - view_two).abs(), observed, lam, update_index=1)
    assert torch.equal(active["soft_targets"][labels.bool()], torch.ones_like(active["soft_targets"][labels.bool()]))
    assert torch.all((active["pu_score"] >= 0.0) & (active["pu_score"] <= 1.0))
    assert torch.all((active["soft_targets"] >= 0.0) & (active["soft_targets"] <= 1.0))
    assert all(not value.requires_grad for value in active.values())


def test_labelwise_pu_audit_is_deterministic_uses_train_audit_only_and_opens_only_positive_lcb_labels() -> None:
    _, pu = _modules()
    labels = torch.zeros(160, 21)
    labels[:60, 0] = 1.0
    hidden = pu.deterministic_known_positive_mask(labels, sample_ids=_sample_ids(len(labels)), seed=19, fraction=0.30)
    pu_scores = torch.zeros_like(labels)
    baseline_scores = torch.zeros_like(labels)
    pu_scores[hidden] = 0.99
    baseline_scores[hidden] = 0.01
    baseline_scores[(labels == 0.0)] = 0.8
    first = pu.labelwise_pu_audit(labels, pu_scores, baseline_scores, sample_ids=_sample_ids(len(labels)), split="train_audit", update_index=1, seed=19, resample_count=80)
    second = pu.labelwise_pu_audit(labels, pu_scores, baseline_scores, sample_ids=_sample_ids(len(labels)), split="train_audit", update_index=1, seed=19, resample_count=80)
    assert torch.equal(first["hidden_positive_mask"], hidden)
    assert torch.equal(first["pu_lambda"], second["pu_lambda"])
    assert first["positive_count"][0].item() == 60
    assert first["lcb95_delta_auprc"][0].item() > 0.0
    assert first["active"][0].item() is True
    assert 0.0 < first["pu_lambda"][0].item() <= 0.20 + 1e-7
    assert all(reason in {"active", "count_below_min", "lcb_not_positive", "epoch0_all_off"} for reason in first["active_reason"])
    epoch_zero = pu.labelwise_pu_audit(labels, pu_scores, baseline_scores, sample_ids=_sample_ids(len(labels)), split="train_audit", update_index=0, seed=19, resample_count=20)
    assert torch.equal(epoch_zero["pu_lambda"], torch.zeros(21))
    assert set(epoch_zero["active_reason"]) == {"epoch0_all_off"}
    with pytest.raises(ValueError, match="train_audit"):
        pu.labelwise_pu_audit(labels, pu_scores, baseline_scores, sample_ids=_sample_ids(len(labels)), split="test", update_index=1)


def test_labelwise_pu_audit_all_off_is_valid_when_positive_count_is_below_twenty() -> None:
    _, pu = _modules()
    labels = torch.zeros(100, 21)
    labels[:19, 3] = 1.0
    output = pu.labelwise_pu_audit(labels, torch.rand_like(labels), torch.rand_like(labels), sample_ids=_sample_ids(len(labels)), split="train_audit", update_index=1, resample_count=20)
    assert not output["active"].any()
    assert torch.equal(output["pu_lambda"], torch.zeros(21))
    assert output["active_reason"][3] == "count_below_min"


def test_reason_private_pu_loss_has_a_strict_gradient_firewall() -> None:
    task, pu = _modules()
    labels, pi, rho, contribution, view_one, view_two, observed = _reason_inputs(batch=3)
    private_logits = torch.randn(3, 21, requires_grad=True)
    confidence = pu.reason_confidence_weights(pi, rho, contribution, view_one, view_two, observed)
    targets = pu.build_pu_soft_targets(
        labels,
        contribution.sigmoid().mean(dim=-1),
        private_logits.sigmoid(),
        confidence["c_view"],
        observed,
        torch.full((21,), 0.2),
        update_index=1,
    )
    loss = pu.reason_private_pu_loss(private_logits, targets["soft_targets"], confidence["positive_weight"], confidence["negative_weight"])
    assert torch.allclose(loss, task.evidence_conditional_loss(private_logits, targets["soft_targets"], confidence["positive_weight"], confidence["negative_weight"]))
    loss.backward()
    assert private_logits.grad is not None and float(private_logits.grad.abs().sum()) > 0.0
    for value in (pi, rho, contribution, view_one, view_two, observed):
        assert value.grad is None


def test_pu_supports_batch_one_k21_noncontiguous_and_cuda_bf16() -> None:
    _, pu = _modules()
    labels, pi, rho, contribution, view_one, view_two, observed = _reason_inputs(batch=1)
    pi = torch.softmax(torch.randn(1, 21, 40), dim=-1)[..., ::2].requires_grad_()
    contribution = torch.randn(1, 21, 40)[..., ::2].requires_grad_()
    assert not pi.is_contiguous() and not contribution.is_contiguous()
    values = pu.reason_confidence_weights(pi, rho, contribution, view_one, view_two, observed)
    assert all(torch.isfinite(value).all() for value in values.values())
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    labels, pi, rho, contribution, view_one, view_two, observed = _reason_inputs(batch=1, dtype=torch.bfloat16, device=device)
    values = pu.reason_confidence_weights(pi, rho, contribution, view_one, view_two, observed)
    target = pu.build_pu_soft_targets(labels, contribution.sigmoid().mean(dim=-1), torch.rand_like(labels), values["c_view"], observed, torch.zeros(21, device=device, dtype=torch.bfloat16), update_index=0)
    assert all(torch.isfinite(value).all() for value in values.values())
    assert torch.equal(target["soft_targets"], labels)


def test_p12_strict_public_input_validation_and_zero_or_full_positive_edges() -> None:
    task, pu = _modules()
    labels, pi, rho, contribution, view_one, view_two, observed = _reason_inputs(batch=1)
    with pytest.raises(ValueError, match=r"\[B,21,20\]"):
        pu.reason_confidence_weights(pi[..., :19], rho, contribution[..., :19], view_one, view_two, observed)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        pu.reason_confidence_weights(pi, torch.full_like(rho, 1.1), contribution, view_one, view_two, observed, validate_values=True)
    values = pu.reason_confidence_weights(pi, rho, contribution, view_one, view_two, observed)
    for edge_labels in (torch.zeros_like(labels), torch.ones_like(labels)):
        with pytest.raises(ValueError, match=r"\[0,0.20\]"):
            pu.build_pu_soft_targets(edge_labels, contribution.sigmoid().mean(-1), torch.full_like(edge_labels, 0.5), values["c_view"], observed, torch.full((21,), 0.21), update_index=1, validate_values=True)
        target = pu.build_pu_soft_targets(edge_labels, contribution.sigmoid().mean(-1), torch.full_like(edge_labels, 0.5), values["c_view"], observed, torch.full((21,), 0.2), update_index=1)
        if edge_labels.bool().any():
            assert torch.equal(target["soft_targets"], torch.ones_like(target["soft_targets"]))
        extreme = torch.full((1, 21), 1.0e4, requires_grad=True)
        loss = pu.reason_private_pu_loss(extreme, target["soft_targets"], values["positive_weight"], values["negative_weight"])
        loss = loss + task.multilabel_asymmetric_loss(-extreme, edge_labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert extreme.grad is not None and torch.isfinite(extreme.grad).all()
    if torch.cuda.is_available():
        cuda_pi = pi.detach().cuda()
        cuda_contribution = contribution.detach().cuda()
        cuda_view_one = view_one.detach().cuda()
        cuda_view_two = view_two.detach().cuda()
        cuda_observed = observed.detach().cuda()
        with pytest.raises(ValueError, match="share a device"):
            pu.reason_confidence_weights(cuda_pi, rho, cuda_contribution, cuda_view_one, cuda_view_two, cuda_observed)


def test_atomic_12_hidden_positive_mask_uses_stable_sample_ids_and_exact_half_up_count() -> None:
    _, pu = _modules()
    labels = torch.zeros(20, 21)
    labels[:, 5] = 1.0
    sample_ids = [f"frame-{index:03d}.jpg" for index in range(20)]
    hidden = pu.deterministic_known_positive_mask(labels, sample_ids=sample_ids, seed=880, fraction=0.30)
    assert hidden[:, 5].sum().item() == 6
    permutation = torch.tensor([17, 3, 12, 0, 18, 1, 5, 8, 2, 19, 7, 10, 4, 6, 11, 9, 13, 14, 15, 16])
    shuffled = pu.deterministic_known_positive_mask(labels[permutation], sample_ids=[sample_ids[index] for index in permutation.tolist()], seed=880, fraction=0.30)
    restored = torch.zeros_like(shuffled)
    restored[permutation] = shuffled
    assert torch.equal(hidden, restored)
    with pytest.raises(ValueError, match="sample_ids"):
        pu.deterministic_known_positive_mask(labels, sample_ids=sample_ids[:-1], seed=880)
    two = torch.zeros(2, 21)
    two[:, 5] = 1.0
    assert pu.deterministic_known_positive_mask(two, sample_ids=["a.jpg", "b.jpg"], seed=880, fraction=0.30)[:, 5].sum().item() == 1


def test_p12_sample_id_canonicalization_is_ascii_windows_path_and_integer_stable() -> None:
    _, pu = _modules()
    canonical = pu.canonicalize_sample_id(r"C:\BDD\A.JPG")
    assert canonical == "path:drive:c:/bdd/a.jpg"
    assert canonical == pu.canonicalize_sample_id(r"C:\BDD\seq\..\A.JPG")
    assert canonical == pu.canonicalize_sample_id(r"C:\BDD\.\A.JPG")
    assert canonical == pu.canonicalize_sample_id(r"C:\BDD\A.JPG.")
    assert canonical == pu.canonicalize_sample_id("c:/bdd/a.jpg")
    assert pu.canonicalize_sample_id(r"\\Server\Share\folder\..\A.JPG.") == "path:unc:server/share/a.jpg"
    assert pu.canonicalize_sample_id(r"images\.\front.JPG") == "path:relative/images/front.jpg"
    assert pu.canonicalize_sample_id(17) == "int:17"
    assert pu.canonicalize_sample_id(17) != pu.canonicalize_sample_id("17.jpg")
    for invalid in (True, 1.5, float("nan"), object(), b"sample"):
        with pytest.raises((TypeError, ValueError), match="sample_ids"):
            pu.canonicalize_sample_id(invalid)
    for invalid in ("stra\u00dfe/frame.jpg", "strasse/stra\u00dfe.jpg", "\u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac/frame.jpg", "\u0130stanbul/frame.jpg", "caf\u00e9/scene.jpg", "cafe\u0301/scene.jpg"):
        with pytest.raises(ValueError, match="ASCII"):
            pu.canonicalize_sample_id(invalid)
    for invalid in ("", " file.jpg", "file.jpg ", r"C:relative.jpg", r"\rooted.jpg", r"C:\..\outside.jpg", r"\\server\share\..\outside.jpg", r"..\outside.jpg", "not_an_image.txt"):
        with pytest.raises(ValueError):
            pu.canonicalize_sample_id(invalid)
    labels = torch.zeros(4, 21)
    labels[:, 3] = 1.0
    ids = [101, 205, 307, 409]
    first = pu.deterministic_known_positive_mask(labels, sample_ids=ids, seed=23)
    permutation = torch.tensor([2, 0, 3, 1])
    shuffled = pu.deterministic_known_positive_mask(labels[permutation], sample_ids=[ids[index] for index in permutation.tolist()], seed=23)
    restored = torch.zeros_like(shuffled)
    restored[permutation] = shuffled
    assert torch.equal(first, restored)
    with pytest.raises(ValueError, match="unique"):
        pu.deterministic_known_positive_mask(labels[:2], sample_ids=[r"C:\BDD\A.JPG", "c:/bdd/a.jpg"], seed=23)


def test_p12_debug_validation_rejects_extreme_or_soft_audit_values_but_hot_path_stays_compilable() -> None:
    task, pu = _modules()
    logits = torch.full((1, 21), 1.0e38)
    labels = torch.zeros_like(logits)
    # Default hot path intentionally avoids host-side tensor reductions.
    assert torch.isfinite(task.multilabel_asymmetric_loss(logits, labels))
    with pytest.raises(ValueError, match="abs"):
        task.multilabel_asymmetric_loss(logits, labels, validate_values=True)
    with pytest.raises(ValueError, match="exactly binary"):
        pu.average_precision_binary(torch.tensor([0.2, 0.8]), torch.tensor([0.2, 0.6]))
    _, pi, rho, contribution, view_one, view_two, observed = _reason_inputs(batch=2)
    with pytest.raises(ValueError, match="safe bound"):
        pu.reason_confidence_weights(pi, rho, torch.full_like(contribution, 1.0e38), view_one, view_two, observed, validate_values=True)
    if not hasattr(torch, "compile"):
        return
    hot_logits = torch.randn(2, 21)
    hot_labels = torch.randint(0, 2, (2, 21), dtype=torch.float32)
    compiled_asl = torch.compile(lambda x, y: task.multilabel_asymmetric_loss(x, y), backend="eager", fullgraph=True)
    compiled_evidence = torch.compile(lambda x, y: task.evidence_conditional_loss(x, y, torch.full_like(y, 0.7), torch.full_like(y, 0.2)), backend="eager", fullgraph=True)
    assert torch.isfinite(compiled_asl(hot_logits, hot_labels))
    assert torch.isfinite(compiled_evidence(hot_logits, hot_labels * 0.5))
    compiled_confidence = torch.compile(
        lambda p, r, c, first, second, o: pu.reason_confidence_weights(
            p, r, c, first, second, o
        )["positive_weight"],
        backend="eager",
        fullgraph=True,
    )
    compiled_weights = compiled_confidence(pi, rho, contribution, view_one, view_two, observed)
    assert compiled_weights.shape == hot_labels.shape and torch.isfinite(compiled_weights).all()
    confidence = pu.reason_confidence_weights(pi, rho, contribution, view_one, view_two, observed)
    compiled_soft_target = torch.compile(
        lambda y, e, p, v, o: pu.build_pu_soft_targets(y, e, p, v, o, torch.full((21,), 0.2), update_index=1)["soft_targets"],
        backend="eager",
        fullgraph=True,
    )
    output = compiled_soft_target(hot_labels, contribution.sigmoid().mean(-1), torch.full_like(hot_labels, 0.5), confidence["c_view"], observed)
    assert output.shape == hot_labels.shape and torch.isfinite(output).all()


def test_p12_label_audit_update_zero_and_under_minimum_short_circuit_without_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    _, pu = _modules()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("diagnostic must have short-circuited")

    monkeypatch.setattr(pu, "average_precision_binary", forbidden)
    labels = torch.zeros(64, 21)
    output = pu.labelwise_pu_audit(labels, torch.zeros_like(labels), torch.zeros_like(labels), sample_ids=_sample_ids(64), split="train_audit", update_index=0, resample_count=10)
    assert not output["active"].any()
    assert set(output["active_reason"]) == {"epoch0_all_off"}
    monkeypatch.undo()
    rows = 16_082
    sparse = torch.zeros(rows, 21)
    for label_index in range(21):
        sparse[:19, label_index] = 1.0
    started = time.perf_counter()
    output = pu.labelwise_pu_audit(sparse, torch.zeros_like(sparse), torch.zeros_like(sparse), sample_ids=_sample_ids(rows), split="train_audit", update_index=1, resample_count=200)
    elapsed = time.perf_counter() - started
    assert elapsed < 4.0
    assert not output["active"].any()
    assert set(output["active_reason"]) == {"count_below_min"}
    assert set(output["bootstrap_resample_index_digest"]) == {"not_run"}


def test_atomic_12_one_sided_paired_bootstrap_is_seeded_and_skips_no_positive_resamples() -> None:
    _, pu = _modules()
    scores = torch.tensor([0.95, 0.90, 0.88, 0.30, 0.20, 0.10, 0.05, 0.01])
    baseline = torch.tensor([0.05, 0.10, 0.08, 0.80, 0.70, 0.60, 0.50, 0.40])
    targets = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    first = pu.paired_bootstrap_lcb95(scores, baseline, targets, seed=73, resample_count=100)
    second = pu.paired_bootstrap_lcb95(scores, baseline, targets, seed=73, resample_count=100)
    other_seed = pu.paired_bootstrap_lcb95(scores, baseline, targets, seed=74, resample_count=100)
    assert first["percentile"] == 0.05
    assert first["valid_resample_count"] == second["valid_resample_count"]
    assert torch.equal(first["lcb95"], second["lcb95"])
    assert first["resample_index_digest"] != other_seed["resample_index_digest"]
    assert first["valid_resample_count"] > 0 and torch.isfinite(first["lcb95"])
    degenerate = pu.paired_bootstrap_lcb95(scores, baseline, torch.zeros_like(targets), seed=73, resample_count=20)
    assert degenerate["valid_resample_count"] == 0
    assert torch.isnan(degenerate["lcb95"])
