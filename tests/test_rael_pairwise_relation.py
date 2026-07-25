from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from fate_oia.models.rael_relation_contributions import RAELPairwiseContribution


def _inputs(
    *,
    batch: int = 2,
    targets: int = 4,
    dim: int = 16,
    height: int = 5,
    width: int = 7,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    noncontiguous: bool = False,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260726 + targets)
    def dense(shape: tuple[int, ...]) -> torch.Tensor:
        value = torch.randn(*shape, device=device, dtype=dtype)
        if not noncontiguous:
            return value
        return torch.randn(*shape[:-1], shape[-1] * 2, device=device, dtype=dtype)[..., ::2]

    target_tokens = dense((batch, targets, dim)).requires_grad_()
    evidence_tokens = dense((batch, 20, dim)).requires_grad_()
    masks = torch.rand(batch, 20, height, width * (2 if noncontiguous else 1), device=device, dtype=dtype)
    slot_masks = (masks[..., ::2] if noncontiguous else masks).requires_grad_()
    sectors = torch.softmax(dense((batch, 20, 3)), dim=-1).requires_grad_()
    pi = torch.softmax(dense((batch, targets, 20)), dim=-1).requires_grad_()
    return {
        "target_tokens": target_tokens,
        "evidence_tokens": evidence_tokens,
        "slot_masks": slot_masks,
        "sector_probs": sectors,
        "unary_public_pi": pi,
        "reliability": torch.full((batch, 20), 0.75, device=device, dtype=dtype).requires_grad_(),
    }


def _module(*, targets: int = 4, dim: int = 16, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> RAELPairwiseContribution:
    torch.manual_seed(91 + targets)
    return RAELPairwiseContribution(num_targets=targets, dim=dim, gamma_cap=0.25).to(device=device, dtype=dtype)


def _pair_column(module: RAELPairwiseContribution, left: int, right: int) -> int:
    matches = torch.nonzero(
        (module.pair_indices[:, 0] == left) & (module.pair_indices[:, 1] == right),
        as_tuple=False,
    )
    assert matches.numel() == 1
    return int(matches.item())


def _asl(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.float().sigmoid().clamp(1e-6, 1.0 - 1e-6)
    return (
        -targets.float() * torch.log(probabilities)
        - (1.0 - targets.float()) * probabilities.square() * torch.log1p(-probabilities)
    ).mean()


def _evidence_conditional(
    logits: torch.Tensor,
    y_tilde: torch.Tensor,
    w_pos: torch.Tensor,
    w_neg: torch.Tensor,
) -> torch.Tensor:
    """Independent K=21 EvidenceConditional reference; no action ASL is mixed in."""
    return (
        w_pos.float() * y_tilde.float() * F.softplus(-logits.float())
        + w_neg.float() * (1.0 - y_tilde.float()) * F.softplus(logits.float())
    ).mean()


def _owner_inputs(values: dict[str, torch.Tensor], *, targets: int) -> dict[str, torch.Tensor]:
    return {"global_logits": torch.randn(values["target_tokens"].shape[0], targets, requires_grad=True), **values}


def _assert_detached_externals(owner: dict[str, torch.Tensor]) -> None:
    for name, value in owner.items():
        assert value.grad is None, name


def _step(optimizer: torch.optim.Optimizer) -> None:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_pairwise_uses_exactly_one_hundred_ninety_canonical_unordered_pairs_and_no_formal_auxiliary() -> None:
    module = _module()
    output = module(**_inputs())
    expected = torch.combinations(torch.arange(20), r=2)
    assert torch.equal(module.pair_indices.cpu(), expected)
    assert output["pair_indices"].shape == (190, 2)
    assert torch.unique(output["pair_indices"], dim=0).shape[0] == 190
    assert bool((output["pair_indices"][:, 0] < output["pair_indices"][:, 1]).all())
    assert output["pair_geometry"].shape == (2, 190, 10)
    assert output["pair_contributions_raw"].shape == (2, 4, 190)
    assert output["pair_raw_sum"].shape == (2, 4)
    assert "pair_auxiliary_delta" not in output
    assert "owner_pair_auxiliary_delta" not in output


def test_pair_geometry_has_fp32_finite_centroid_mass_iou_and_sector_terms_for_empty_masks() -> None:
    module = _module()
    values = _inputs(batch=1, height=2, width=2)
    masks = torch.zeros_like(values["slot_masks"])
    masks[0, 0, 0, 0] = 1.0
    masks[0, 1, 0, 1] = 1.0
    masks[0, 1, 1, 1] = 1.0
    values["slot_masks"] = masks.requires_grad_()
    sectors = torch.zeros_like(values["sector_probs"])
    sectors[0, :, 0] = 1.0
    sectors[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    sectors[0, 1] = torch.tensor([0.0, 1.0, 0.0])
    values["sector_probs"] = sectors.requires_grad_()
    geometry = module(**values)["pair_geometry"]
    first_pair = geometry[0, _pair_column(module, 0, 1)]
    assert torch.allclose(first_pair[:4], torch.tensor([-2.0, -1.0, torch.log(torch.tensor(0.5)), 0.0]), atol=1e-6)
    assert torch.equal(first_pair[4:7], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(first_pair[7:10], torch.tensor([0.0, 1.0, 0.0]))
    assert bool(torch.isfinite(geometry).all())


def test_formal_pair_path_keeps_pi_trainable_stops_only_rho_and_reconstructs_exactly() -> None:
    module = _module()
    with torch.no_grad():
        module.pair_output.normal_(std=0.1)
        module.gamma_pair_raw.fill_(0.6)
    values = _inputs()
    output = module(**values)
    output["pair_contributions_raw"].sum().backward()
    assert values["unary_public_pi"].grad is not None
    assert float(values["unary_public_pi"].grad.abs().sum()) > 0.0
    assert values["reliability"].grad is None
    assert torch.allclose(output["pair_raw_sum"], output["pair_contributions_raw"].sum(dim=-1), atol=1e-6)
    for slot in (0, 7, 19):
        involved = (module.pair_indices == slot).any(dim=-1)
        assert torch.allclose(output["incident_raw_by_slot"][..., slot], output["pair_contributions_raw"][..., involved].sum(dim=-1), atol=1e-6)
        assert torch.allclose(
            module.delete_slot_from_pair_sum(output["pair_raw_sum"], output["incident_raw_by_slot"], slot),
            output["pair_contributions_raw"][..., ~involved].sum(dim=-1),
            atol=1e-6,
        )
    global_logits = torch.randn(2, 4)
    rebuilt = module.reconstruct_with_pair(global_logits, output["pair_postgamma_sum"])
    assert rebuilt.dtype == torch.float32
    assert float((rebuilt - global_logits.float() - output["pair_postgamma_sum"].float()).abs().max()) < 1e-6


def test_owner_isolated_combined_real_asl_and_evidenceconditional_bootstrap_detaches_every_external() -> None:
    action = _module(targets=4)
    reason = _module(targets=21)
    action_owner = _owner_inputs(_inputs(targets=4), targets=4)
    reason_owner = _owner_inputs(_inputs(targets=21), targets=21)
    action_output = action.owner_isolated_auxiliary(**action_owner)
    reason_output = reason.owner_isolated_auxiliary(**reason_owner)
    assert action_output["action_pair_auxiliary_delta"].shape == (2, 4)
    assert reason_output["reason_pair_auxiliary_delta"].shape == (2, 21)
    assert "pair_contributions_raw" not in action_output
    assert "pair_contributions_raw" not in reason_output
    action_targets = torch.randint(0, 2, (2, 4), dtype=torch.float32)
    y_tilde = torch.randint(0, 2, (2, 21), dtype=torch.float32)
    w_pos = torch.ones_like(y_tilde)
    w_neg = torch.full_like(y_tilde, 0.35)
    total = _asl(action_output["action_auxiliary_logits"], action_targets) + _evidence_conditional(
        reason_output["reason_auxiliary_logits"], y_tilde, w_pos, w_neg
    )
    total.backward()
    for module in (action, reason):
        assert module.pair_output.grad is not None
        assert float(module.pair_output.grad.abs().sum()) > 0.0
    _assert_detached_externals(action_owner)
    _assert_detached_externals(reason_owner)


def _three_updates(*, targets: int) -> None:
    module = _module(targets=targets)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.05)
    owner = _owner_inputs(_inputs(targets=targets), targets=targets)
    isolated = module.owner_isolated_auxiliary(**owner)
    if targets == 4:
        loss = _asl(isolated["action_auxiliary_logits"], torch.randint(0, 2, (2, 4), dtype=torch.float32))
    else:
        y_tilde = torch.randint(0, 2, (2, 21), dtype=torch.float32)
        loss = _evidence_conditional(
            isolated["reason_auxiliary_logits"], y_tilde, torch.ones_like(y_tilde), torch.full_like(y_tilde, 0.4)
        )
    loss.backward()
    assert module.pair_output.grad is not None and float(module.pair_output.grad.abs().sum()) > 0.0
    _assert_detached_externals(owner)
    _step(optimizer)
    assert float(module.pair_output.abs().sum()) > 0.0

    update_one = module(**_inputs(targets=targets))
    update_one["pair_contributions"].sum().backward()
    assert module.gamma_pair_raw.grad is not None and float(module.gamma_pair_raw.grad.abs().sum()) > 0.0
    _step(optimizer)

    values = _inputs(targets=targets)
    module(**values)["pair_contributions"].sum().backward()
    for value in (module.Wj.weight, module.Wl.weight, module.Wr.weight, module.Wq.weight, values["target_tokens"], values["evidence_tokens"], values["slot_masks"], values["sector_probs"], values["unary_public_pi"]):
        assert value.grad is not None and float(value.grad.abs().sum()) > 0.0
    assert values["reliability"].grad is None


def test_action_and_reason_each_complete_real_three_update_bootstrap() -> None:
    _three_updates(targets=4)
    _three_updates(targets=21)


def test_pairwise_raw_responds_to_geometry_pi_and_detached_rho_with_zero_pair_weights() -> None:
    module = _module()
    with torch.no_grad():
        module.pair_output.fill_(0.1)
    values = _inputs()
    baseline = module(**values)["pair_contributions_raw"].detach()
    for name, transform in (("slot_masks", lambda value: value.detach().flip(-1)), ("sector_probs", lambda value: value.detach().roll(1, dims=-1)), ("unary_public_pi", lambda value: value.detach().roll(1, dims=-1)), ("reliability", lambda value: value.detach() * 0.5)):
        changed = {key: value.detach().clone() for key, value in values.items()}
        changed[name] = transform(changed[name])
        assert not torch.allclose(baseline, module(**changed)["pair_contributions_raw"].detach()), name
    zero_pi = {key: value.detach().clone() for key, value in values.items()}
    zero_pi["unary_public_pi"][..., 0] = 0.0
    involving_zero = (module.pair_indices == 0).any(dim=-1)
    assert torch.allclose(module(**zero_pi)["pair_contributions_raw"][..., involving_zero], torch.zeros_like(module(**zero_pi)["pair_contributions_raw"][..., involving_zero]))
    zero_rho = {key: value.detach().clone() for key, value in values.items()}
    zero_rho["reliability"][:, 1] = 0.0
    involving_one = (module.pair_indices == 1).any(dim=-1)
    assert torch.allclose(module(**zero_rho)["pair_contributions_raw"][..., involving_one], torch.zeros_like(module(**zero_rho)["pair_contributions_raw"][..., involving_one]))


@pytest.mark.parametrize(
    ("field", "mutate", "code"),
    [
        ("unary_public_pi", lambda value: value.mul(0.0).add_(-0.1), "E_P10_PI_RANGE"),
        ("unary_public_pi", lambda value: value.mul(0.0).add_(1.1), "E_P10_PI_RANGE"),
        ("unary_public_pi", lambda value: value.mul(0.0).add_(0.9), "E_P10_PI_PUBLIC_MASS"),
        ("reliability", lambda value: value.mul(0.0).add_(1.1), "E_P10_RHO_RANGE"),
    ],
)
def test_validate_values_is_explicit_debug_path_and_rejects_invalid_pi_or_rho(field: str, mutate, code: str) -> None:
    module = _module()
    values = _inputs()
    values[field] = mutate(values[field].detach().clone())
    with pytest.raises(ValueError, match=code):
        module.validate_values(**values)


def test_validate_values_rejects_nonfinite_and_extreme_overflow_while_production_never_runs_value_sync() -> None:
    module = _module()
    values = _inputs()
    values["target_tokens"] = torch.full_like(values["target_tokens"], float("inf"))
    with pytest.raises(ValueError, match="E_P10_NONFINITE_target_tokens"):
        module.validate_values(**values)
    values = _inputs()
    values["target_tokens"] = torch.full_like(values["target_tokens"], 1e38)
    with pytest.raises(ValueError, match="E_P10_MAGNITUDE_target_tokens"):
        module.validate_values(**values)
    assert "bool(torch." not in inspect.getsource(RAELPairwiseContribution.forward)


def test_noncontiguous_and_fullgraph_compile_formal_forward() -> None:
    module = _module()
    values = _inputs(noncontiguous=True)
    assert not values["target_tokens"].is_contiguous()
    assert torch.isfinite(module(**values)["pair_raw_sum"]).all()
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    compiled = torch.compile(_module(), backend="eager", fullgraph=True)
    output = compiled(**_inputs(noncontiguous=True))
    assert output["pair_postgamma_sum"].shape == (2, 4)


def test_pairwise_cuda_bf16_k4_k21_three_step_and_detached_diagnostics() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the bf16 probe")
    archive: list[dict[str, torch.Tensor]] = []
    for targets in (4, 21):
        device = torch.device("cuda")
        module = _module(targets=targets, device=device, dtype=torch.bfloat16)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.05)
        owner_values = {key: value.to(device=device, dtype=torch.bfloat16) for key, value in _inputs(targets=targets, noncontiguous=True).items()}
        owner = _owner_inputs(owner_values, targets=targets)
        owner["global_logits"] = owner["global_logits"].to(device=device, dtype=torch.bfloat16)
        isolated = module.owner_isolated_auxiliary(**owner)
        if targets == 4:
            loss = _asl(isolated["action_auxiliary_logits"], torch.randint(0, 2, (2, 4), device=device, dtype=torch.float32))
        else:
            y_tilde = torch.randint(0, 2, (2, 21), device=device, dtype=torch.float32)
            loss = _evidence_conditional(isolated["reason_auxiliary_logits"], y_tilde, torch.ones_like(y_tilde), torch.full_like(y_tilde, 0.4))
        loss.backward(); _step(optimizer)
        output = module(**{key: value.to(device=device, dtype=torch.bfloat16) for key, value in _inputs(targets=targets, noncontiguous=True).items()})
        output["pair_contributions"].float().sum().backward(); _step(optimizer)
        archive.append(output["diagnostics"])
    assert all(value.grad_fn is None for item in archive for value in item.values())
