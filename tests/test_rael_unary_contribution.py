from __future__ import annotations

import torch

from fate_oia.models.rael_relation_contributions import RAELUnaryContribution


def _independent_entmax_reference(scores: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """D05 oracle: a test-local alpha-entmax solver, independent of production."""
    work_scores = scores.float()
    work_alpha = alpha.float()
    beta = work_alpha - 1.0
    lower = work_scores.amax(dim=-1, keepdim=True) - beta.reciprocal()
    upper = work_scores.amax(dim=-1, keepdim=True)
    for _ in range(96):
        midpoint = (lower + upper) * 0.5
        trial = (beta * (work_scores - midpoint)).clamp_min(0.0).pow(beta.reciprocal())
        lower = torch.where(trial.sum(dim=-1, keepdim=True) > 1.0, midpoint, lower)
        upper = torch.where(trial.sum(dim=-1, keepdim=True) > 1.0, upper, midpoint)
    tau = (lower + upper) * 0.5
    probabilities = (beta * (work_scores - tau)).clamp_min(0.0).pow(beta.reciprocal())
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)


def _inputs(
    *,
    batch: int = 2,
    targets: int = 4,
    dim: int = 16,
    attributes: int = 5,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260726)
    return {
        "target_tokens": torch.randn(batch, targets, dim, device=device, requires_grad=True),
        "evidence_tokens": torch.randn(batch, 20, dim, device=device, requires_grad=True),
        "attributes": torch.randn(batch, 20, attributes, device=device, requires_grad=True),
        "presence": torch.full((batch, 20), 0.65, device=device, requires_grad=True),
        "reliability": torch.full((batch, 20), 0.75, device=device, requires_grad=True),
    }


def _module(*, targets: int = 4, dim: int = 16, attributes: int = 5) -> RAELUnaryContribution:
    torch.manual_seed(13)
    return RAELUnaryContribution(
        num_targets=targets,
        dim=dim,
        attribute_dim=attributes,
        gamma_cap=0.25,
    )


def test_unary_uses_public_twenty_slots_and_null_with_function_preserving_postgamma() -> None:
    module = _module()
    values = _inputs()
    output = module(**values)

    assert output["unary_contributions_raw"].shape == (2, 4, 20)
    assert output["unary_contributions"].shape == (2, 4, 20)
    assert output["pi"].shape == (2, 4, 21)
    assert output["slot_weights"].shape == (2, 4, 21)
    assert output["null_mass"].shape == (2, 4)
    assert output["alpha"].shape == (2, 4)
    assert output["gamma_unary"].shape == (4,)
    assert module.null_evidence.shape == (4, 16)
    assert bool(torch.isfinite(output["unary_contributions_raw"]).all())
    assert bool(torch.isfinite(output["slot_weights"]).all())
    assert torch.allclose(
        output["slot_weights"].sum(dim=-1),
        torch.ones_like(output["null_mass"]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(output["unary_contributions"], torch.zeros_like(output["unary_contributions"]))
    assert float(output["unary_contributions_raw"].abs().sum()) > 0.0
    assert torch.allclose(output["pi"], output["slot_weights"])
    assert torch.allclose(output["null_mass"], output["pi"][..., -1])
    assert module.parameter_owner == "unary_contribution"
    assert module.learning_rate == 2e-4


def test_unary_formula_uses_detached_rho_and_never_exposes_null_as_contribution() -> None:
    module = _module()
    values = _inputs()
    output = module(**values)
    loss = output["unary_contributions_raw"].sum()
    loss.backward()

    assert values["reliability"].grad is None
    assert values["evidence_tokens"].grad is not None
    assert values["target_tokens"].grad is not None
    assert values["attributes"].grad is not None
    assert values["presence"].grad is not None

    zero_reliability = torch.zeros_like(values["reliability"])
    with torch.no_grad():
        module.null_score.fill_(-20.0)
    zero_output = module(
        target_tokens=values["target_tokens"].detach(),
        evidence_tokens=values["evidence_tokens"].detach(),
        attributes=values["attributes"].detach(),
        presence=values["presence"].detach(),
        reliability=zero_reliability,
    )
    assert torch.allclose(
        zero_output["unary_contributions_raw"],
        torch.zeros_like(zero_output["unary_contributions_raw"]),
        atol=1e-7,
    )
    # rho=0 obeys log(rho + eps) routing. It is not a special hard -80 route.
    assert float(zero_output["pi"][..., :20].sum()) > 0.0


def test_unary_raw_contribution_matches_independent_d05_oracle() -> None:
    module = _module()
    values = _inputs()
    output = module(**values)
    query = module.query_proj(values["target_tokens"])
    evidence = module.evidence_proj(values["evidence_tokens"])
    query_expanded = query.unsqueeze(2).expand(-1, -1, 20, -1)
    evidence_expanded = evidence.unsqueeze(1).expand(-1, 4, -1, -1)
    attributes = values["attributes"].unsqueeze(1).expand(-1, 4, -1, -1)
    presence = values["presence"].unsqueeze(1).unsqueeze(-1).expand(-1, 4, -1, -1)
    expected_score = (
        (query_expanded * evidence_expanded).sum(dim=-1) / (module.dim ** 0.5)
        + module.score_bias(torch.cat((query_expanded, attributes), dim=-1)).squeeze(-1)
        + torch.log(values["reliability"].detach().unsqueeze(1) + module.eps)
    )
    expected_null = (
        (
            query.unsqueeze(2)
            * module.null_evidence.view(1, 4, 1, module.dim)
        ).sum(dim=-1)
        / (module.dim ** 0.5)
        + module.null_score.view(1, -1, 1)
    )
    expected_weights = _independent_entmax_reference(
        torch.cat((expected_score, expected_null), dim=-1),
        alpha=module.adaptive_alpha().view(1, 4, 1),
    )
    expected_phi = module.phi(
        torch.cat((query_expanded, evidence_expanded, attributes, presence), dim=-1)
    )
    expected_raw = (
        expected_weights[..., :20]
        * values["reliability"].detach().unsqueeze(1)
        * torch.einsum("bkjd,kd->bkj", expected_phi, module.unary_vector)
    )
    assert torch.allclose(output["pi"], expected_weights, atol=1e-5, rtol=1e-5)
    assert torch.allclose(output["unary_contributions_raw"], expected_raw, atol=1e-6, rtol=1e-6)


def test_unary_supports_twenty_one_reason_targets_and_gamma_is_bounded() -> None:
    module = _module(targets=21)
    values = _inputs(targets=21)
    output = module(**values)
    assert output["unary_contributions_raw"].shape == (2, 21, 20)
    assert output["slot_weights"].shape == (2, 21, 21)
    with torch.no_grad():
        module.gamma_unary_raw.copy_(torch.linspace(-100.0, 100.0, 21))
    gamma = module.bounded_gamma()
    assert bool((gamma >= -0.25).all())
    assert bool((gamma <= 0.25).all())
    assert gamma[0].item() == -0.25
    assert gamma[-1].item() == 0.25


def test_presence_changes_only_phi_contribution_and_never_routing_weights() -> None:
    module = _module()
    values = _inputs()
    baseline = module(**values)
    altered = {key: value.detach().clone() for key, value in values.items()}
    altered["presence"] = torch.full_like(altered["presence"], 0.05)
    changed = module(**altered)
    assert torch.equal(baseline["pi"].detach(), changed["pi"].detach())
    assert not torch.allclose(
        baseline["unary_contributions_raw"].detach(),
        changed["unary_contributions_raw"].detach(),
    )


def test_unary_responds_to_target_evidence_attributes_and_reliability() -> None:
    module = _module()
    values = _inputs()
    baseline = module(**values)["unary_contributions_raw"].detach()
    for name, delta in (
        ("target_tokens", 0.25),
        ("evidence_tokens", 0.25),
        ("attributes", 0.25),
        ("reliability", -0.35),
    ):
        altered = {key: value.detach().clone() for key, value in values.items()}
        altered[name] = altered[name] + delta
        changed = module(**altered)["unary_contributions_raw"].detach()
        assert not torch.allclose(baseline, changed), name


def test_unary_diagnostics_split_positive_negative_raw_and_postgamma_without_graph_retention() -> None:
    module = _module()
    values = _inputs()
    with torch.no_grad():
        module.gamma_unary_raw.fill_(0.8)
    output = module(**values)
    diagnostics = output["diagnostics"]
    for stage, contribution in (
        ("raw", output["unary_contributions_raw"]),
        ("postgamma", output["unary_contributions"]),
    ):
        positive = contribution.detach().float().clamp_min(0.0)
        negative = (-contribution.detach().float()).clamp_min(0.0)
        for sign, values_by_sign in (("positive", positive), ("negative", negative)):
            assert diagnostics[f"{stage}_{sign}_contribution_mean"].grad_fn is None
            assert diagnostics[f"{stage}_{sign}_contribution_rms"].grad_fn is None
            assert diagnostics[f"{stage}_{sign}_contribution_mass"].grad_fn is None
            assert torch.allclose(
                diagnostics[f"{stage}_{sign}_contribution_mean"], values_by_sign.mean(dim=-1)
            )
            assert torch.allclose(
                diagnostics[f"{stage}_{sign}_contribution_rms"], values_by_sign.square().mean(dim=-1).sqrt()
            )
            assert torch.allclose(
                diagnostics[f"{stage}_{sign}_contribution_mass"], values_by_sign.sum(dim=-1)
            )


def test_unary_bootstraps_gamma_then_internal_parameters_on_second_update() -> None:
    module = _module()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.05)
    values = _inputs()

    update_zero = module(**values)
    loss_zero = update_zero["unary_contributions"].sum()
    loss_zero.backward()
    assert module.gamma_unary_raw.grad is not None
    assert float(module.gamma_unary_raw.grad.abs().sum()) > 0.0
    assert module.query_proj.weight.grad is None or float(module.query_proj.weight.grad.abs().sum()) == 0.0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    values_one = _inputs()
    update_one = module(**values_one)
    assert float(update_one["unary_contributions"].abs().sum()) > 0.0
    update_one["unary_contributions"].sum().backward()
    required = (
        module.query_proj.weight,
        module.evidence_proj.weight,
        module.phi[0].weight,
        module.unary_vector,
        module.eta,
        values_one["evidence_tokens"],
    )
    for value in required:
        assert value.grad is not None
        assert float(value.grad.abs().sum()) > 0.0


def test_unary_rejects_background_shape_and_detaches_diagnostics() -> None:
    module = _module()
    values = _inputs()
    output = module(**values)
    for diagnostic in output["diagnostics"].values():
        assert diagnostic.grad_fn is None

    invalid = {key: value.detach() for key, value in values.items()}
    invalid["evidence_tokens"] = torch.randn(2, 21, 16)
    try:
        module(**invalid)
    except ValueError as error:
        assert "20 public evidence slots" in str(error)
    else:
        raise AssertionError("background-inclusive evidence must be rejected")


def test_unary_cuda_bf16_two_step_and_diagnostic_retention() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    module = _module().to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.03)
    archive = []

    for step in range(2):
        values = _inputs(device=device)
        values = {key: value.to(dtype=torch.bfloat16) for key, value in values.items()}
        output = module(**values)
        assert bool(torch.isfinite(output["unary_contributions_raw"]).all())
        assert bool(torch.isfinite(output["slot_weights"]).all())
        archive.append(output["diagnostics"])
        output["unary_contributions"].float().sum().backward()
        if step == 0:
            assert module.gamma_unary_raw.grad is not None
            assert float(module.gamma_unary_raw.grad.abs().sum()) > 0.0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    assert all(value.grad_fn is None for item in archive for value in item.values())

