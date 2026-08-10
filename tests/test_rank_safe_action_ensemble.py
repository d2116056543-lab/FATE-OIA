import torch

from fate_oia.engine.eval_rank_safe_action_ensemble import _blend, fit_ensemble


def _fixture():
    generator = torch.Generator().manual_seed(17)
    target = (torch.rand(240, 4, generator=generator) > torch.tensor([0.45, 0.55, 0.70, 0.72])).float()
    signal = (target * 2 - 1) + 0.5 * torch.randn(240, 4, generator=generator)
    pact = signal + torch.tensor([0.1, -0.2, 0.4, -0.1]) * torch.randn(240, 4, generator=generator)
    aie = signal + torch.tensor([-0.1, 0.3, -0.2, 0.4]) * torch.randn(240, 4, generator=generator)
    names = [f"sample_{index:04d}.jpg" for index in range(240)]
    return pact, aie, target, names


def test_fit_uses_only_calibration_inputs():
    pact, aie, target, names = _fixture()
    first, _ = fit_ensemble(pact, aie, target, names)
    # Unrelated tensors stand in for arbitrary test-set changes; the fit API cannot consume them.
    _test_logits = torch.randn(99, 4)
    _test_targets = torch.randint(0, 2, (99, 4)).float()
    second, _ = fit_ensemble(pact, aie, target, names)
    assert first == second


def test_fit_returns_bounded_weights_and_thresholds():
    pact, aie, target, names = _fixture()
    fit, diagnostics = fit_ensemble(pact, aie, target, names)
    assert fit.family in {"global", "per_action_shrunk"}
    assert len(fit.weights) == len(fit.thresholds) == 4
    assert all(0.0 <= value <= 1.0 for value in fit.weights)
    assert all(0.0 < value < 1.0 for value in fit.thresholds)
    assert len(diagnostics["fold_ids"]) == len(names)


def test_blend_endpoints_are_exact():
    pact, aie, _, _ = _fixture()
    assert torch.equal(_blend(pact, aie, 1.0), pact.float())
    assert torch.equal(_blend(pact, aie, 0.0), aie.float())
