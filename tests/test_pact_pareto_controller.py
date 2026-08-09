import torch
from torch import nn

from fate_oia.utils.pact_pareto_controller import PACTParetoController


def test_pareto_selects_largest_safe_candidate_and_restores_state():
    model = nn.Linear(3, 2)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    controller = PACTParetoController(license_ema=0.0)
    result = controller.evaluate_candidates(lambda value: {0.0: 1.0, 0.25: 1.0005, 0.5: 1.002, 0.75: 1.1}[value], model)
    assert result["selected_lambda"] == 0.25
    assert float(controller.semantic_share_license) == 0.25
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_pareto_resume_preserves_license_and_rotation():
    first = PACTParetoController(license_ema=0.5)
    first.select({0.0: 1.0, 0.25: 1.0, 0.5: 1.0, 0.75: 1.0})
    second = PACTParetoController(license_ema=0.5)
    second.load_state_dict(first.state_dict())
    assert torch.equal(first.semantic_share_license, second.semantic_share_license)
    assert torch.equal(first.audit_rotation, second.audit_rotation)
