import torch
from torch import nn
import pytest

from fate_oia.engine.train_pact_oia_probe import load_config, pareto_controller_step
from fate_oia.models.pact_oia_model import PACTOIAModel
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA BF16 autocast")
def test_pareto_step_accepts_bf16_training_outputs_outside_autocast():
    device = torch.device("cuda")
    model = PACTOIAModel(use_mock_dino=True).to(device)
    controller = PACTParetoController(license_ema=0.0).to(device)
    image = torch.randn(2, 3, 360, 640, device=device)
    action = torch.tensor([[1.0, 0, 1, 0], [0.0, 1, 0, 1]], device=device)
    reason = torch.stack((torch.arange(21) % 2, (torch.arange(21) + 1) % 2)).float().to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(image, semantic_share_license=0.0, action_scale=1.0, reason_budget=0.6)
    result = pareto_controller_step(
        model,
        controller,
        output,
        {"action": action, "reason": reason},
        {"image": image.cpu(), "action": action.cpu()},
        load_config("configs/fate_oia_train_360x640_pact_oia_v1_probe.yaml"),
        device,
    )
    assert result["selected_lambda"] in (0.0, 0.25, 0.5, 0.75)
    assert torch.isfinite(torch.tensor(result["action_semantic_grad_cosine"]))
