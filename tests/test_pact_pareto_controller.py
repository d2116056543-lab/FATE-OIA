import torch
from torch import nn
import pytest

from fate_oia.engine.train_pact_oia_probe import load_config, pareto_controller_step
from fate_oia.models.pact_oia_model import PACTOIAModel
from fate_oia.utils.pact_pareto_controller import PACTParetoController


def test_pareto_selects_semantic_best_inside_action_priority_band_and_restores_state():
    model = nn.Linear(3, 2)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    controller = PACTParetoController(license_ema=0.0, action_best_tolerance=0.00005)
    measurements = {
        0.0: {"action_loss": 1.0, "semantic_loss": 1.0},
        0.25: {"action_loss": 0.99998, "semantic_loss": 0.90},
        0.5: {"action_loss": 1.0002, "semantic_loss": 0.80},
        0.75: {"action_loss": 1.0005, "semantic_loss": 0.70},
    }
    result = controller.evaluate_candidates(lambda value: measurements[value], model)
    assert result["selected_lambda"] == 0.25
    assert float(controller.semantic_share_license) == 0.25
    assert result["candidate_measurements"]["0.25"]["semantic_loss"] == 0.90
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_pareto_prefers_smaller_lambda_when_audit_losses_tie():
    controller = PACTParetoController(license_ema=0.0, action_best_tolerance=0.001)
    result = controller.select({
        0.0: {"action_loss": 1.0, "semantic_loss": 1.0},
        0.25: {"action_loss": 1.0, "semantic_loss": 0.9},
        0.5: {"action_loss": 1.0, "semantic_loss": 0.9},
        0.75: {"action_loss": 1.0, "semantic_loss": 0.9},
    })
    assert result["selected_lambda"] == 0.25


def test_pareto_falls_back_to_zero_without_semantic_improvement():
    controller = PACTParetoController(license_ema=0.0, action_best_tolerance=0.001)
    result = controller.select({
        0.0: {"action_loss": 1.0, "semantic_loss": 1.0},
        0.25: {"action_loss": 0.9999, "semantic_loss": 1.01},
        0.5: {"action_loss": 0.9998, "semantic_loss": 1.02},
        0.75: {"action_loss": 0.9997, "semantic_loss": 1.03},
    })
    assert result["selected_lambda"] == 0.0


def test_pareto_resume_preserves_license_and_rotation():
    first = PACTParetoController(license_ema=0.5)
    first.select({value: {"action_loss": 1.0, "semantic_loss": 1.0 - value}
                  for value in (0.0, 0.25, 0.5, 0.75)})
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
        {"image": image.cpu(), "action": action.cpu(), "reason": reason.cpu()},
        load_config("configs/fate_oia_train_360x640_pact_oia_v1_probe.yaml"),
        device,
    )
    assert result["selected_lambda"] in (0.0, 0.25, 0.5, 0.75)
    assert torch.isfinite(torch.tensor(result["action_semantic_grad_cosine"]))
