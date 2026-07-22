from pathlib import Path

import pytest
import torch

from fate_oia.losses.precise_intervention_losses import packed_target_specific_interventions
from fate_oia.losses.precise_losses import total_precise_losses
from fate_oia.models.precise_oia_model import PRECISEOIAModel


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA dynamic contract requires a GPU")
def test_cuda_complete_mechanism_path_is_finite_and_backpropagates():
    device = torch.device("cuda")
    model = PRECISEOIAModel(ROOT / "configs", use_mock_dino=True).to(device)
    image = torch.randn(1, 3, 360, 640, device=device)
    action = torch.tensor([[1.0, 0.0, 0.0, 1.0]], device=device)
    reason = torch.zeros(1, 21, device=device)
    reason[0, 3] = 1.0
    output = model(image)
    targets = {
        "presence": torch.ones(1, 10, device=device),
        "presence_valid": torch.ones(1, 10, device=device),
        "observability": torch.ones(1, 10, device=device),
        "state": torch.zeros(1, 10, 4, device=device),
        "state_valid": torch.zeros(1, 10, device=device),
        "part_coordinates": output["evidence_part_coordinates"].detach(),
        "part_scales": output["evidence_part_scales"].detach(),
        "soft_masks": output["evidence_masks"].detach(),
        "part_valid": torch.ones(1, 10, device=device),
    }
    losses = total_precise_losses(output, action, reason, targets)
    intervention = packed_target_specific_interventions(model, output, action, reason, max_pairs=4)
    total = losses["loss_total"] + intervention["loss_intervention"]
    assert torch.isfinite(total)
    total.backward()
    assert model.exchange.action_query.weight.grad is not None
    assert torch.isfinite(model.exchange.action_query.weight.grad).all()
    assert output["diagnostics"]["dino_call_count"] == 1
    assert output["evidence_masks"].shape[-2:] == (45, 80)
