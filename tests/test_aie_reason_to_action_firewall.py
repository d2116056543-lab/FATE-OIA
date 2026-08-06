from fate_oia.engine.audit_aie_oia_implementation import gradient_probe
from fate_oia.models.aie_oia_model import AIEOIAModel
import torch


def test_reason_loss_has_zero_action_and_primary_grad():
    model = AIEOIAModel(dim=32, mock_dim=32, use_mock_dino=True)
    row = gradient_probe(model, torch.randn(1, 3, 360, 640))["final_reason"]
    assert row["primary"] == row["action_evidence"] == row["action_contribution"] == 0

