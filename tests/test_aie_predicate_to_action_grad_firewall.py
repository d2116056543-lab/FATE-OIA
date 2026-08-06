from fate_oia.engine.audit_aie_oia_implementation import gradient_probe
from fate_oia.models.aie_oia_model import AIEOIAModel
import torch


def test_predicate_loss_has_zero_action_evidence_grad():
    model = AIEOIAModel(dim=32, mock_dim=32, use_mock_dino=True)
    assert gradient_probe(model, torch.randn(1, 3, 360, 640))["predicate_only"]["action_evidence"] == 0

