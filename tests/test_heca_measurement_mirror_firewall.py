import torch

from fate_oia.losses.meter_grounding_losses import meter_grounding_loss
from fate_oia.models.meter_oia_model import METEROIAModel


def test_measurement_mirror_loss_cannot_update_foundation_task_branches() -> None:
    """Mirror grounding may train typed factors, never the visual foundation."""
    model = METEROIAModel(dim=384, use_mock_dino=True)
    pair = model.forward_view_pair(
        torch.randn(1, 3, 360, 640),
        torch.randn(1, 3, 360, 640),
        progress=0.5,
    )
    target = {
        "factor_anchor_map": torch.zeros(1, 21, 45, 80),
        "factor_anchor_valid": torch.zeros(1, 21, dtype=torch.bool),
        "factor_state_target": torch.full((1, 21), -1, dtype=torch.long),
        "factor_state_valid": torch.zeros(1, 21, dtype=torch.bool),
        "factor_present_valid": torch.zeros(1, 21, dtype=torch.bool),
        "factor_absent_valid": torch.zeros(1, 21, dtype=torch.bool),
        "factor_observability": torch.zeros(1, 21),
        "factor_observability_valid": torch.zeros(1, 21, dtype=torch.bool),
        "factor_source_weight": torch.zeros(1, 21),
    }
    loss = meter_grounding_loss(
        pair["original"],
        target,
        observability_tau=torch.full((21,), 0.5),
        mirrored_output=pair["view"],
        weights={
            "anchor": 0.0,
            "state": 0.0,
            "null": 0.0,
            "observability": 0.0,
            "discrimination": 0.0,
            "mirror": 1.0,
            "ontology_identity": 0.0,
        },
    )["total"]
    loss.backward()

    assert all(
        parameter.grad is None or parameter.grad.count_nonzero() == 0
        for parameter in model.foundation.parameters()
    )
