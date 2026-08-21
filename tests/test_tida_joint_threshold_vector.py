import torch

from fate_oia.utils.aie_metrics import aie_branch_metrics


def test_joint_threshold_vector_is_split_between_action_and_reason() -> None:
    action_logits = torch.tensor([[2.0, -2.0, 1.0, -1.0], [-1.0, 1.0, -2.0, 2.0]])
    reason_logits = torch.stack((torch.full((21,), 2.0), torch.full((21,), -2.0)))
    thresholds = torch.cat((torch.full((4,), 0.4), torch.full((21,), 0.6)))
    action_targets = (action_logits.sigmoid() >= thresholds[:4]).float()
    reason_targets = (reason_logits.sigmoid() >= thresholds[4:]).float()

    metrics = aie_branch_metrics(
        action_logits,
        reason_logits,
        action_targets,
        reason_targets,
        threshold=thresholds,
    )

    assert metrics["Act_mF1"] == 1.0
    assert metrics["Exp_mF1"] == 1.0
