import torch

from fate_oia.engine.evaluate_tida_oia import dynamic_slice_metrics


def test_dynamic_slices_report_image_to_video_delta():
    rows = {
        "image_action": torch.zeros(4, 4), "video_action": torch.ones(4, 4),
        "image_reason": torch.zeros(4, 21), "video_reason": torch.ones(4, 21),
        "action_target": torch.ones(4, 4), "reason_target": torch.ones(4, 21),
        "rho": torch.tensor([[0.0] * 36, [0.05] * 36, [0.5] * 36, [0.8] * 36]),
    }
    metrics = dynamic_slice_metrics(rows)
    assert metrics["low_dynamic"]["count"] == 2
    assert metrics["high_dynamic"]["count"] == 2
    assert "action_mf1_delta" in metrics["high_dynamic"]
