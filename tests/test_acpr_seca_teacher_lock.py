import torch

from fate_oia.models.acpr_oia_model import ACPROIAModel
from fate_oia.engine.train_acpr_oia import collect_threshold_teacher


class TinyCalibLoader:
    def __iter__(self):
        for _ in range(2):
            yield {
                "image": torch.randn(2, 3, 360, 640),
                "action": torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]),
                "reason": torch.zeros(2, 21),
                "file_name": ["a.jpg", "b.jpg"],
            }


def test_seca_threshold_teacher_uses_train_calib_loader_only():
    model = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, seca_enabled=True)
    teacher = collect_threshold_teacher(model, TinyCalibLoader(), torch.device("cpu"), 0, {"threshold": {"grid_step": 0.5}})
    assert teacher["source"] == "train_calib"
    assert teacher["threshold_prob"].numel() == 25
    assert teacher["pred_rate"].numel() == 25
