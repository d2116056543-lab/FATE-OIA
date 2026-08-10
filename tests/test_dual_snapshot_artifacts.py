import json

import torch

from fate_oia.engine.train_aie_oia import save_calibration_artifacts


def test_save_calibration_artifacts_reuses_collected_logits(tmp_path):
    calibration = {
        "action_final": torch.randn(3, 4),
        "reason_final": torch.randn(3, 21),
        "action_target": torch.randint(0, 2, (3, 4)).float(),
        "reason_target": torch.randint(0, 2, (3, 21)).float(),
    }
    names = ["a.jpg", "b.jpg", "c.jpg"]

    save_calibration_artifacts(tmp_path, calibration, names)

    payload = torch.load(tmp_path / "train_calib_logits.pt", weights_only=True)
    assert torch.equal(payload["action_logits"], calibration["action_final"])
    assert torch.equal(payload["reason_logits"], calibration["reason_final"])
    assert torch.equal(payload["action_labels"], calibration["action_target"])
    assert torch.equal(payload["reason_labels"], calibration["reason_target"])
    assert json.loads((tmp_path / "file_names_train_calib.json").read_text()) == names
