import torch

from fate_oia.engine.eval_save_oia import evaluate_save_oia
from fate_oia.models.save_oia_model import SAVEOIAModel


def test_eval_reuses_one_encoded_field_for_fixed_audit(monkeypatch):
    model = SAVEOIAModel(use_mock_dino=True)
    calls = {"dino": 0}
    original = model.encode_images
    def wrapped(images):
        calls["dino"] += 1
        return original(images)
    monkeypatch.setattr(model, "encode_images", wrapped)
    loader = [{
        "image": torch.randn(2, 3, 360, 640),
        "action": torch.zeros(2, 4), "reason": torch.zeros(2, 21),
        "file_name": ["a.jpg", "b.jpg"],
    }]
    result = evaluate_save_oia(model, loader, device="cpu", fixed_audit_size=2)
    assert calls["dino"] == 1
    assert result["dino_calls"] == result["ordinary_batches"] == 1
    assert len(result["fixed_audit"]) == 2
    assert "selected_deletion" in result["branch_metrics"]
    row = result["fixed_audit"][0]
    assert {"target_factor", "selected_target_delta", "matched_control_target_delta", "final_target_margin", "evidence_only_target_margin"} <= row.keys()
