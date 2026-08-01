import torch

from fate_oia.models.meter_oia_model import METEROIAModel


def test_progress_zero_preserves_heca_visual_and_reason_anchors() -> None:
    model = METEROIAModel(use_mock_dino=True)
    out = model(torch.randn(1, 3, 360, 640), progress=0)
    torch.testing.assert_close(out["action_logits_final"], out["action_logits_visual"], atol=1e-6, rtol=0)
    torch.testing.assert_close(out["reason_logits_global"], out["reason_logits_calalign"], atol=1e-6, rtol=0)
    torch.testing.assert_close(out["reason_logits_final"], out["reason_logits_calalign"], atol=1e-6, rtol=0)
