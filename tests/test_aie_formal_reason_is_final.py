import torch

from fate_oia.models.aie_oia_model import AIEOIAModel


def test_formal_reason_output_is_private_refined_reason():
    out = AIEOIAModel(dim=32, mock_dim=32, use_mock_dino=True)(torch.randn(1, 3, 360, 640))
    torch.testing.assert_close(out["reason_logits_final"], out["reason_logits_primary"] + out["reason_delta"])
    assert out["branch_logits"]["final_reason"] is out["reason_logits_final"]

