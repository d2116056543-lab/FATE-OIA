import torch

from fate_oia.models.acpr_oia_model import ACPROIAModel


def test_acpr_model_forward_contract():
    m = ACPROIAModel(use_mock_dino=True)
    out = m(torch.randn(1, 3, 360, 640))
    assert out["action_logits_final_raw"].shape == (1, 4)
    assert out["reason_logits_final_raw"].shape == (1, 21)
    assert out["logits_final_raw"].shape == (1, 25)
    assert out["predicate_logits"].shape[1] >= 32
    assert out["action_set_logits"].shape == (1, 16)
    assert torch.allclose(out["action_logits_final_raw"], out["action_logits_direct"])
