import torch

from fate_oia.models.acpr_dino_field import ACPRDinoFieldExtractor


def test_acpr_mock_dino_field_contract():
    model = ACPRDinoFieldExtractor(use_mock_dino=True)
    out = model(torch.randn(2, 3, 360, 640))
    assert out["patch_tokens_by_layer"].shape == (2, 3, 3600, 384)
    assert out["cls_tokens_by_layer"].shape == (2, 3, 384)
    assert out["grid_hw"] == (45, 80)
    assert out["original_tokens"] == 3601
    assert all(not p.requires_grad for p in model.backbone.parameters())
