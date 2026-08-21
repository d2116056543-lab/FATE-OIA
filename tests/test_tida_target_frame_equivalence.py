import torch

from fate_oia.models.acpr_dino_field import ACPRDinoFieldExtractor


def test_legacy_forward_equals_dynamic_target_forward():
    torch.manual_seed(4)
    model = ACPRDinoFieldExtractor(use_mock_dino=True, mock_dim=8)
    image = torch.randn(1, 3, 360, 640)
    legacy = model(image)
    dynamic = model.forward_at_resolution(image, expected_hw=(360, 640))
    for key in ("patch_tokens_by_layer", "cls_tokens_by_layer"):
        assert torch.equal(legacy[key], dynamic[key])
