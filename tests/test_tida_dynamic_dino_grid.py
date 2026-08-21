import torch

from fate_oia.models.acpr_dino_field import ACPRDinoFieldExtractor


def test_mock_dino_dynamic_grid_shapes():
    model = ACPRDinoFieldExtractor(use_mock_dino=True, mock_dim=16)
    target = model.forward_at_resolution(torch.randn(2, 3, 360, 640), expected_hw=(360, 640))
    context = model.forward_at_resolution(torch.randn(2, 3, 192, 344), expected_hw=(192, 344))
    assert target["patch_tokens_by_layer"].shape == (2, 3, 3600, 16)
    assert context["patch_tokens_by_layer"].shape == (2, 3, 1032, 16)
    assert context["grid_hw"] == (24, 43)
