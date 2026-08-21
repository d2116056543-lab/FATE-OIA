import torch

from fate_oia.models.acpr_dino_field import ACPRDinoFieldExtractor


def test_dynamic_dino_outputs_detached_and_backbone_frozen():
    model = ACPRDinoFieldExtractor(use_mock_dino=True, mock_dim=8)
    out = model.forward_at_resolution(torch.randn(1, 3, 192, 344, requires_grad=True), expected_hw=(192, 344))
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert not out["patch_tokens_by_layer"].requires_grad
