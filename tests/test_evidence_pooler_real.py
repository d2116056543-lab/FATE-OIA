import torch

from fate_oia.models.evidence.evidence_pooler import EvidenceTokenPooler, map_image_xy_to_patch_index, masked_average_pool


def test_map_image_xy_to_patch_index_uses_grid_metadata():
    meta = {"image_width": 640, "image_height": 360, "patch_grid_w": 80, "patch_grid_h": 45}
    idx = map_image_xy_to_patch_index(639, 359, meta, patch_size=8)
    assert idx == 45 * 80 - 1


def test_masked_average_pool_shape():
    tokens = torch.randn(2, 6, 4)
    masks = torch.zeros(2, 3, 6)
    masks[:, :, :2] = 1
    pooled = masked_average_pool(tokens, masks)
    assert pooled.shape == (2, 3, 4)


def test_evidence_token_pooler_forward_schema():
    pooler = EvidenceTokenPooler(dim=4)
    tokens = torch.randn(2, 6, 4)
    masks = torch.zeros(2, 3, 6)
    masks[:, 0, :2] = 1
    cats = torch.zeros(2, 3, dtype=torch.long)
    geom = torch.zeros(2, 3, 9)
    out = pooler(tokens, masks, cats, geom)
    assert out["evidence_tokens"].shape == (2, 3, 4)
    assert out["valid_mask"].shape == (2, 3)
