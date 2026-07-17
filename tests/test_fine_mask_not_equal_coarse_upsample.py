import torch
import torch.nn.functional as F

from fate_oia.models.mosaic_typed_evidence_splat import typed_evidence_splat


def test_fine_mask_is_not_just_coarse_upsample():
    coords = torch.zeros(1, 1, 1, 1, 2, 2)
    coords[..., 0, :] = -0.8
    coords[..., 1, :] = 0.8
    sampled = torch.ones(1, 1, 1, 1, 2, 4)
    attention = torch.ones(1, 1, 1, 1, 2)
    result = typed_evidence_splat(coords, sampled, attention, ["point"], output_hw=(8, 8), coarse_hw=(2, 2))
    coarse_up = F.interpolate(result["coarse_mask"], size=(8, 8), mode="bilinear", align_corners=False)
    assert not torch.allclose(result["fine_mask"], coarse_up)
