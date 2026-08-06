import torch

from fate_oia.models.aie_deformable_reread import AIEDeformableReread


def test_offsets_stay_bounded_for_extreme_queries():
    module = AIEDeformableReread(dim=16, grid_hw=(4, 5), points_per_layer=2, max_offset=0.25)
    out = module(torch.full((1, 4, 4, 16), 1e4), torch.randn(1, 3, 20, 16), torch.full((1, 4, 4, 20), 0.05))
    assert float(out["sampling_offsets"].abs().max()) <= 0.25

