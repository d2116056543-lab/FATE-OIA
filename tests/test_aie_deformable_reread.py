import torch

from fate_oia.models.aie_deformable_reread import AIEDeformableReread


def test_deformable_reread_uses_bounded_grid_sampling():
    module = AIEDeformableReread(dim=16, grid_hw=(4, 5), num_layers=3, points_per_layer=4, max_offset=0.25)
    out = module(torch.randn(2, 4, 4, 16), torch.randn(2, 3, 20, 16), torch.softmax(torch.randn(2, 4, 4, 20), -1))
    assert out["local_token"].shape == (2, 4, 4, 16)
    assert float(out["sampling_offsets"].abs().max()) <= 0.25 + 1e-7
    torch.testing.assert_close(out["sampling_weights"].sum((-1, -2)), torch.ones(2, 4, 4), atol=1e-6, rtol=0)


