import torch
from fate_oia.models.diva_deformable_attention import MultiScaleDeformableSampler

def test_deformable_sampler_uses_grid_sample_shapes():
    sampler = MultiScaleDeformableSampler(dim=32, num_scales=3, num_points=4)
    q = torch.randn(2,6,32)
    feats = [torch.randn(2,32,45,80), torch.randn(2,32,23,40), torch.randn(2,32,12,20)]
    refs = torch.rand(2,6,2)
    out = sampler(q, feats, refs)
    assert out['context'].shape == (2,6,32)
    assert out['sample_points'].shape == (2,6,3,4,2)
    assert out['sample_weights'].shape == (2,6,3,4)
    assert torch.isfinite(out['context']).all()
