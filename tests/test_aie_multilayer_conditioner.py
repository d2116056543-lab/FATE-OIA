import torch

from fate_oia.models.aie_evidence_interface import AIEEvidenceInterface


def test_each_dino_layer_has_independent_projection_and_rmsnorm():
    module = AIEEvidenceInterface(dim=32, grid_hw=(4, 5), local_points_per_layer=2)
    assert len(module.layer_projections) == len(module.layer_norms) == 3
    conditioned = module._condition(torch.randn(2, 3, 20, 32))
    assert conditioned.shape == (2, 3, 20, 32)
    assert not torch.allclose(conditioned[:, 0], conditioned[:, 1])

