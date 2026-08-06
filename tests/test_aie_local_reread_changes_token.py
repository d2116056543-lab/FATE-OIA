import torch

from fate_oia.models.aie_evidence_interface import AIEEvidenceInterface


def test_local_reread_changes_formal_evidence_token():
    module = AIEEvidenceInterface(dim=32, grid_hw=(4, 5), local_points_per_layer=2).eval()
    args = (torch.randn(1, 4, 32), torch.randn(1, 3, 20, 32), torch.softmax(torch.randn(1, 32, 20), -1), torch.rand(1, 32))
    with torch.no_grad():
        local = module(*args, local_reread_enabled=True)["evidence_token"]
        global_only = module(*args, local_reread_enabled=False)["evidence_token"]
    assert not torch.allclose(local, global_only)

