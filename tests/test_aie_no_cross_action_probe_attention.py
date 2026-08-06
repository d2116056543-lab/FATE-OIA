import torch

from fate_oia.models.aie_evidence_interface import AIEEvidenceInterface


def test_group_attention_does_not_mix_actions():
    module = AIEEvidenceInterface(dim=32, num_predicates=32, grid_hw=(4, 5), local_points_per_layer=2).eval()
    action = torch.randn(1, 4, 32)
    field = torch.randn(1, 3, 20, 32)
    pattn = torch.softmax(torch.randn(1, 32, 20), -1)
    pprob = torch.sigmoid(torch.randn(1, 32))
    with torch.no_grad():
        base = module(action, field, pattn, pprob)["evidence_token"]
        changed = action.clone(); changed[:, 0, 0] += 10
        altered = module(changed, field, pattn, pprob)["evidence_token"]
    torch.testing.assert_close(base[:, 1:], altered[:, 1:], atol=1e-6, rtol=0)
    assert not torch.allclose(base[:, 0], altered[:, 0])

