import torch

from fate_oia.models.token_provenance import identity_provenance, keep_merge_tokens, recover_attribution


def test_keep_merge_provenance_rows_sum_and_recover():
    x = torch.randn(2, 10, 4)
    reduced, prov, stats = keep_merge_tokens(x, keep_ratio=0.5, num_summary_tokens=2, min_tokens=3)
    assert reduced.shape[0] == 2
    assert prov.shape[:2] == (2, 10)
    assert torch.allclose(prov.sum(-1), torch.ones(2, 10), atol=1e-5)
    attr = torch.randn(2, reduced.shape[1])
    rec = recover_attribution(attr, prov)
    assert rec.shape == (2, 10)


def test_identity_provenance():
    p = identity_provenance(2, 5)
    assert p.shape == (2, 5, 5)
    assert torch.allclose(p.sum(-1), torch.ones(2, 5))
