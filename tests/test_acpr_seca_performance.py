import time
import torch

from fate_oia.models.acpr_semantic_evidence_coattention import ACPRSparseEvidenceCoAttention


def test_seca_module_mock_performance_bound():
    m = ACPRSparseEvidenceCoAttention(dim=64, num_heads=4)
    action = torch.randn(2, 4, 64)
    reason = torch.randn(2, 21, 64)
    start = time.time()
    for _ in range(3):
        m(action, reason)
    assert time.time() - start < 5.0
