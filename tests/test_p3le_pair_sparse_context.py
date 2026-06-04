import torch

from fate_oia.models.p3le_pair_head import PairAwareTensorHead
from fate_oia.models.p3le_pair_sparse_context import PairSparseContext


def test_sparse_context_changes_pair_tensor_path():
    torch.manual_seed(3)
    context = PairSparseContext(dim=16, topk=4)
    head = PairAwareTensorHead(dim=16, action_dim=4, reason_dim=21, rank=8)
    action_tokens = torch.randn(2, 4, 16)
    reason_tokens = torch.randn(2, 21, 16)
    tokens = torch.randn(2, 32, 16)
    shared = torch.randn(2, 16)

    sparse = context(action_tokens, reason_tokens, tokens)
    plain = head(action_tokens, reason_tokens, shared)["pair_tensor"]
    enriched = head(
        action_tokens,
        reason_tokens,
        shared,
        action_sparse_context=sparse["action_sparse_context"],
        reason_sparse_context=sparse["reason_sparse_context"],
    )["pair_tensor"]

    assert sparse["action_sparse_context"].shape == action_tokens.shape
    assert sparse["reason_sparse_context"].shape == reason_tokens.shape
    assert not torch.allclose(plain, enriched)

