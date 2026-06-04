import torch

from fate_oia.models.ceai_pair_sparse_attention import TaskGuidedPairSparseAttention


def test_pair_sparse_attention_shapes_and_perturbation():
    mod = TaskGuidedPairSparseAttention(dim=32, action_dim=4, reason_group_count=6, topk=5, heads=4)
    action = torch.randn(2, 4, 32)
    reason_groups = torch.randn(2, 6, 32)
    scene = torch.randn(2, 3, 32)
    visual = torch.randn(2, 19, 32)
    out1 = mod(action, reason_groups, scene, visual)
    assert out1["pair_group_context"].shape == (2, 4, 6, 32)
    assert out1["attention_indices"].shape[-1] == 5
    assert out1["attention_entropy"].numel() > 0
    visual2 = visual.clone()
    visual2[:, :5] += 2.5
    out2 = mod(action, reason_groups, scene, visual2)
    assert not torch.allclose(out1["pair_group_context"], out2["pair_group_context"])
