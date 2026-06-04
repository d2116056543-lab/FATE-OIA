import torch
from torch import nn

from fate_oia.models.ceai_expert_adapter import ActionExpert, ExpertAdapterBlock, ReasonExpert, TailExpert


def test_expert_adapter_uses_attention_and_shapes():
    block = ExpertAdapterBlock(dim=32, num_heads=4)
    assert any(isinstance(m, nn.MultiheadAttention) for m in block.modules())
    action = ActionExpert(dim=32, action_dim=4, depth=2, heads=4)
    reason = ReasonExpert(dim=32, reason_dim=21, depth=2, heads=4)
    tail = TailExpert(dim=32, reason_dim=21, tail_indices=[5, 6, 9], depth=1, heads=4)
    action_tokens = torch.randn(2, 4, 32)
    reason_tokens = torch.randn(2, 21, 32)
    context = torch.randn(2, 7, 32)
    a_out = action(action_tokens, context)
    r_out = reason(reason_tokens, context)
    t_out = tail(reason_tokens, context)
    assert a_out["tokens"].shape == (2, 4, 32)
    assert a_out["logits"].shape == (2, 4)
    assert r_out["tokens"].shape == (2, 21, 32)
    assert r_out["logits"].shape == (2, 21)
    assert t_out["tail_delta"].shape == (2, 21)
    assert t_out["tail_delta"][:, 0].abs().max() == 0
    (a_out["logits"].sum() + r_out["logits"].sum() + t_out["tail_delta"].sum()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in action.parameters())
