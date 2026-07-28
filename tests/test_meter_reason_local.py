from types import SimpleNamespace

import torch
from torch import nn

from fate_oia.losses.meter_reason_losses import meter_reason_loss
from fate_oia.models.meter_reason_decoder import METERPrivateReasonDecoder
from test_meter_reason_global import test_global_reason_view_is_directly_computed


def test_local_reason_view_shares_the_formal_decoder_path() -> None:
    # The global test exercises the same complete forward contract; this
    # named test makes the local-view requirement explicit for the audit.
    test_global_reason_view_is_directly_computed()


def _foundation(dim: int) -> SimpleNamespace:
    trunk = SimpleNamespace(
        label_queries=nn.Parameter(torch.randn(25, dim)),
        query_proj=nn.Linear(dim, dim),
        key_proj=nn.Linear(dim, dim),
        value_proj=nn.Linear(dim, dim),
        label_self_attn=nn.MultiheadAttention(dim, num_heads=4, batch_first=True),
        logit_head=nn.Linear(dim, 1),
    )
    return SimpleNamespace(trunk=trunk, action_dim=4)


def test_local_view_is_initialized_from_foundation() -> None:
    torch.manual_seed(43)
    decoder = METERPrivateReasonDecoder(dim=16, reason_dim=21, action_dim=4)
    foundation = _foundation(16)
    decoder.initialize_from_foundation(foundation)

    torch.testing.assert_close(
        decoder.local_proj.weight, foundation.trunk.value_proj.weight
    )
    torch.testing.assert_close(
        decoder.local_proj.bias, foundation.trunk.value_proj.bias
    )
    torch.testing.assert_close(
        decoder.local_head.weight, foundation.trunk.logit_head.weight
    )
    torch.testing.assert_close(
        decoder.local_head.bias, foundation.trunk.logit_head.bias
    )


def test_reason_mix_regret_updates_mix_gate_only() -> None:
    torch.manual_seed(47)
    decoder = METERPrivateReasonDecoder(dim=16, reason_dim=21, action_dim=4)
    output = decoder(
        patch_tokens_by_layer=torch.randn(4, 3, 12, 16),
        reason_logits_calalign=torch.randn(4, 21),
        action_logits_final=torch.randn(4, 4),
        action_nodes=torch.randn(4, 4, 16),
        factor_to_reason_tokens=torch.randn(4, 21, 16),
        factor_support_map=torch.softmax(torch.randn(4, 21, 12), -1),
        factor_counter_map=torch.softmax(torch.randn(4, 21, 12), -1),
        factor_reliability=torch.rand(4, 21),
        factor_support_null=torch.rand(4, 21),
        progress=1.0,
    )
    output["reason_logits_global"].retain_grad()
    output["reason_logits_local"].retain_grad()
    loss = meter_reason_loss(
        output,
        torch.randint(0, 2, (4, 21)).float(),
        torch.rand(4, 21),
    )["mix_regret"]
    loss.backward()

    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in decoder.mix_gate.parameters()
    )
    assert output["reason_logits_global"].grad is None
    assert output["reason_logits_local"].grad is None
