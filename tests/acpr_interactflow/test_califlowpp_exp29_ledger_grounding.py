from __future__ import annotations

import torch

from fate_oia.acpr_interactflow.exp29_head import Exp29Head


def test_exp29_attention_and_logits_depend_on_ledger_contributions() -> None:
    torch.manual_seed(1)
    head = Exp29Head(dim=24, exp_dim=29)
    factors = torch.randn(2, 6, 24)
    predicates = torch.randn(2, 48, 24)
    global_hidden = torch.randn(2, 24)
    action_logits = torch.randn(2, 3)
    contrib_a = torch.zeros(2, 6, 3)
    contrib_b = contrib_a.clone()
    contrib_b[:, 3, :] = 2.0

    out_a = head(factors, predicates, contrib_a, global_hidden, action_logits)
    out_b = head(factors, predicates, contrib_b, global_hidden, action_logits)

    assert out_a.cluster_attention_to_factors.shape == (2, 29, 6)
    assert out_a.logits_raw.shape == (2, 29)
    assert out_a.logits_calibrated.shape == (2, 29)
    assert out_a.cluster_reliability.shape == (29,)
    assert out_a.cluster_to_state_prior.shape == (29, 6)
    assert not torch.allclose(out_a.cluster_attention_to_factors, out_b.cluster_attention_to_factors)
    assert not torch.allclose(out_a.logits_raw, out_b.logits_raw)
