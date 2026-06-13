from __future__ import annotations

import torch
from fate_oia.models.eagle_pu_action_set_aux import ActionSetAuxiliaryHead

def test_action_set_aux_enumerates_16_subsets_without_final_marginal_contract():
    head = ActionSetAuxiliaryHead(dim=32, action_dim=4)
    assert head.subset_membership.shape == (16, 4)
    assert head.subset_membership[0].sum().item() == 0
    assert head.subset_membership[-1].sum().item() == 4
    out = head(torch.randn(2, 25, 32), torch.randn(2, 4))
    assert out["action_set_logits"].shape == (2, 16)
    assert out["action_set_probs"].shape == (2, 16)
    assert out["cardinality_logits"].shape == (2, 5)
    assert torch.allclose(out["action_set_probs"].sum(-1), torch.ones(2), atol=1e-5)
