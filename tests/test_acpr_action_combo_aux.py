import torch

from fate_oia.models.acpr_action_combo_aux import ACPRActionComboAux
from fate_oia.utils.acpr_pair_mining import action_vectors_to_subset_id


def test_acpr_action_combo_ids():
    x = torch.tensor([[1,0,0,1],[1,0,1,0],[0,1,0,1]]).float()
    ids = action_vectors_to_subset_id(x)
    assert ids.tolist() == [9, 5, 10]
    head = ACPRActionComboAux()
    out = head(torch.randn(2, 25, 384), torch.randn(2, 4))
    assert out["action_set_logits"].shape == (2, 16)
