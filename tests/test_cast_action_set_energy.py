import torch

from fate_oia.models.cast_action_set_energy import CastActionSetEnergy, action_targets_to_subset_ids
from fate_oia.losses.cast_oia_losses import cardinality_loss, drop_add_subset_margin_loss


def test_action_subset_bit_coding_and_marginalization():
    model = CastActionSetEnergy(dim=16, action_dim=4)
    targets = torch.tensor([[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
    assert action_targets_to_subset_ids(targets).tolist() == [9, 5, 10]
    action_nodes = torch.randn(3, 4, 16)
    graph_context = torch.randn(3, 16)
    subset_context = torch.randn(3, 16, 16)
    out = model(action_nodes, graph_context, subset_context)
    assert out["action_set_logits"].shape == (3, 16)
    assert out["action_set_probs"].shape == (3, 16)
    assert out["action_logits"].shape == (3, 4)
    assert torch.allclose(out["action_set_probs"].sum(-1), torch.ones(3), atol=1e-5)
    expected = out["action_set_probs"] @ model.subset_membership.to(out["action_set_probs"].device)
    assert torch.allclose(out["action_marginal_probs"], expected, atol=1e-5)


def test_combo_losses_penalize_collapse_and_cardinality():
    gt = torch.tensor([[1, 0, 0, 1]], dtype=torch.float32)
    logits = torch.zeros(1, 16)
    logits[0, 1] = 3.0   # forward only
    logits[0, 8] = 3.0   # right only
    logits[0, 11] = 3.0  # forward+stop+right superset
    loss = drop_add_subset_margin_loss(logits, gt, margin=0.25)
    assert loss.item() > 0
    probs = torch.softmax(logits, dim=-1)
    c_loss = cardinality_loss(probs, gt)
    assert c_loss.item() > 0
