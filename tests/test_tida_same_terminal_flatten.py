import torch

from fate_oia.utils.tida_temporal_interventions import flatten_selected_predicates


def test_flatten_changes_only_selected_predicate_history():
    history = torch.randn(1, 6, 36, 8)
    terminal = torch.randn(1, 32, 8)
    out = flatten_selected_predicates(history, terminal, predicate_indices=[3, 7], action_count=4)
    assert torch.equal(out[:, :, 7], terminal[:, 3:4].expand(-1, 6, -1))
    assert torch.equal(out[:, :, 11], terminal[:, 7:8].expand(-1, 6, -1))
    assert torch.equal(out[:, :, 6], history[:, :, 6])
