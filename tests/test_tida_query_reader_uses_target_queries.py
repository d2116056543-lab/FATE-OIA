import torch

from fate_oia.models.tida_terminal_query_reader import TIDATerminalQueryReader


def test_action_target_query_changes_corresponding_history_token():
    torch.manual_seed(2)
    reader = TIDATerminalQueryReader(dim=8)
    patches = torch.randn(1, 3, 16, 8)
    actions = torch.randn(1, 4, 8)
    predicates = torch.randn(1, 32, 8)
    identities = torch.randn(32, 8)
    a = reader(patches, actions, predicates, identities, grid_hw=(4, 4))["query_tokens"]
    actions[:, 0] += 2.0
    b = reader(patches, actions, predicates, identities, grid_hw=(4, 4))["query_tokens"]
    assert not torch.allclose(a[:, 0], b[:, 0])
