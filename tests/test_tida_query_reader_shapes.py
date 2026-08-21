import torch

from fate_oia.models.tida_terminal_query_reader import TIDATerminalQueryReader


def test_query_reader_shapes_and_sparse_attention():
    reader = TIDATerminalQueryReader(dim=16, num_actions=4, num_predicates=32, layer_ids=(3, 7, 11))
    patches = torch.randn(2, 3, 40, 16)
    actions = torch.randn(2, 4, 16)
    predicates = torch.randn(2, 32, 16)
    identities = torch.randn(32, 16)
    out = reader(patches, actions, predicates, identities, grid_hw=(5, 8))
    assert out["query_tokens"].shape == (2, 36, 16)
    assert out["query_attention"].shape == (2, 36, 40)
    assert torch.allclose(out["query_attention"].sum(-1), torch.ones(2, 36), atol=1e-5)
    assert out["layer_order"] == (11, 7, 3)
