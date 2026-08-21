import torch

from fate_oia.models.tida_terminal_query_reader import TIDATerminalQueryReader


def test_each_dino_layer_can_independently_change_recursive_read():
    torch.manual_seed(9)
    reader = TIDATerminalQueryReader(dim=8).eval()
    field = torch.randn(1, 3, 16, 8)
    args = (torch.randn(1, 4, 8), torch.randn(1, 32, 8), torch.randn(32, 8))
    base = reader(field, *args, grid_hw=(4, 4))["query_tokens"]
    changed = field.clone(); changed[:, 0] += 3.0
    other = reader(changed, *args, grid_hw=(4, 4))["query_tokens"]
    assert not torch.allclose(base, other)
