import torch


def typed_inputs(batch: int = 2, dim: int = 32, patches: int = 20):
    torch.manual_seed(7)
    return (
        torch.randn(batch, 21, dim),
        torch.randn(batch, 3, patches, dim),
    )
