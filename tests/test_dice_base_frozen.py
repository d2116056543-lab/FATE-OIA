import torch
from torch import nn

from fate_oia.models.dice_oia_model import DICEOIAModel


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.ones(()))


def test_base_is_frozen_by_construction():
    model = DICEOIAModel(FakeBase(), dim=8, num_layers=2, num_predicates=5)
    assert not any(p.requires_grad for p in model.base_model.parameters())
    model.train(); assert not model.base_model.training
