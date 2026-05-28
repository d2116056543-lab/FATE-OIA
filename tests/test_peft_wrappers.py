import torch
from torch import nn

from fate_oia.models.peft.adaptformer_dino_wrapper import AdaptFormerDINOBlockWrapper
from fate_oia.models.peft.lora_dino_wrapper import LoRALinear
from fate_oia.models.peft.vpt_dino_wrapper import ShallowVPT


def test_lora_linear_preserves_shape_and_freezes_base():
    base = nn.Linear(8, 4)
    wrapped = LoRALinear(base, rank=2)
    out = wrapped(torch.randn(3, 8))
    assert out.shape == (3, 4)
    assert not any(p.requires_grad for p in wrapped.base.parameters())


def test_vpt_prepends_prompt_tokens():
    vpt = ShallowVPT(dim=8, prompt_len=3)
    out = vpt(torch.randn(2, 5, 8))
    assert out.shape == (2, 8, 8)


def test_adaptformer_wrapper_freezes_block_and_preserves_shape():
    block = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 8))
    wrapped = AdaptFormerDINOBlockWrapper(block, dim=8, bottleneck_dim=4)
    out = wrapped(torch.randn(2, 6, 8))
    assert out.shape == (2, 6, 8)
    assert not any(p.requires_grad for p in wrapped.block.parameters())
