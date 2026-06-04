import torch

from fate_oia.losses.pcgrad_lite import apply_pcgrad_lite


def test_pcgrad_accumulates_instead_of_overwriting_existing_grad():
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    p.grad = torch.tensor([0.25, 0.25])
    loss1 = (p * torch.tensor([1.0, 0.0])).sum() / 2.0
    loss2 = (p * torch.tensor([0.0, 1.0])).sum() / 2.0
    stats = apply_pcgrad_lite([loss1, loss2], [p], retain_graph=True, grad_accumulation_steps=2)
    assert stats["overwrote_existing_grad"] is False
    assert stats["grad_accumulation_steps"] == 2
    assert stats["accumulated_microbatches"] >= 1
    assert torch.all(p.grad != torch.tensor([0.5, 0.5]))
    assert torch.all(p.grad >= 0.25)
