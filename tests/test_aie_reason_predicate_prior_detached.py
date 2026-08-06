import torch

from fate_oia.models.aie_reason_rereader import AIEReasonRereader


def test_reason_predicate_inputs_are_detached():
    module = AIEReasonRereader(dim=32)
    attention = torch.softmax(torch.randn(1, 32, 20), -1).requires_grad_(); probability = torch.rand(1, 32, requires_grad=True)
    out = module(torch.randn(1, 21, 32), torch.randn(1, 3, 20, 32), torch.randn(1, 4, 4, 32), torch.softmax(torch.randn(1, 4, 4, 20), -1), torch.randn(1, 4, 4), attention, probability, torch.randn(1, 21))
    out["reason_logits_final_train"].sum().backward()
    assert attention.grad is None and probability.grad is None

