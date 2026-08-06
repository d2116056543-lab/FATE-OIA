import torch

from fate_oia.models.aie_reason_rereader import AIEReasonRereader


def test_reason_attention_spans_all_layers_and_patches():
    module = AIEReasonRereader(dim=32)
    out = module(torch.randn(1, 21, 32), torch.randn(1, 3, 20, 32), torch.randn(1, 4, 4, 32), torch.softmax(torch.randn(1, 4, 4, 20), -1), torch.randn(1, 4, 4), torch.softmax(torch.randn(1, 32, 20), -1), torch.rand(1, 32), torch.randn(1, 21))
    assert out["reason_private_attention"].shape == (1, 21, 3, 20)
    torch.testing.assert_close(out["reason_private_attention"].sum((2, 3)), torch.ones(1, 21), atol=1e-5, rtol=0)

