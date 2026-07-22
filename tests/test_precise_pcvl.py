import torch

from fate_oia.models.precise_pcvl_probes import PRECISEPCVLProbes


def test_pcvl_uses_equal_capacity_detached_base_probes():
    probes = PRECISEPCVLProbes()
    base = torch.randn(2, 4, 384, requires_grad=True)
    out = probes(base, torch.randn(2, 10, 384), torch.randn(2, 10, 384), torch.randn(2, 4, 384))
    assert set(out) == {"u0", "u1", "u2", "u3"}
    sum(value.square().mean() for value in out.values()).backward()
    assert base.grad is None
    assert len(list(probes.u0.parameters())) == len(list(probes.u1.parameters())) == len(list(probes.u2.parameters())) == len(list(probes.u3.parameters()))
