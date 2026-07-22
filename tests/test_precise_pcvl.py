import torch

from fate_oia.engine.run_precise_pcvl import build_oracle_structured_evidence

from fate_oia.models.precise_pcvl_probes import PRECISEPCVLProbes


def test_pcvl_uses_equal_capacity_detached_base_probes():
    probes = PRECISEPCVLProbes()
    base = torch.randn(2, 4, 384, requires_grad=True)
    out = probes(base, torch.randn(2, 4, 384), torch.randn(2, 4, 384), torch.randn(2, 4, 384))
    assert set(out) == {"u0", "u1", "u2", "u3"}
    sum(value.square().mean() for value in out.values()).backward()
    assert base.grad is None
    assert len(list(probes.u0.parameters())) == len(list(probes.u1.parameters())) == len(list(probes.u2.parameters())) == len(list(probes.u3.parameters()))


def test_pcvl_oracle_is_constructed_from_train_only_structured_targets():
    targets = {
        "presence": torch.tensor([[1.0, 0.0]]),
        "presence_valid": torch.ones(1, 2),
        "observability": torch.tensor([[1.0, 1.0]]),
        "state": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        "state_valid": torch.ones(1, 2),
        "part_coordinates": torch.tensor([[[[0.2, 0.3]], [[0.8, 0.7]]]]),
        "part_scales": torch.ones(1, 2, 1, 2) * 0.1,
        "part_valid": torch.ones(1, 2),
        "soft_masks": torch.zeros(1, 2, 2, 2),
    }
    reference = torch.randn(1, 2, 16)
    oracle = build_oracle_structured_evidence(targets, reference)
    changed_reference = torch.randn_like(reference) * 100.0
    assert oracle.shape == reference.shape
    assert torch.equal(oracle, build_oracle_structured_evidence(targets, changed_reference))
    assert oracle[0, 0].abs().sum() > 0
    assert oracle[0, 1].abs().sum() > 0
    assert not torch.equal(oracle[0, 0], oracle[0, 1])
