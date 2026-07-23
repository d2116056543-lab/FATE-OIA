from pathlib import Path

import torch

from fate_oia.engine.run_precise_pcvl import build_oracle_structured_evidence, train_pcvl_step

from fate_oia.models.precise_pcvl_probes import PRECISEPCVLProbes


def test_pcvl_uses_equal_capacity_detached_base_probes():
    probes = PRECISEPCVLProbes()
    reference = probes.u0.state_dict()
    for probe in (probes.u1, probes.u2, probes.u3):
        assert all(torch.equal(reference[name], probe.state_dict()[name]) for name in reference)
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


def test_pcvl_oracle_ignores_padded_part_coordinates_and_scales():
    targets = {
        "presence": torch.ones(1, 1), "presence_valid": torch.ones(1, 1),
        "observability": torch.ones(1, 1), "state": torch.zeros(1, 1, 2),
        "state_valid": torch.ones(1, 1), "part_valid": torch.ones(1, 1),
        "part_coordinates": torch.tensor([[[[0.25, 0.75], [0.0, 0.0], [0.0, 0.0]]]]),
        "part_scales": torch.tensor([[[[0.10, 0.20], [0.0, 0.0], [0.0, 0.0]]]]),
        "soft_masks": torch.zeros(1, 1, 2, 2),
    }
    changed_padding = {key: value.clone() for key, value in targets.items()}
    changed_padding["part_coordinates"][0, 0, 1:] = torch.tensor([[0.9, 0.1], [0.7, 0.3]])
    reference = torch.randn(1, 1, 32)
    first = build_oracle_structured_evidence(targets, reference)
    second = build_oracle_structured_evidence(changed_padding, reference)
    assert torch.equal(first, second)


def test_pcvl_pilot_records_optimizer_steps_for_gate_validation():
    source = (Path(__file__).resolve().parents[1] / "fate_oia" / "engine" / "train_precise_oia.py").read_text(encoding="utf-8")
    assert "pcvl_optimizer_step_count" in source


def test_pcvl_reports_learned_evidence_and_exchange_value_not_only_oracle_value():
    source = (Path(__file__).resolve().parents[1] / "fate_oia" / "engine" / "run_precise_pcvl.py").read_text(encoding="utf-8")
    assert '"delta_learned_value"' in source
    assert '"delta_learned_interaction"' in source


def test_pcvl_step_reports_real_gradient_and_parameter_delta():
    probes = PRECISEPCVLProbes(dim=16)
    optimizer = torch.optim.AdamW(probes.parameters(), lr=1e-3)
    output = {
        "action_tokens_direct": torch.randn(2, 4, 16),
        "explicit_evidence_tokens": torch.randn(2, 10, 16),
        "action_evidence_family_mask": torch.ones(4, 10, dtype=torch.bool),
        "action_exchange_delta": torch.randn(2, 4, 16),
    }
    structured = {
        "presence": torch.ones(2, 10), "presence_valid": torch.ones(2, 10),
        "observability": torch.ones(2, 10), "state_valid": torch.ones(2, 10),
        "state": torch.zeros(2, 10, 4), "part_valid": torch.ones(2, 10),
        "part_coordinates": torch.rand(2, 10, 4, 2), "part_scales": torch.rand(2, 10, 4, 2),
        "soft_masks": torch.rand(2, 10, 4, 4),
    }
    result = train_pcvl_step(probes, optimizer, output, structured, torch.randint(0, 2, (2, 4)).float())
    assert result["grad_norm"] > 0
    assert result["parameter_delta_norm"] > 0
