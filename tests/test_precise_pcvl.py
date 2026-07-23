from pathlib import Path

import torch

from fate_oia.engine.run_precise_pcvl import build_oracle_structured_evidence, train_pcvl_step, validate_pcvl_artifacts

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
    for artifact in ("pcvl_probabilities.pt", "pcvl_labels.pt", "pcvl_file_names.json"):
        assert artifact in source


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


def test_pcvl_validation_rejects_unbound_or_nonfinite_artifacts(tmp_path):
    (tmp_path / "pcvl_metrics.json").write_text('{"u0_action_map": 0.1, "u1_action_map": 0.2, "u2_action_map": 0.2, "u3_action_map": 0.3, "predicate_action_value_supported": true}', encoding="utf-8")
    (tmp_path / "pcvl_per_action.json").write_text('{"u0": [0.1, 0.1, 0.1, 0.1], "u1": [0.2, 0.2, 0.2, 0.2], "u2": [0.2, 0.2, 0.2, 0.2], "u3": [0.3, 0.3, 0.3, 0.3]}', encoding="utf-8")
    (tmp_path / "pcvl_bootstrap.json").write_text('{"delta_value": {"mean": 0.1, "ci_low": 0.01, "ci_high": 0.2, "positive_rate": 1.0}}', encoding="utf-8")
    (tmp_path / "pcvl_value_decomposition.json").write_text('{"delta_value": 0.1, "delta_measurement": 0.0, "delta_interaction": 0.1, "delta_learned_value": 0.1, "delta_learned_interaction": 0.1}', encoding="utf-8")
    try:
        validate_pcvl_artifacts(tmp_path)
    except RuntimeError as error:
        assert "provenance" in str(error).lower()
    else:
        raise AssertionError("unbound PCVL artifacts were accepted")


def test_pcvl_validation_recomputes_metrics_from_raw_predictions(tmp_path):
    probabilities = {key: torch.tensor([[0.9, 0.1, 0.8, 0.2], [0.1, 0.9, 0.2, 0.8]]) for key in ("u0", "u1", "u2", "u3")}
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    torch.save(probabilities, tmp_path / "pcvl_probabilities.pt")
    torch.save(labels, tmp_path / "pcvl_labels.pt")
    (tmp_path / "pcvl_file_names.json").write_text('{"file_names": ["a", "b"]}', encoding="utf-8")
    (tmp_path / "pcvl_metrics.json").write_text('{"u0_action_map": 0.0, "u1_action_map": 0.0, "u2_action_map": 0.0, "u3_action_map": 0.0, "predicate_action_value_supported": false}', encoding="utf-8")
    (tmp_path / "pcvl_per_action.json").write_text('{"u0": [0,0,0,0], "u1": [0,0,0,0], "u2": [0,0,0,0], "u3": [0,0,0,0]}', encoding="utf-8")
    (tmp_path / "pcvl_bootstrap.json").write_text('{"delta_value":{"mean":0,"ci_low":0,"ci_high":0,"positive_rate":0},"delta_measurement":{"mean":0,"ci_low":0,"ci_high":0,"positive_rate":0},"delta_interaction":{"mean":0,"ci_low":0,"ci_high":0,"positive_rate":0},"delta_learned_value":{"mean":0,"ci_low":0,"ci_high":0,"positive_rate":0},"delta_learned_interaction":{"mean":0,"ci_low":0,"ci_high":0,"positive_rate":0}}', encoding="utf-8")
    (tmp_path / "pcvl_value_decomposition.json").write_text('{"delta_value":0,"delta_measurement":0,"delta_interaction":0,"delta_learned_value":0,"delta_learned_interaction":0}', encoding="utf-8")
    names_hash = __import__("hashlib").sha256("a\nb".encode()).hexdigest()
    provenance = {key: "x" for key in ("git_head","source_tree_sha256","config_sha256","skill_sha256","pretrained_weights_sha256","action_schema_sha256","train_audit_indices_sha256","model_trainable_state_sha256","probe_state_sha256")}
    provenance.update({"train_audit_file_names_sha256": names_hash, "file_names_sha256": names_hash, "sample_count": 2, "epoch": 2})
    (tmp_path / "pcvl_provenance.json").write_text(__import__("json").dumps(provenance), encoding="utf-8")
    try:
        validate_pcvl_artifacts(tmp_path)
    except RuntimeError as error:
        assert "recomputed" in str(error).lower()
    else:
        raise AssertionError("tampered PCVL aggregate metrics were accepted")
