from __future__ import annotations

from copy import deepcopy
import json

import pytest

from fate_oia.engine.evaluate_meter_oia_v3_heca_pilot import (
    evaluate_heca_pilot,
    validate_heca_pilot_recomputation,
)


def _epoch() -> dict:
    per_factor = []
    for factor in range(21):
        good = factor < 12
        per_factor.append(
            {
                "factor_id": factor,
                "source_count": 10 if good else 0,
                "same_type_margin": 0.1 if good else None,
                "state_auprc": 0.7 if good else None,
                "state_frequency_baseline": 0.3 if good else None,
                "state_auc": 0.7 if good else None,
                "state_positive_count": 40 if good else 0,
                "state_negative_count": 40 if good else 0,
                "state_identifiable": good,
                "audit_split": "train_audit",
                "provenance_valid_count": 40 if good else 0,
                "visual_confidence_mean": 0.5 if good else None,
                "visual_confidence_std": 0.1 if good else None,
            }
        )
    global_ap = [0.38] * 21
    final_ap = [0.386] * 19 + [0.38] * 2
    return {
        "branches": {
            "action_visual": {"Act_mAP": 0.700, "Act_mF1": 0.700},
            "action_final": {
                "Act_mAP": 0.706,
                "Act_mF1": 0.704,
                "Act_per_label_ap": [0.70] * 4,
            },
            "factor_off": {"Act_mAP": 0.699, "Act_mF1": 0.698},
            "state_uniform": {"Act_per_label_ap": [0.698, 0.698, 0.698, 0.70]},
            "reason_calalign": {
                "Exp_mAP": 0.380,
                "Exp_mF1": 0.35,
                "Exp_per_label_ap": global_ap,
            },
            "reason_global": {
                "Exp_mAP": 0.379,
                "Exp_mF1": 0.35,
                "Exp_per_label_ap": global_ap,
            },
            "reason_final": {
                "Exp_mAP": 0.385,
                "Exp_mF1": 0.36,
                "Exp_per_label_ap": final_ap,
            },
        },
        "typed": {
            "train_audit": {"per_factor": per_factor},
            "action_correction_rms_ratio_mean": [0.1] * 4,
            "patch_audit": {
                "unique_sample_count": 512,
                "action_coverage": [0, 1, 2, 3],
                "factor_coverage": list(range(12)),
                "selected_minus_control_mean": 0.02,
                "selected_minus_control_ci": {"low": 0.001},
            },
            "identity_target_delta": [0.02] * 4,
            "identity_wrong_delta": [0.01] * 4,
        },
        "runtime": {
            "peak_reserved_gb": 40.0,
            "dino_call_count": {
                "main": 10,
                "factor_off": 0,
                "state_uniform": 0,
                "reason_correction_off": 0,
            },
        },
        "max_action_logit": 20.0,
        "foundation_grad_ema": 2.0,
        "emergency_cap_rate": 0.0,
        "action_emergency_cap_rate": 0.0,
    }


def _inputs() -> tuple[list[dict], dict, dict, dict, list[dict]]:
    epochs = [_epoch() for _ in range(4)]
    audit = {
        "pass": True,
        "git_head": "abc",
        "dynamic_checks": {
            "checks": {
                "action_progress_zero_equivalence": True,
                "reason_progress_zero_equivalence": True,
                "label_nodes_progress_zero_equivalence": True,
            }
        },
    }
    ontology = {
        "schema_version": 4,
        "factor_count": 21,
        "state_count": 63,
        "sha256": "schema",
    }
    tau = {
        "source_split": "train_main",
        "provenance_valid_count": [40] * 21,
    }
    gradients = [
        {
            "optimizer_step": step,
            "action_credit_ramp": 1.0,
            # Gate E only assesses the route after the zero-initialized
            # state-effect table has become mature.
            "action_state_effect_norm": 0.12,
            "action_to_anchor_query": 0.0,
            "action_to_state_bridge_ratio": 0.05,
            "action_to_credit_adapter": 1.0,
            "reason_to_action_credit": 0.0,
            "pu_to_action_factor": 0.0,
            "measurement_to_foundation": 0.0,
        }
        for step in (1, 2)
    ]
    return epochs, audit, ontology, tau, gradients


def test_real_pilot_evaluator_requires_all_gate_evidence() -> None:
    epochs, audit, ontology, tau, gradients = _inputs()
    result = evaluate_heca_pilot(
        epochs=epochs,
        implementation_audit=audit,
        ontology_manifest=ontology,
        tau_stats=tau,
        gradient_rows=gradients,
        git_head="abc",
    )
    assert result["pass"] is True
    assert result["gates"] == {letter: True for letter in "ABCDEFG"}


def test_pilot_evaluator_fails_closed_when_gate_c_is_not_two_epoch_stable() -> None:
    epochs, audit, ontology, tau, gradients = _inputs()
    broken = deepcopy(epochs)
    broken[2]["branches"]["action_final"]["Act_mAP"] = 0.701
    result = evaluate_heca_pilot(
        epochs=broken,
        implementation_audit=audit,
        ontology_manifest=ontology,
        tau_stats=tau,
        gradient_rows=gradients,
        git_head="abc",
    )
    assert result["pass"] is False
    assert result["gates"]["C"] is False


def test_gate_g_uses_action_emergency_cap_not_foundation_gradient_clipping() -> None:
    epochs, audit, ontology, tau, gradients = _inputs()
    for epoch in epochs:
        epoch["emergency_cap_rate"] = 1.0
        epoch["action_emergency_cap_rate"] = 0.0
    result = evaluate_heca_pilot(
        epochs=epochs,
        implementation_audit=audit,
        ontology_manifest=ontology,
        tau_stats=tau,
        gradient_rows=gradients,
        git_head="abc",
    )

    assert result["gates"]["G"] is True


def test_pilot_evaluator_requires_exactly_four_epochs() -> None:
    epochs, audit, ontology, tau, gradients = _inputs()
    with pytest.raises(ValueError, match="exactly four"):
        evaluate_heca_pilot(
            epochs=epochs[:2],
            implementation_audit=audit,
            ontology_manifest=ontology,
            tau_stats=tau,
            gradient_rows=gradients,
            git_head="abc",
        )


def test_gate_b_accepts_one_class_provenance_when_visual_measurement_is_healthy() -> None:
    epochs, audit, ontology, tau, gradients = _inputs()
    result = evaluate_heca_pilot(
        epochs=epochs,
        implementation_audit=audit,
        ontology_manifest=ontology,
        tau_stats=tau,
        gradient_rows=gradients,
        git_head="abc",
    )
    assert result["gates"]["B"] is True
    assert result["gate_payloads"]["B"]["evidence"]["provenance_coverage_factor_count"] >= 12


def test_gate_b_uses_auc_random_baseline_not_prevalence_and_skips_unidentifiable_states() -> None:
    epochs, audit, ontology, tau, gradients = _inputs()
    high_prevalence = epochs[-1]["typed"]["train_audit"]["per_factor"][0]
    high_prevalence.update(
        state_auprc=0.90,
        state_frequency_baseline=0.80,
        state_auc=0.75,
        state_positive_count=80,
        state_negative_count=20,
        state_identifiable=True,
    )
    unknown = epochs[-1]["typed"]["train_audit"]["per_factor"][12]
    unknown.update(
        source_count=12,
        same_type_margin=0.20,
        state_auprc=0.99,
        state_frequency_baseline=0.01,
        state_auc=0.99,
        state_positive_count=0,
        state_negative_count=100,
        state_identifiable=False,
        provenance_valid_count=100,
        visual_confidence_mean=0.5,
        visual_confidence_std=0.1,
    )
    result = evaluate_heca_pilot(
        epochs=epochs,
        implementation_audit=audit,
        ontology_manifest=ontology,
        tau_stats=tau,
        gradient_rows=gradients,
        git_head="abc",
    )
    evidence = result["gate_payloads"]["B"]["evidence"]
    assert result["gates"]["B"] is True
    assert 0 in evidence["quality_factor_ids"]
    assert 12 not in evidence["identifiable_factor_ids"]


def test_pilot_recomputation_rejects_saved_gate_tampering(tmp_path) -> None:
    epochs, audit, ontology, tau, gradients = _inputs()
    (tmp_path / "heca_implementation_audit_input.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )
    (tmp_path / "heca_ontology_manifest_input.json").write_text(
        json.dumps(ontology), encoding="utf-8"
    )
    (tmp_path / "heca_tau_stats_input.json").write_text(
        json.dumps(tau), encoding="utf-8"
    )
    (tmp_path / "heca_gradient_ownership.jsonl").write_text(
        "\n".join(json.dumps(row) for row in gradients) + "\n", encoding="utf-8"
    )
    loss_rows = []
    for index, epoch in enumerate(epochs):
        epoch_dir = tmp_path / f"epoch_{index:03d}"
        epoch_dir.mkdir()
        for name, payload in (
            ("branch_metrics.json", epoch["branches"]),
            ("typed_evidence.json", epoch["typed"]),
            ("runtime.json", epoch["runtime"]),
        ):
            (epoch_dir / name).write_text(json.dumps(payload), encoding="utf-8")
        loss_rows.append(
            {
                "epoch": index,
                "action_final_logit_abs_max": epoch["max_action_logit"],
                "foundation_grad_ema": epoch["foundation_grad_ema"],
                "foundation_grad_norm": 0.0,
                "foundation_grad_cap": 1.0,
                "action_emergency_cap_rate": epoch["action_emergency_cap_rate"],
            }
        )
    (tmp_path / "loss_components.jsonl").write_text(
        "\n".join(json.dumps(row) for row in loss_rows) + "\n", encoding="utf-8"
    )
    result = evaluate_heca_pilot(
        epochs=epochs,
        implementation_audit=audit,
        ontology_manifest=ontology,
        tau_stats=tau,
        gradient_rows=gradients,
        git_head="abc",
    )
    (tmp_path / "HECA_PILOT_PASS.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    for letter, payload in result["gate_payloads"].items():
        (tmp_path / f"HECA_GATE_{letter}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    assert validate_heca_pilot_recomputation(
        tmp_path, expected_git_head="abc"
    ) == []
    gate_c = tmp_path / "HECA_GATE_C.json"
    tampered = json.loads(gate_c.read_text(encoding="utf-8"))
    tampered["evidence"] = {"forged": True}
    gate_c.write_text(json.dumps(tampered), encoding="utf-8")
    assert "pilot_recomputation:gate_C_mismatch" in validate_heca_pilot_recomputation(
        tmp_path, expected_git_head="abc"
    )
