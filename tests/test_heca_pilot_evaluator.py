from __future__ import annotations

from copy import deepcopy

from fate_oia.engine.evaluate_meter_oia_v3_heca_pilot import evaluate_heca_pilot


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
                "observability_auc": 0.7 if good else None,
            }
        )
    global_ap = [0.38] * 21
    final_ap = [0.386] * 12 + [0.38] * 9
    return {
        "branches": {
            "action_visual": {"Act_mAP": 0.700, "Act_mF1": 0.700},
            "action_final": {"Act_mAP": 0.706, "Act_mF1": 0.704},
            "factor_off": {"Act_mAP": 0.699, "Act_mF1": 0.698},
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
                "Exp_mAP": 0.383,
                "Exp_mF1": 0.36,
                "Exp_per_label_ap": final_ap,
            },
        },
        "typed": {
            "train_audit": {"per_factor": per_factor},
            "action_correction_rms_ratio_mean": [0.1] * 4,
            "patch_audit": {
                "unique_sample_count": 128,
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
    }


def _inputs() -> tuple[list[dict], dict, dict, dict, list[dict]]:
    epochs = [_epoch(), _epoch()]
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
        "alpha": 20.0,
        "tau": [0.10 + index * 0.01 for index in range(21)],
    }
    gradients = [
        {
            "optimizer_step": 1,
            "action_to_anchor_query": 0.0,
            "action_to_state_bridge_ratio": 0.05,
            "action_to_credit_adapter": 1.0,
            "reason_to_action_credit": 0.0,
            "pu_to_action_factor": 0.0,
            "measurement_to_foundation": 0.0,
        }
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
    broken[0]["branches"]["action_final"]["Act_mAP"] = 0.701
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
