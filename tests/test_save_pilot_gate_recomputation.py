import copy

from fate_oia.engine.evaluate_save_oia_pilot import recompute_save_pilot_gates


def test_saved_pass_cannot_override_recomputed_failure():
    evidence = _passing_evidence()
    evidence["saved_pass"] = True
    evidence["epochs"][-1]["action"]["final_mAP"] = 0.50
    result = recompute_save_pilot_gates(evidence)
    assert result["gates"]["B"] is False
    assert result["pass"] is False


def test_all_gates_are_recomputed_and_hashes_propagate():
    evidence = _passing_evidence()
    result = recompute_save_pilot_gates(evidence)
    assert result["pass"] is True
    assert result["gates"] == {letter: True for letter in "ABCDEFG"}
    assert result["bindings"] == evidence["bindings"]
    assert result["numeric_candidate_eligible"] is True
    assert result["selection"]["primary"] == "final_raw_joint"


def _passing_evidence():
    epochs = []
    for _ in range(4):
        epochs.append(
            {
                "action": {
                    "base_mAP": 0.66,
                    "final_mAP": 0.67,
                    "base_mF1": 0.67,
                    "final_mF1": 0.68,
                    "evidence_rms": [0.05] * 4,
                    "logit_collapsed": False,
                    "emergency_cap_rate": 0.0,
                },
                "reason": {
                    "clean_mAP": 0.38,
                    "final_mAP": 0.39,
                    "clean_mF1": 0.37,
                    "final_mF1": 0.40,
                    "private_tail_mAP": 0.22,
                    "clean_tail_mAP": 0.20,
                    "clean_metric": 0.38,
                    "reliability_min": 0.1,
                    "reliability_max": 0.9,
                },
            }
        )
    return {
        "bindings": {name: name + "-hash" for name in (
            "git_head", "config_hash", "source_tree_hash", "schema_hash",
            "split_hash", "checkpoint_hash", "logits_hash", "labels_hash",
            "file_order_hash",
        )},
        "structure": {
            "progress_zero_max_abs": 1e-8,
            "ordinary_batches": 10,
            "dino_calls": 10,
            "dino_grad_norm": 0.0,
            "feature_cache": False,
            "token_compression": "none",
        },
        "epochs": epochs,
        "utility": {
            "audit_auc": 0.70,
            "selected_minus_control": 0.02,
            "action_coverage": [0, 1, 2, 3],
            "valid_factor_count": 12,
            "std": 0.05,
        },
        "specificity": {
            "target_deletion": 0.03,
            "wrong_deletion": 0.01,
            "identity_corruption_ap_drop": 0.01,
            "max_factor_share": 0.5,
            "effective_factor_count": 2.0,
        },
        "faithfulness": {
            "evidence_only_margin_retention": 0.95,
            "selected_deletion": 0.03,
            "matched_control": 0.01,
            "target_action_change": 0.04,
            "wrong_action_change": 0.01,
            "conservation_max_abs": 1e-7,
        },
        "gradient_runtime": {
            "private_to_action": 0.0,
            "clean_to_shared": 0.1,
            "action_to_inquiry": 0.1,
            "action_to_utility": 0.1,
            "grounding_to_foundation": 0.0,
            "pu_non_private": 0.0,
            "reserved_gb": 40.0,
            "finite": True,
            "oom": False,
        },
    }
