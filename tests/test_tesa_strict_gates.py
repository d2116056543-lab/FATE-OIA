import math

from fate_oia.engine import evaluate_tesa_pilot as pilot
from fate_oia.engine import train_acpr_meter_oia as trainer
from fate_oia.engine.eval_acpr_meter_oia import SEQUENTIAL_MODES


def test_targeted_identity_modes_and_ap_matrix_are_not_dense_proxies():
    assert all(f"schema_target_{action}" in SEQUENTIAL_MODES for action in range(4))
    branches = {
        "action_final": {"Act_per_label_ap": [0.8, 0.7, 0.6, 0.5]},
        "schema_target_0": {"Act_per_label_ap": [0.7, 0.69, 0.59, 0.49]},
        "schema_target_1": {"Act_per_label_ap": [0.79, 0.6, 0.59, 0.49]},
        "schema_target_2": {"Act_per_label_ap": [0.79, 0.69, 0.5, 0.49]},
        "schema_target_3": {"Act_per_label_ap": [0.79, 0.69, 0.59, 0.4]},
    }
    result = trainer._identity_ap_diagnostics(branches)

    assert result["identity_target_delta"] == [0.1, 0.1, 0.1, 0.1]
    assert len(result["identity_wrong_delta"]) == 4
    assert len(result["identity_ap_delta_matrix"]) == 4
    assert all(
        target > wrong
        for target, wrong in zip(
            result["identity_target_delta"], result["identity_wrong_delta"]
        )
    )


def test_pu_gate_requires_one_valid_record_for_every_active_label():
    dynamic = {"pu_zero_exact": True, "pu_active_private_only": True}
    pu = {
        "active_labels": [3, 7],
        "lambda": [0.0] * 21,
        "labels": [
            {"label_id": 3, "eligible": True, "lcb95": 0.02, "lambda": 0.02}
        ],
    }
    pu["lambda"][3] = 0.02
    pu["lambda"][7] = 0.03
    assert not pilot._pu_gate_pass(dynamic, pu)

    pu["labels"].append(
        {"label_id": 7, "eligible": True, "lcb95": 0.03, "lambda": 0.03}
    )
    assert pilot._pu_gate_pass(dynamic, pu)

    pu["active_labels"] = [3]
    assert not pilot._pu_gate_pass(dynamic, pu)

    pu["active_labels"] = [3, 7]
    pu["labels"][1]["lambda"] = 0.02
    assert not pilot._pu_gate_pass(dynamic, pu)


def test_strict_artifact_numbers_must_be_finite():
    assert pilot._finite(0.0)
    assert not pilot._finite(math.nan)
