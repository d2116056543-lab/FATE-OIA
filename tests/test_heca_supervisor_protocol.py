import json

import pytest

from fate_oia.engine import supervise_meter_oia_v3_heca_foreground as supervisor
from fate_oia.engine.eval_acpr_meter_oia import (
    CHEAP_SAME_FORWARD_MODES,
    INDEPENDENT_HECA_ABLATIONS,
)
from fate_oia.utils.meter_artifacts import write_heca_artifact_sidecar


def test_full_supervisor_requires_clean_matching_a_to_g(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(supervisor, "_git", lambda *args: "head" if args[0] == "rev-parse" else "")
    review = tmp_path / "review.json"
    pilot = tmp_path / "pilot.json"
    gate_c = tmp_path / "HECA_GATE_C.json"
    gate_payloads = {
        letter: {"gate": letter, "pass": True, "evidence": {"rows": 1}}
        for letter in "ABCDEFG"
    }
    write_heca_artifact_sidecar(
        tmp_path,
        {
            "ontology_manifest": {
                "schema_version": 4,
                "factor_count": 21,
                "state_count": 63,
                "sha256": "schema",
            },
            "tau_stats": {
                "source_split": "train_main",
                "alpha": 20.0,
                "tau": [0.5] * 21,
            },
            "gradient_ownership": [{
                "optimizer_step": 1,
                "action_to_anchor_query": 0.0,
                "action_to_state_bridge_ratio": 0.05,
                "reason_to_action_credit": 0.0,
                "measurement_to_foundation": 0.0,
            }],
            "loss_wiring": {
                "registry": ["action_final"],
                "counts": {"action_final": 1},
                "duplicates": [],
                "pass": True,
            },
            "component_call_counters": {
                "components": {
                    "dino_encode": 1,
                    "typed_measurement": 1,
                    "action_credit": 1,
                    "reason_correction": 1,
                },
                "one_dino_encode_per_batch": True,
            },
            "contribution_conservation": [{
                "action": 0,
                "sum_contribution": 0.1,
                "action_credit_sum": 0.1,
                "abs_error": 0.0,
            }],
            "schedule_state": {
                "optimizer_step": 1,
                "progress": 0.1,
                "credit_ramp": 0.5,
                "foundation_grad_cap": 0.25,
                "excess_risk": {"action": 0.0, "reason": 0.0},
            },
            "ablation_manifest": {
                "cheap_same_forward": list(CHEAP_SAME_FORWARD_MODES),
                "independent_runs": INDEPENDENT_HECA_ABLATIONS,
            },
            "gates": gate_payloads,
        },
    )
    review.write_text(json.dumps({"pass": True, "git_head": "head"}), encoding="utf-8")
    pilot.write_text(json.dumps({
        "pass": True,
        "git_head": "head",
        "gates": {letter: True for letter in "ABCDEFG"},
        "gate_payloads": gate_payloads,
    }), encoding="utf-8")
    assert supervisor.validate_training_readiness(review, pilot, gate_c)["git_head"] == "head"
    payload = json.loads(pilot.read_text(encoding="utf-8"))
    payload["gates"]["E"] = False
    pilot.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete"):
        supervisor.validate_training_readiness(review, pilot, gate_c)
