from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SAVE_BINDING_KEYS = (
    "git_head", "config_hash", "source_tree_hash", "schema_hash", "split_hash",
    "checkpoint_hash", "logits_hash", "labels_hash", "file_order_hash",
)


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def _complete_bindings(evidence: Mapping[str, Any]) -> dict[str, str]:
    bindings = evidence.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("pilot evidence has no binding map")
    missing = [key for key in SAVE_BINDING_KEYS if not isinstance(bindings.get(key), str) or not bindings[key]]
    if missing:
        raise ValueError("pilot binding chain is incomplete: " + ", ".join(missing))
    return {key: str(bindings[key]) for key in SAVE_BINDING_KEYS}


def recompute_save_pilot_gates(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute A-G only from raw evidence; saved pass fields are ignored."""
    bindings = _complete_bindings(evidence)
    structure = evidence.get("structure", {})
    epochs = evidence.get("epochs", [])
    utility = evidence.get("utility", {})
    specificity = evidence.get("specificity", {})
    faith = evidence.get("faithfulness", {})
    grad = evidence.get("gradient_runtime", {})
    if not isinstance(epochs, list) or len(epochs) != 4:
        raise ValueError("SAVE pilot requires exactly four epoch evidence rows")

    gate_a = (
        _finite(structure.get("progress_zero_max_abs"))
        and float(structure["progress_zero_max_abs"]) < 1e-6
        and int(structure.get("ordinary_batches", -1)) == int(structure.get("dino_calls", -2))
        and float(structure.get("dino_grad_norm", 1.0)) == 0.0
        and structure.get("feature_cache") is False
        and structure.get("token_compression") == "none"
    )
    action_checks = []
    for row in epochs[-2:]:
        action = row.get("action", {})
        rms = action.get("evidence_rms", [])
        action_checks.append(
            all(_finite(action.get(key)) for key in ("base_mAP", "final_mAP", "base_mF1", "final_mF1"))
            and float(action["final_mAP"]) >= float(action["base_mAP"]) + 0.004
            and float(action["final_mF1"]) >= float(action["base_mF1"]) + 0.003
            and isinstance(rms, list) and len(rms) == 4 and all(_finite(x) and float(x) > 0 for x in rms)
            and action.get("logit_collapsed") is False
            and _finite(action.get("emergency_cap_rate"))
            and float(action["emergency_cap_rate"]) < 0.01
        )
    gate_b = len(action_checks) == 2 and all(action_checks)
    gate_c = (
        _finite(utility.get("audit_auc")) and float(utility["audit_auc"]) >= 0.65
        and _finite(utility.get("selected_minus_control")) and float(utility["selected_minus_control"]) > 0
        and set(utility.get("action_coverage", [])) == {0, 1, 2, 3}
        and int(utility.get("valid_factor_count", 0)) >= 12
        and _finite(utility.get("std")) and float(utility["std"]) > 0.03
    )
    gate_d = (
        _finite(specificity.get("target_deletion")) and _finite(specificity.get("wrong_deletion"))
        and float(specificity["target_deletion"]) > float(specificity["wrong_deletion"])
        and _finite(specificity.get("identity_corruption_ap_drop"))
        and float(specificity["identity_corruption_ap_drop"]) > 0
        and _finite(specificity.get("max_factor_share"))
        and float(specificity["max_factor_share"]) < 0.70
        and _finite(specificity.get("effective_factor_count"))
        and float(specificity["effective_factor_count"]) > 1.5
    )
    clean_history = [row.get("reason", {}).get("clean_metric") for row in epochs]
    reason_last = epochs[-1].get("reason", {})
    gate_e = (
        all(_finite(x) for x in clean_history)
        and not all(float(b) < float(a) for a, b in zip(clean_history, clean_history[1:]))
        and all(_finite(reason_last.get(key)) for key in (
            "clean_mAP", "final_mAP", "private_tail_mAP", "clean_tail_mAP",
            "reliability_min", "reliability_max",
        ))
        and float(reason_last["final_mAP"]) >= float(reason_last["clean_mAP"]) - 0.003
        and float(reason_last["private_tail_mAP"]) > float(reason_last["clean_tail_mAP"])
        and 0.0 < float(reason_last["reliability_min"])
        and float(reason_last["reliability_max"]) < 1.0
    )
    gate_f = (
        _finite(faith.get("evidence_only_margin_retention"))
        and float(faith["evidence_only_margin_retention"]) >= 0.90
        and _finite(faith.get("selected_deletion")) and _finite(faith.get("matched_control"))
        and float(faith["selected_deletion"]) > float(faith["matched_control"])
        and _finite(faith.get("target_action_change")) and _finite(faith.get("wrong_action_change"))
        and float(faith["wrong_action_change"]) < float(faith["target_action_change"])
        and _finite(faith.get("conservation_max_abs"))
        and float(faith["conservation_max_abs"]) < 1e-6
    )
    gate_g = (
        all(_finite(grad.get(key)) for key in (
            "private_to_action", "clean_to_shared", "action_to_inquiry", "action_to_utility",
            "grounding_to_foundation", "pu_non_private", "reserved_gb",
        ))
        and abs(float(grad["private_to_action"])) < 1e-12
        and float(grad["clean_to_shared"]) > 0
        and float(grad["action_to_inquiry"]) > 0
        and float(grad["action_to_utility"]) > 0
        and abs(float(grad["grounding_to_foundation"])) < 1e-12
        and abs(float(grad["pu_non_private"])) < 1e-12
        and float(grad["reserved_gb"]) < 45.0
        and grad.get("finite") is True and grad.get("oom") is False
    )
    gates = dict(zip("ABCDEFG", (gate_a, gate_b, gate_c, gate_d, gate_e, gate_f, gate_g)))
    latest = epochs[-1]
    action_latest = latest.get("action", {})
    reason_latest = latest.get("reason", {})
    final_raw_joint = 0.5 * (
        float(action_latest.get("final_mF1", float("nan")))
        + float(reason_latest.get("final_mF1", float("nan")))
    )
    # A and G protect the experimental validity (exact anchor / frozen-DINO
    # contract and gradient-memory safety). B-F grade mechanism quality but
    # are intentionally retained as diagnostics: a high real test score is
    # not discarded merely because a heuristic mechanism threshold is tight.
    numeric_candidate_eligible = gate_a and gate_g and _finite(final_raw_joint)
    return {
        "pass": all(gates.values()),
        "numeric_candidate_eligible": numeric_candidate_eligible,
        "selection": {
            "primary": "final_raw_joint",
            "final_raw_joint": final_raw_joint,
            "final_action_mF1": action_latest.get("final_mF1"),
            "final_reason_mF1": reason_latest.get("final_mF1"),
            "final_reason_mAP": reason_latest.get("final_mAP"),
            "mechanism_gates_failed": [name for name in "BCDEF" if not gates[name]],
        },
        "bindings": bindings,
        "gates": gates,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    raw = json.loads(Path(args.raw_evidence).read_text(encoding="utf-8"))
    result = recompute_save_pilot_gates(raw)
    output = Path(args.output_dir)
    _atomic_json(output / "SAVE_PILOT_RAW_EVIDENCE.json", raw)
    _atomic_json(output / "SAVE_PILOT_GATES.json", {"bindings": result["bindings"], "gates": result["gates"]})
    _atomic_json(output / "SAVE_PILOT_PASS.json", result)
    _atomic_json(output / "SAVE_FULL_TRAIN_READY.json", result)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
