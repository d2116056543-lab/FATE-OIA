from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from fate_oia.utils.meter_artifacts import validate_epoch_artifacts, write_json


GROUNDABLE = tuple(index for index in range(21) if index not in (14, 20))


def validate_pilot_protocol(
    manifest: dict[str, Any],
    expected: dict[str, Any],
    *,
    completed_epochs: int,
) -> list[str]:
    failures: list[str] = []
    if bool(manifest.get("use_mock_dino", True)):
        failures.append("mock_dino")
    if int(manifest.get("seed", -1)) != int(expected["seed"]):
        failures.append("seed")
    counts = manifest.get("runtime_subset_counts", {})
    for name in ("train_main", "train_audit", "train_calib", "test"):
        if int(counts.get(name, -1)) != int(expected[name]):
            failures.append(name)
    if int(completed_epochs) != int(expected["epochs"]):
        failures.append("epochs")
    return failures


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _pu_gate_pass(dynamic: dict[str, Any], pu: dict[str, Any]) -> bool:
    active = [int(value) for value in pu.get("active_labels", [])]
    raw_rows = [
        row for row in pu.get("labels", []) if isinstance(row, dict)
    ]
    rows = {
        int(row["label_id"]): row
        for row in raw_rows
        if "label_id" in row
    }
    lambdas = pu.get("lambda", [])
    lambda_active = {
        label
        for label, value in enumerate(lambdas)
        if float(value) > 0.0
    }
    return (
        bool(dynamic.get("pu_zero_exact"))
        and bool(dynamic.get("pu_active_private_only"))
        and bool(active)
        and len(lambdas) == 21
        and all(0.0 <= float(value) <= 0.15 for value in lambdas)
        and len(active) == len(set(active))
        and set(active) == lambda_active
        and len(rows) == len(raw_rows)
        and all(label in rows for label in active)
        and all(
            bool(rows[label].get("eligible"))
            and float(rows[label].get("lcb95", 0.0)) > 0.0
            and float(rows[label].get("lambda", 0.0)) > 0.0
            and float(lambdas[label]) > 0.0
            and abs(
                float(rows[label].get("lambda", 0.0))
                - float(lambdas[label])
            )
            < 1e-8
            for label in active
        )
    )


def evaluate_pilot(
    run_dir: str | Path, implementation_audit: str | Path
) -> dict[str, Any]:
    root = Path(run_dir)
    epochs = sorted(root.glob("epoch_*"))
    if not epochs:
        raise FileNotFoundError("No pilot epoch artifacts were found")
    latest = epochs[-1]
    branches = _read_json(latest / "branch_metrics.json")
    typed = _read_json(latest / "typed_evidence.json")
    pu = _read_json(latest / "pu_stats.json")
    runtime = _read_json(latest / "runtime.json")
    manifest = _read_json(root / "run_manifest.json")
    protocol_failures = validate_pilot_protocol(
        manifest, manifest["config"]["pilot"], completed_epochs=len(epochs)
    )
    audit = _read_json(Path(implementation_audit))
    losses = _read_jsonl(root / "loss_components.jsonl")
    visual = branches["action_visual"]
    final = branches["action_final"]
    global_reason = branches["reason_global"]
    final_reason = branches["reason_final"]
    ratios = typed["action_correction_rms_ratio_per_action"]
    null = typed["anchor_null_mass_per_factor"]
    factor_rows = typed.get("train_audit", {}).get("per_factor", [])
    valid_states = [
        row
        for row in factor_rows
        if _finite(row.get("state_auprc"))
        and _finite(row.get("state_frequency_baseline"))
    ]
    ground_global = [
        global_reason["Exp_per_label_ap"][index]
        for index in GROUNDABLE
        if _finite(global_reason["Exp_per_label_ap"][index])
    ]
    ground_final = [
        final_reason["Exp_per_label_ap"][index]
        for index in GROUNDABLE
        if _finite(final_reason["Exp_per_label_ap"][index])
    ]
    patch = typed.get("patch_audit", {})
    identity_target = typed.get("identity_target_delta", [])
    identity_wrong = typed.get("identity_wrong_delta", [])
    schema_reason = branches.get("schema_corruption", {})
    schema_reason_ap = schema_reason.get("Exp_per_label_ap", [])
    reason_identity_delta = typed.get("reason_identity_delta_per_label", [])
    recent = losses[-max(1, min(len(losses), 200)) :]
    gates = {
        "A": (
            bool(audit.get("pass"))
            and float(
                audit["dynamic_checks"].get("progress_zero_action_error", 1.0)
            )
            < 1e-6
            and float(
                audit["dynamic_checks"].get("progress_zero_reason_error", 1.0)
            )
            < 1e-6
            and float(
                audit["dynamic_checks"].get(
                    "progress_zero_label_node_error", 1.0
                )
            )
            < 1e-6
        ),
        "B": (
            bool(factor_rows)
            and min(null) > 0.0
            and max(null) < 1.0
            and len(valid_states) >= 12
            and all(
                float(row["state_auprc"])
                > float(row["state_frequency_baseline"])
                for row in valid_states
            )
            and sum(
                _finite(row.get("same_type_margin"))
                and float(row["same_type_margin"]) > 0
                for row in factor_rows
            ) >= 12
            and sum(
                _finite(row.get("mirror_equivariance"))
                and float(row["mirror_equivariance"]) > 0
                for row in factor_rows
            ) >= 8
        ),
        "C": (
            float(final["Act_mAP"]) >= float(visual["Act_mAP"]) + 0.005
            and float(final["Act_mF1"]) >= float(visual["Act_mF1"]) - 0.005
            and len(ratios) == 4
            and all(0.03 <= float(value) <= 0.25 for value in ratios)
            and float(final["Act_mAP"])
            > float(branches["schema_corruption"]["Act_mAP"])
            and len(identity_target) == 4
            and len(identity_wrong) == 4
            and sum(
                float(target) >= float(wrong) + 0.001
                for target, wrong in zip(identity_target, identity_wrong)
            )
            == 4
        ),
        "D": (
            float(final_reason["Exp_mAP"])
            >= float(global_reason["Exp_mAP"]) - 0.002
            and bool(ground_final)
            and sum(ground_final) / len(ground_final)
            >= sum(ground_global) / len(ground_global) + 0.005
            and float(final_reason["Exp_mAP"])
            > float(branches["reason_correction_off"]["Exp_mAP"])
            and len(reason_identity_delta) == 21
            and all(
                _finite(reason_identity_delta[index]) for index in GROUNDABLE
            )
            and all(
                float(reason_identity_delta[index]) >= 0.001
                for index in GROUNDABLE
            )
        ),
        "E": (
            bool(recent)
            and max(int(row.get("dense_action_coverage", 0)) for row in recent) == 4
            and max(int(row.get("dense_factor_coverage", 0)) for row in recent)
            >= 12
            and sum(
                float(row.get("dense_correct_effect_abs", 0.0))
                > float(row.get("dense_wrong_effect_abs", 0.0))
                for row in recent
            )
            / len(recent)
            >= 0.75
            and len(typed.get("schema_corruption_delta_per_action", [])) == 4
            and all(
                float(value) > 0
                for value in typed["schema_corruption_delta_per_action"]
            )
            and float(final["Act_mAP"])
            > float(branches["cross_sample_swap"]["Act_mAP"])
        ),
        "F": (
            int(patch.get("unique_sample_count", 0))
            >= min(128, len(_read_json(latest / "file_names_test.json")["file_names"]))
            and len(patch.get("action_coverage", [])) == 4
            and len(patch.get("factor_coverage", [])) >= 12
            and float(patch.get("selected_positive_rate", 0.0)) > 0.5
            and float(patch.get("selected_minus_control_mean", -1.0)) > 0
        ),
        "G": _pu_gate_pass(audit["dynamic_checks"], pu),
        "H": (
            not protocol_failures
            and
            not validate_epoch_artifacts(latest)
            and all(int(row.get("dino_call_count", 0)) == 1 for row in recent)
            and float(runtime.get("peak_reserved_gb", 99.0)) < 45.0
            and all(
                _finite(row.get("loss_total"))
                and _finite(row.get("grad_norm"))
                for row in recent
            )
        ),
    }
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = {
        "pass": all(gates.values()),
        "git_head": head,
        "epoch": int(latest.name.split("_")[-1]),
        "gates": gates,
        "gate_inputs": {
            "action_visual": visual,
            "action_final": final,
            "reason_global": global_reason,
            "reason_final": final_reason,
            "correction_rms_ratio": ratios,
            "null_mass": null,
            "valid_state_rows": valid_states,
            "patch_audit": patch,
            "pu_active_labels": pu.get("active_labels", []),
            "runtime": runtime,
        },
        "artifact_missing": validate_epoch_artifacts(latest),
        "protocol_failures": protocol_failures,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--implementation_audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_pilot(args.run_dir, args.implementation_audit)
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
