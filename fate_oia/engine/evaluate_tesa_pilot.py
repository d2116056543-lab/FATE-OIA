from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from fate_oia.utils.meter_artifacts import validate_epoch_artifacts, write_json


GROUNDABLE = tuple(index for index in range(21) if index not in (14, 20))


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
    recent = losses[-max(1, min(len(losses), 200)) :]
    gates = {
        "A": bool(audit.get("pass"))
        and float(
            audit["dynamic_checks"].get("progress_zero_action_error", 1.0)
        )
        < 1e-6
        and float(
            audit["dynamic_checks"].get("progress_zero_reason_error", 1.0)
        )
        < 1e-6,
        "B": (
            bool(factor_rows)
            and min(null) > 0.0
            and max(null) < 1.0
            and all(
                float(row["state_auprc"])
                > float(row["state_frequency_baseline"])
                for row in valid_states
            )
            and any(
                _finite(row.get("same_type_margin"))
                and float(row["same_type_margin"]) > 0
                for row in factor_rows
            )
        ),
        "C": (
            float(final["Act_mAP"]) >= float(visual["Act_mAP"]) + 0.005
            and float(final["Act_mF1"]) >= float(visual["Act_mF1"]) - 0.005
            and len(ratios) == 4
            and all(0.03 <= float(value) <= 0.25 for value in ratios)
            and float(final["Act_mAP"])
            > float(branches["schema_corruption"]["Act_mAP"])
        ),
        "D": (
            float(final_reason["Exp_mAP"])
            >= float(global_reason["Exp_mAP"]) - 0.002
            and bool(ground_final)
            and sum(ground_final) / len(ground_final)
            >= sum(ground_global) / len(ground_global) + 0.005
            and float(final_reason["Exp_mAP"])
            > float(branches["reason_correction_off"]["Exp_mAP"])
        ),
        "E": (
            bool(recent)
            and max(int(row.get("dense_action_coverage", 0)) for row in recent) == 4
            and max(int(row.get("dense_factor_coverage", 0)) for row in recent)
            >= 12
            and float(final["Act_mAP"])
            > float(branches["cross_sample_swap"]["Act_mAP"])
        ),
        "F": (
            int(patch.get("unique_sample_count", 0))
            >= min(128, len(_read_json(latest / "file_names_test.json")["file_names"]))
            and len(patch.get("action_coverage", [])) == 4
            and len(patch.get("factor_coverage", [])) >= 12
            and float(patch.get("selected_minus_control_mean", -1.0)) > 0
        ),
        "G": bool(audit["dynamic_checks"].get("pu_zero_exact"))
        and all(0.0 <= float(value) <= 0.15 for value in pu.get("lambda", [])),
        "H": (
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
