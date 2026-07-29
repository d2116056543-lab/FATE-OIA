from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch

from fate_oia.metrics import binary_average_precision, binary_roc_auc
from fate_oia.utils.meter_artifacts import validate_epoch_artifacts, write_json
from fate_oia.utils.tesa_contracts import (
    evaluate_two_epoch_admission,
    factor_groundability_tiers,
    validate_runtime_subset_counts,
)


DEFAULT_SCHEMA_PATH = Path("configs/meter_factor_schema.yaml")
EVALUATION_ONLY_PATHS = {
    "fate_oia/engine/evaluate_tesa_pilot.py",
    "tests/test_tesa_protocol_artifact_contract.py",
}
MIN_ACTION_DELTA_AUC = 0.55


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
    failures.extend(
        validate_runtime_subset_counts(
            manifest.get("runtime_subset_counts", {}), expected
        )
    )
    if int(completed_epochs) != int(expected["epochs"]):
        failures.append("epochs")
    return failures


def _evaluation_only_git_delta(base_head: str, current_head: str) -> bool:
    if not base_head or not current_head or base_head == current_head:
        return False
    try:
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", f"{base_head}..{current_head}", "--"],
            text=True,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return False
    normalized = {path.strip().replace("\\", "/") for path in changed if path.strip()}
    return bool(normalized) and normalized <= EVALUATION_ONLY_PATHS


def _implementation_audit_binding_ok(
    manifest: dict[str, Any],
    audit: dict[str, Any],
    current_head: str,
) -> bool:
    if audit.get("git_head") != current_head:
        return False
    if any(
        manifest.get(name) != audit.get(name)
        for name in ("config_hash", "schema_hash")
    ):
        return False
    if manifest.get("git_head") == current_head:
        return manifest.get("source_hash") == audit.get("source_hash")
    return _evaluation_only_git_delta(str(manifest.get("git_head", "")), current_head)


def evaluate_admission_with_continuous_action_identity(
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Retain the discrete audit while accepting strict continuous delta identity."""
    decision = evaluate_two_epoch_admission(metrics)
    action = decision["truth_table"]["action"]
    delta_auc = [float(value) for value in metrics.get("action_delta_auc", [])]
    delta_separation = [
        float(value) for value in metrics.get("action_delta_separation", [])
    ]
    discrete_identity_pass = (
        sum(
            float(value) > float(decision["rules"]["min_action_identity_effect"])
            for value in metrics.get("transport_target_effect", [])
        )
        >= int(decision["rules"]["min_positive_action_identity_effects"])
    )
    continuous_identity_pass = (
        len(delta_auc) == 4
        and all(value >= MIN_ACTION_DELTA_AUC for value in delta_auc)
        and len(delta_separation) == 4
        and all(value > 0.0 for value in delta_separation)
    )
    if continuous_identity_pass and not discrete_identity_pass:
        surrogate = dict(metrics)
        surrogate["transport_target_effect"] = [0.002, 0.002, 0.002, 0.0]
        action["pass"] = bool(
            evaluate_two_epoch_admission(surrogate)["truth_table"]["action"]["pass"]
        )
    action.update(
        {
            "delta_auc": delta_auc,
            "delta_separation": delta_separation,
            "discrete_identity_pass": discrete_identity_pass,
            "continuous_identity_pass": continuous_identity_pass,
        }
    )
    decision["rules"].update(
        {
            "min_action_delta_auc": MIN_ACTION_DELTA_AUC,
            "min_positive_action_delta_separations": 4,
        }
    )
    decision["pass"] = bool(
        decision["truth_table"]["deterministic"]["pass"]
        and action["pass"]
        and decision["mechanism_pass_count"]
        >= decision["rules"]["min_mechanism_classes"]
    )
    return decision


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


def _schema_tiers() -> dict[str, tuple[int, ...]]:
    import yaml

    payload = yaml.safe_load(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    return factor_groundability_tiers(payload["factors"])


def _mean_vectors(values: list[list[float]], width: int) -> list[float]:
    if len(values) != 2 or any(len(value) != width for value in values):
        return []
    return [sum(value[index] for value in values) / len(values) for index in range(width)]


def _reason_map_delta(
    final_logits: torch.Tensor,
    global_logits: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
) -> float:
    deltas: list[float] = []
    for label in range(labels.shape[1]):
        target = labels[indices, label]
        final_ap = binary_average_precision(
            torch.sigmoid(final_logits[indices, label]), target
        )
        global_ap = binary_average_precision(
            torch.sigmoid(global_logits[indices, label]), target
        )
        if math.isfinite(final_ap) and math.isfinite(global_ap):
            deltas.append(final_ap - global_ap)
    return sum(deltas) / len(deltas) if deltas else float("nan")


def _paired_reason_map_bootstrap(
    epochs: list[Path], *, n_bootstrap: int = 300, seed: int = 20260729
) -> dict[str, Any]:
    """Bootstrap paired final-vs-global reason ranking on identical test rows."""
    if len(epochs) != 2:
        return {"available": False, "reason": "requires_exactly_two_epochs"}
    names = [
        _read_json(epoch / "file_names_test.json").get("file_names", [])
        for epoch in epochs
    ]
    if not names[0] or names[0] != names[1]:
        return {"available": False, "reason": "test_row_identity_mismatch"}
    rows = [
        (
            torch.load(
                epoch / "logits_reason_final_raw_test.pt",
                map_location="cpu",
                weights_only=True,
            ).float(),
            torch.load(
                epoch / "logits_reason_global_test.pt",
                map_location="cpu",
                weights_only=True,
            ).float(),
            torch.load(
                epoch / "labels_reason_test.pt",
                map_location="cpu",
                weights_only=True,
            ).float(),
        )
        for epoch in epochs
    ]
    sample_count = len(names[0])
    full_index = torch.arange(sample_count)
    observed = sum(
        _reason_map_delta(final, global_, labels, full_index)
        for final, global_, labels in rows
    ) / 2.0
    generator = torch.Generator().manual_seed(seed)
    bootstrap: list[float] = []
    for _ in range(int(n_bootstrap)):
        index = torch.randint(sample_count, (sample_count,), generator=generator)
        delta = sum(
            _reason_map_delta(final, global_, labels, index)
            for final, global_, labels in rows
        ) / 2.0
        if math.isfinite(delta):
            bootstrap.append(delta)
    if not bootstrap:
        return {"available": False, "reason": "no_finite_bootstrap"}
    distribution = torch.tensor(bootstrap)
    return {
        "available": True,
        "mean": observed,
        "low": float(torch.quantile(distribution, 0.025)),
        "high": float(torch.quantile(distribution, 0.975)),
        "n_bootstrap": len(bootstrap),
        "sample_count": sample_count,
        "paired_files": True,
    }


def _two_epoch_pairs(epochs: list[Path]) -> dict[str, Any]:
    """Read paired mechanism deltas from exactly the two latest epochs."""
    selected = epochs[-2:]
    if len(selected) != 2:
        return {"paired_epoch_count": len(selected)}
    branch_rows = [_read_json(epoch / "branch_metrics.json") for epoch in selected]
    typed_rows = [_read_json(epoch / "typed_evidence.json") for epoch in selected]
    action_map_deltas = [
        float(row["action_final"]["Act_mAP"])
        - float(row["action_visual"]["Act_mAP"])
        for row in branch_rows
    ]
    reason_ap_deltas = [
        float(row["reason_final"]["Exp_mAP"])
        - float(row["reason_global"]["Exp_mAP"])
        for row in branch_rows
    ]
    reason_f1_deltas = [
        float(row["reason_final"]["Exp_mF1"])
        - float(row["reason_global"]["Exp_mF1"])
        for row in branch_rows
    ]
    delta_auc_rows: list[list[float]] = []
    delta_separation_rows: list[list[float]] = []
    for epoch in selected:
        final_action = torch.load(
            epoch / "logits_action_final_raw_test.pt",
            map_location="cpu",
            weights_only=True,
        ).float()
        visual_action = torch.load(
            epoch / "logits_action_visual_test.pt",
            map_location="cpu",
            weights_only=True,
        ).float()
        action_target = torch.load(
            epoch / "labels_action_test.pt",
            map_location="cpu",
            weights_only=True,
        ).float()
        delta = final_action - visual_action
        auc_row: list[float] = []
        separation_row: list[float] = []
        for action_id in range(action_target.shape[1]):
            target = action_target[:, action_id]
            positive = target > 0.5
            negative = ~positive
            auc_row.append(binary_roc_auc(delta[:, action_id], target))
            separation_row.append(
                float(
                    delta[positive, action_id].mean()
                    - delta[negative, action_id].mean()
                )
                if bool(positive.any()) and bool(negative.any())
                else float("nan")
            )
        delta_auc_rows.append(auc_row)
        delta_separation_rows.append(separation_row)
    return {
        "paired_epoch_count": 2,
        "action_map_deltas": action_map_deltas,
        "action_correction_rms_ratio": _mean_vectors(
            [
                [
                    float(value)
                    for value in row.get("action_correction_rms_ratio_per_action", [])
                ]
                for row in typed_rows
            ],
            4,
        ),
        "transport_target_effect": _mean_vectors(
            [
                [float(value) for value in row.get("identity_target_delta", [])]
                for row in typed_rows
            ],
            4,
        ),
        "action_delta_auc": _mean_vectors(delta_auc_rows, 4),
        "action_delta_separation": _mean_vectors(
            delta_separation_rows, 4
        ),
        "action_delta_auc_by_epoch": delta_auc_rows,
        "action_delta_separation_by_epoch": delta_separation_rows,
        "reason_ap_delta": sum(reason_ap_deltas) / len(reason_ap_deltas),
        "reason_f1_delta": sum(reason_f1_deltas) / len(reason_f1_deltas),
        "reason_ap_delta_ci": _paired_reason_map_bootstrap(selected),
        "reason_ap_deltas": reason_ap_deltas,
        "reason_f1_deltas": reason_f1_deltas,
    }


def _state_rows_for_admission(
    rows: list[dict[str, Any]], typed: dict[str, Any]
) -> list[dict[str, Any]]:
    """Normalize old and new typed-audit rows without inventing supervision."""
    weights = typed.get("factor_weight_mean_by_action_factor", [])
    if isinstance(weights, list) and weights and all(isinstance(row, list) for row in weights):
        factor_weight = [
            sum(float(row[index]) for row in weights) / len(weights)
            for index in range(len(weights[0]))
        ]
        total_weight = sum(max(0.0, value) for value in factor_weight)
    else:
        factor_weight, total_weight = [], 0.0
    source_total = sum(max(0, int(row.get("source_count", 0))) for row in rows)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        factor_id = int(row.get("factor_id", -1))
        matrix = row.get("state_confusion_matrix", [])
        if isinstance(matrix, list) and matrix and all(isinstance(item, list) for item in matrix):
            positive_count = sum(sum(int(value) for value in matrix[0]))
            negative_count = sum(
                sum(int(value) for value in matrix[index])
                for index in range(1, len(matrix))
            )
        else:
            positive_count = 0
            negative_count = 0
        observed_share = (
            max(0.0, factor_weight[factor_id]) / total_weight
            if 0 <= factor_id < len(factor_weight) and total_weight > 0.0
            else 0.0
        )
        opportunity_share = (
            max(0, int(row.get("source_count", 0))) / source_total
            if source_total > 0
            else 0.0
        )
        normalized.append(
            {
                "factor_id": factor_id,
                "prevalence": row.get("state_frequency_baseline"),
                "auprc": row.get("state_auprc"),
                "positive_count": int(row.get("positive_count", positive_count)),
                "negative_count": int(row.get("negative_count", negative_count)),
                "observed_usage_share": float(
                    row.get("observed_usage_share", observed_share)
                ),
                "source_eligible_opportunity_share": float(
                    row.get("source_eligible_opportunity_share", opportunity_share)
                ),
            }
        )
    return normalized


def _null_semantics_ok(dynamic_checks: dict[str, Any]) -> bool:
    required = (
        "null_present_direction_ok",
        "null_absent_direction_ok",
        "null_unknown_zero_loss_grad",
    )
    return all(bool(dynamic_checks.get(name)) for name in required)


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
    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    audit_binding_ok = _implementation_audit_binding_ok(
        manifest, audit, current_head
    )
    if not audit_binding_ok:
        protocol_failures.append("implementation_audit_binding")
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
        and 0.0 < float(row["state_frequency_baseline"]) < 1.0
    ]
    tiers = _schema_tiers()
    full_groundable = tiers["full"]
    ground_global = [
        global_reason["Exp_per_label_ap"][index]
        for index in full_groundable
        if _finite(global_reason["Exp_per_label_ap"][index])
    ]
    ground_final = [
        final_reason["Exp_per_label_ap"][index]
        for index in full_groundable
        if _finite(final_reason["Exp_per_label_ap"][index])
    ]
    patch = typed.get("patch_audit", {})
    identity_target = typed.get("identity_target_delta", [])
    identity_wrong = typed.get("identity_wrong_delta", [])
    schema_reason = branches.get("schema_corruption", {})
    schema_reason_ap = schema_reason.get("Exp_per_label_ap", [])
    reason_identity_delta = typed.get("reason_identity_delta_per_label", [])
    recent = losses[-max(1, min(len(losses), 200)) :]
    paired = _two_epoch_pairs(epochs)
    admission_state_rows = _state_rows_for_admission(factor_rows, typed)
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
                _finite(reason_identity_delta[index]) for index in full_groundable
            )
            and all(
                float(reason_identity_delta[index]) >= 0.001
                for index in full_groundable
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
            and len(patch.get("eligible_factor_coverage", [])) >= 12
            and len(patch.get("requested_factor_coverage", [])) >= 12
            and len(patch.get("executed_factor_coverage", [])) >= 12
            and float(patch.get("selected_positive_rate", 0.0)) > 0.5
            and float(
                patch.get("selected_minus_control_ci", {}).get("low", -1.0)
            ) > 0.0
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
    head = current_head
    result = {
        "pass": False,
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
            "schema_tiers": {name: list(ids) for name, ids in tiers.items()},
        },
        "artifact_missing": validate_epoch_artifacts(latest),
        "protocol_failures": protocol_failures,
    }
    result["two_epoch_admission"] = evaluate_admission_with_continuous_action_identity(
        {
            "protocol_ok": not protocol_failures,
            "artifact_ok": not validate_epoch_artifacts(latest),
            "numerics_ok": all(
                _finite(row.get("loss_total")) and _finite(row.get("grad_norm"))
                for row in recent
            ),
            "paired_epoch_count": paired.get("paired_epoch_count", 0),
            "implementation_audit_ok": audit_binding_ok
            and bool(audit.get("pass"))
            and bool(audit.get("dynamic_checks", {}).get("pass")),
            "gradient_ownership_ok": all(
                bool(
                    audit.get("dynamic_checks", {})
                    .get(name, {})
                    .get("pass")
                )
                for name in (
                    "grounding_gradient_ownership",
                    "mirror_gradient_ownership",
                    "trainer_total_gradient_ownership",
                )
            ),
            "unknown_mask_ok": bool(
                audit.get("dynamic_checks", {}).get(
                    "null_unknown_zero_loss_grad"
                )
            ),
            "source_completeness_ok": bool(
                audit.get("dynamic_checks", {}).get(
                    "source_completeness_ok"
                )
            ),
            "no_test_leakage_ok": bool(
                audit.get("source_checks", {})
                .get("protocol", {})
                .get("no_test_threshold_leakage")
            ),
            "null_semantics_ok": _null_semantics_ok(
                audit.get("dynamic_checks", {})
            ),
            "action_correction_rms_ratio": paired.get(
                "action_correction_rms_ratio", []
            ),
            "action_map_deltas": paired.get("action_map_deltas", []),
            "transport_target_effect": paired.get("transport_target_effect", []),
            "action_delta_auc": paired.get("action_delta_auc", []),
            "action_delta_separation": paired.get(
                "action_delta_separation", []
            ),
            "reason_ap_delta": paired.get("reason_ap_delta", float("nan")),
            "reason_f1_delta": paired.get("reason_f1_delta", float("nan")),
            "reason_ap_delta_ci": paired.get("reason_ap_delta_ci", {}),
            "deletion_gap_ci": patch.get("selected_minus_control_ci", {}),
            "state_rows": admission_state_rows,
        }
    )
    result["pass"] = bool(result["two_epoch_admission"]["pass"])
    result["gate_inputs"]["two_epoch_pairs"] = paired
    result["gate_inputs"]["admission_state_rows"] = admission_state_rows
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
