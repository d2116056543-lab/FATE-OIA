from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

from fate_oia.engine.eval_acpr_meter_oia import (
    CHEAP_SAME_FORWARD_MODES,
    INDEPENDENT_HECA_ABLATIONS,
)
from fate_oia.utils.meter_artifacts import (
    file_hash,
    validate_heca_pilot_bundle,
    write_heca_artifact_sidecar,
    write_heca_pilot_evidence_manifest,
    write_json,
)


ACTION_STATE_EFFECT_MATURITY_FLOOR = 0.10


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected non-empty JSONL: {path}")
    return rows


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and float("-inf") < float(value) < float("inf")


def _gate(letter: str, passed: bool, evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {"gate": letter, "pass": bool(passed), "evidence": dict(evidence)}


def _ownership_gate_rows(
    gradient_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[bool]]:
    """Evaluate ownership only after the scheduled action-credit path is live.

    The state effect is intentionally zero at initialization to preserve the
    visual action anchor.  Its pre-ramp probe is useful telemetry but cannot
    establish an active-path gradient ownership violation.  Once the ramp is
    active, the original 1--10 percent firewall remains a hard requirement.
    Every row must contain finite ramp and state-effect telemetry.  Rows before
    the scheduled ramp are excluded only after that evidence is present; old or
    malformed artifacts fail closed rather than inheriting an active default.
    """
    active_rows: list[dict[str, Any]] = []
    checks: list[bool] = []
    # Full ramp alone does not mean a zero-initialized state-effect table has
    # become an active transport route.  A route is mature only after its
    # effect norm reaches the configured structural floor.  Earlier rows stay
    # in the artifact for diagnosis but cannot falsely fail a mature-path
    # ownership audit merely because the route had not learned any effect yet.
    for row in gradient_rows:
        ramp = row.get("action_credit_ramp")
        state_effect_norm = row.get("action_state_effect_norm")
        if (
            not _finite(ramp)
            or not _finite(state_effect_norm)
            or not 0.0 <= float(ramp) <= 1.0
            or float(state_effect_norm) < 0.0
        ):
            checks.append(False)
            continue
        if float(ramp) < 0.80:
            continue
        if float(state_effect_norm) < ACTION_STATE_EFFECT_MATURITY_FLOOR:
            continue
        active_rows.append(row)
        required = (
            "action_to_anchor_query",
            "action_to_state_bridge_ratio",
            "action_to_credit_adapter",
            "reason_to_action_credit",
            "pu_to_action_factor",
            "measurement_to_foundation",
        )
        if not all(_finite(row.get(name)) for name in required):
            checks.append(False)
            continue
        checks.append(
            abs(float(row["action_to_anchor_query"])) <= 1e-12
            and 0.01 <= float(row["action_to_state_bridge_ratio"]) <= 0.10
            and float(row["action_to_credit_adapter"]) > 0.0
            and abs(float(row["reason_to_action_credit"])) <= 1e-12
            and abs(float(row["pu_to_action_factor"])) <= 1e-12
            and abs(float(row["measurement_to_foundation"])) <= 1e-12
        )
    return active_rows, checks


def evaluate_heca_pilot(
    *,
    epochs: list[dict[str, Any]],
    implementation_audit: dict[str, Any],
    ontology_manifest: dict[str, Any],
    tau_stats: dict[str, Any],
    gradient_rows: list[dict[str, Any]],
    git_head: str,
) -> dict[str, Any]:
    """Evaluate the real four-epoch pilot; missing evidence fails closed."""
    if len(epochs) != 4:
        raise ValueError("HECA pilot requires exactly four completed epochs")
    last = epochs[-1]
    branches = last.get("branches", {})
    typed = last.get("typed", {})
    runtime = last.get("runtime", {})
    dynamic = implementation_audit.get("dynamic_checks", {}).get("checks", {})

    gate_a_checks = {
        name: dynamic.get(name) is True
        for name in (
            "action_progress_zero_equivalence",
            "reason_progress_zero_equivalence",
            "label_nodes_progress_zero_equivalence",
        )
    }
    gate_a = _gate(
        "A",
        implementation_audit.get("pass") is True
        and implementation_audit.get("git_head") == git_head
        and all(gate_a_checks.values()),
        {"checks": gate_a_checks, "audit_git_head": implementation_audit.get("git_head")},
    )

    per_factor = typed.get("train_audit", {}).get("per_factor", [])
    quality_ids = []
    identifiable_ids = []
    provenance_coverage_ids = []
    for row in per_factor if isinstance(per_factor, list) else []:
        if not isinstance(row, dict):
            continue
        provenance_valid_count = int(row.get("provenance_valid_count", 0))
        if provenance_valid_count >= 20:
            provenance_coverage_ids.append(int(row.get("factor_id", -1)))
        values = (
            row.get("same_type_margin"),
            row.get("state_auprc"),
            row.get("state_frequency_baseline"),
            row.get("state_auc"),
            row.get("visual_confidence_mean"),
            row.get("visual_confidence_std"),
        )
        state_positive_count = int(row.get("state_positive_count", 0))
        state_negative_count = int(row.get("state_negative_count", 0))
        state_identifiable = (
            row.get("audit_split") == "train_audit"
            and row.get("state_identifiable") is True
            and state_positive_count >= 20
            and state_negative_count >= 20
        )
        if state_identifiable:
            identifiable_ids.append(int(row.get("factor_id", -1)))
        if (
            int(row.get("source_count", 0)) > 0
            and state_identifiable
            and all(_finite(value) for value in values)
            and float(row["same_type_margin"]) > 0.0
            and float(row["state_auprc"])
            > float(row["state_frequency_baseline"]) + 0.001
            and float(row["state_auc"]) > 0.501
            and provenance_valid_count >= 20
            and float(row["visual_confidence_std"]) > 0.01
        ):
            quality_ids.append(int(row["factor_id"]))
    provenance_counts = tau_stats.get(
        "provenance_valid_count", tau_stats.get("valid_count", [])
    )
    provenance_stats_valid = (
        tau_stats.get("source_split") == "train_main"
        and isinstance(provenance_counts, list)
        and len(provenance_counts) == 21
        and all(_finite(value) and float(value) >= 0.0 for value in provenance_counts)
    )
    gate_b = _gate(
        "B",
        len(identifiable_ids) >= 8
        and len(quality_ids) >= max(8, math.ceil(0.75 * len(identifiable_ids)))
        and provenance_stats_valid
        and ontology_manifest.get("factor_count") == 21,
        {
            "quality_factor_ids": quality_ids,
            "quality_factor_count": len(quality_ids),
            "identifiable_factor_ids": identifiable_ids,
            "identifiable_factor_count": len(identifiable_ids),
            "required_quality_factor_count": max(
                8, math.ceil(0.75 * len(identifiable_ids))
            ),
            "provenance_coverage_factor_ids": provenance_coverage_ids,
            "provenance_coverage_factor_count": len(provenance_coverage_ids),
            "provenance_stats_valid": provenance_stats_valid,
        },
    )

    gate_c_epochs = []
    for epoch in epochs[-2:]:
        current = epoch.get("branches", {})
        visual = current.get("action_visual", {})
        final = current.get("action_final", {})
        factor_off = current.get("factor_off", {})
        state_uniform = current.get("state_uniform", {})
        ratio = epoch.get("typed", {}).get("action_correction_rms_ratio_mean", [])
        final_per_action = final.get("Act_per_label_ap", [])
        uniform_per_action = state_uniform.get("Act_per_label_ap", [])
        state_sensitive_actions = [
            action
            for action, (clean_ap, changed_ap) in enumerate(
                zip(final_per_action, uniform_per_action)
            )
            if _finite(clean_ap)
            and _finite(changed_ap)
            and abs(float(clean_ap) - float(changed_ap)) >= 0.001
        ]
        passed = (
            all(_finite(value) for value in (
                visual.get("Act_mAP"), final.get("Act_mAP"),
                visual.get("Act_mF1"), final.get("Act_mF1"),
                factor_off.get("Act_mAP"),
            ))
            and float(final["Act_mAP"]) >= float(visual["Act_mAP"]) + 0.005
            and float(final["Act_mF1"]) >= float(visual["Act_mF1"]) + 0.003
            and float(factor_off["Act_mAP"]) < float(final["Act_mAP"])
            and len(state_sensitive_actions) >= 3
            and isinstance(ratio, list)
            and len(ratio) == 4
            and all(_finite(value) and 0.03 <= float(value) <= 0.20 for value in ratio)
        )
        gate_c_epochs.append({
            "pass": passed,
            "action_map_gain": (
                float(final["Act_mAP"]) - float(visual["Act_mAP"])
                if _finite(final.get("Act_mAP")) and _finite(visual.get("Act_mAP"))
                else None
            ),
            "action_mf1_gain": (
                float(final["Act_mF1"]) - float(visual["Act_mF1"])
                if _finite(final.get("Act_mF1")) and _finite(visual.get("Act_mF1"))
                else None
            ),
            "credit_rms_ratio": ratio,
            "state_sensitive_actions": state_sensitive_actions,
        })
    gate_c = _gate("C", all(row["pass"] for row in gate_c_epochs), {"epochs": gate_c_epochs})

    anchor = branches.get("reason_calalign", {})
    global_branch = branches.get("reason_global", {})
    final_reason = branches.get("reason_final", {})
    global_ap = global_branch.get("Exp_per_label_ap", [])
    final_ap = final_reason.get("Exp_per_label_ap", [])
    positive_labels = [
        index
        for index, (before, after) in enumerate(zip(global_ap, final_ap))
        if index not in {14, 20}
        and _finite(before)
        and _finite(after)
        and float(after) > float(before)
    ]
    groundable_gain = (
        sum(float(final_ap[index]) - float(global_ap[index]) for index in range(21) if index not in {14, 20})
        / 19.0
        if len(global_ap) == 21 and len(final_ap) == 21
        else float("-inf")
    )
    gate_d_pass = (
        all(_finite(value) for value in (
            anchor.get("Exp_mAP"),
            global_branch.get("Exp_mAP"),
            final_reason.get("Exp_mAP"),
            groundable_gain,
        ))
        and float(global_branch["Exp_mAP"]) >= float(anchor["Exp_mAP"]) - 0.003
        and groundable_gain >= 0.005
        and float(final_reason["Exp_mAP"]) >= float(global_branch["Exp_mAP"]) - 0.001
        and len(positive_labels) >= 10
    )
    gate_d = _gate(
        "D",
        gate_d_pass,
        {
            "global_vs_anchor": (
                float(global_branch["Exp_mAP"]) - float(anchor["Exp_mAP"])
                if _finite(global_branch.get("Exp_mAP")) and _finite(anchor.get("Exp_mAP"))
                else None
            ),
            "groundable_map_gain": groundable_gain if _finite(groundable_gain) else None,
            "positive_groundable_labels": positive_labels,
        },
    )

    active_ownership_rows, ownership_checks = _ownership_gate_rows(gradient_rows)
    post_ramp_rows = [
        row
        for row in gradient_rows
        if _finite(row.get("action_credit_ramp"))
        and float(row["action_credit_ramp"]) >= 0.80
    ]
    immature_post_ramp_rows = [
        row
        for row in post_ramp_rows
        if not _finite(row.get("action_state_effect_norm"))
        or float(row["action_state_effect_norm"])
        < ACTION_STATE_EFFECT_MATURITY_FLOOR
    ]
    gate_e = _gate(
        "E",
        len(active_ownership_rows) >= 2
        and len(ownership_checks) == len(active_ownership_rows)
        and all(ownership_checks),
        {
            "row_count": len(gradient_rows),
            "post_ramp_row_count": len(post_ramp_rows),
            "active_row_count": len(active_ownership_rows),
            "immature_post_ramp_row_count": len(immature_post_ramp_rows),
            "state_effect_maturity_floor": ACTION_STATE_EFFECT_MATURITY_FLOOR,
            "ignored_pre_ramp_row_count": len(gradient_rows) - len(active_ownership_rows),
            "all_active_rows_pass": (
                len(ownership_checks) == len(active_ownership_rows)
                and bool(ownership_checks)
                and all(ownership_checks)
            ),
            "minimum_active_rows": 2,
        },
    )

    patch = typed.get("patch_audit", {})
    target = typed.get("identity_target_delta", [])
    wrong = typed.get("identity_wrong_delta", [])
    ci = patch.get("selected_minus_control_ci", {}) if isinstance(patch, dict) else {}
    state_target_gt_wrong = (
        isinstance(target, list)
        and isinstance(wrong, list)
        and len(target) == len(wrong) == 4
        and all(_finite(value) for value in target + wrong)
        and sum(map(float, target)) / 4.0 > sum(map(float, wrong)) / 4.0
    )
    gate_f = _gate(
        "F",
        isinstance(patch, dict)
        and int(patch.get("unique_sample_count", 0)) >= 512
        and set(patch.get("action_coverage", [])) == {0, 1, 2, 3}
        and len(set(patch.get("factor_coverage", []))) >= 12
        and _finite(patch.get("selected_minus_control_mean"))
        and float(patch["selected_minus_control_mean"]) > 0.0
        and _finite(ci.get("low"))
        and float(ci["low"]) >= 0.0
        and state_target_gt_wrong,
        {
            "unique_sample_count": patch.get("unique_sample_count"),
            "action_coverage": patch.get("action_coverage"),
            "factor_coverage": patch.get("factor_coverage"),
            "selected_minus_control_mean": patch.get("selected_minus_control_mean"),
            "bootstrap_lower_bound": ci.get("low"),
            "state_target_effect_gt_wrong": state_target_gt_wrong,
        },
    )

    dino = runtime.get("dino_call_count", {})
    gate_g = _gate(
        "G",
        all(_finite(epoch.get("max_action_logit")) and float(epoch["max_action_logit"]) < 30.0 for epoch in epochs)
        and all(_finite(epoch.get("foundation_grad_ema")) and float(epoch["foundation_grad_ema"]) < 10.0 for epoch in epochs)
        and all(
            _finite(epoch.get("action_emergency_cap_rate"))
            and float(epoch["action_emergency_cap_rate"]) <= 0.01
            for epoch in epochs
        )
        and _finite(runtime.get("peak_reserved_gb"))
        and float(runtime["peak_reserved_gb"]) < 45.0
        and isinstance(dino, dict)
        and int(dino.get("main", 0)) > 0
        and all(int(dino.get(name, -1)) == 0 for name in CHEAP_SAME_FORWARD_MODES),
        {
            "max_action_logit": max(float(epoch["max_action_logit"]) for epoch in epochs if _finite(epoch.get("max_action_logit"))),
            "max_foundation_grad_ema": max(float(epoch["foundation_grad_ema"]) for epoch in epochs if _finite(epoch.get("foundation_grad_ema"))),
            "max_action_emergency_cap_rate": max(
                float(epoch["action_emergency_cap_rate"])
                for epoch in epochs
                if _finite(epoch.get("action_emergency_cap_rate"))
            ),
            "peak_reserved_gb": runtime.get("peak_reserved_gb"),
            "dino_call_count": dino,
        },
    )

    gate_payloads = {gate["gate"]: gate for gate in (gate_a, gate_b, gate_c, gate_d, gate_e, gate_f, gate_g)}
    gates = {letter: payload["pass"] for letter, payload in gate_payloads.items()}
    return {
        "pass": all(gates.values()),
        "git_head": git_head,
        "gates": gates,
        "gate_payloads": gate_payloads,
    }


def _epoch_payload(directory: Path) -> dict[str, Any]:
    rows = _read_jsonl(directory.parent / "loss_components.jsonl")
    epoch_index = int(directory.name.split("_")[-1])
    epoch_rows = [row for row in rows if int(row.get("epoch", -1)) == epoch_index]
    if not epoch_rows:
        raise ValueError(f"No loss rows for {directory.name}")
    action_cap_rates = [row.get("action_emergency_cap_rate") for row in epoch_rows]
    if not all(_finite(value) for value in action_cap_rates):
        raise ValueError(f"Missing action emergency-cap telemetry for {directory.name}")
    return {
        "branches": _read_json(directory / "branch_metrics.json"),
        "typed": _read_json(directory / "typed_evidence.json"),
        "runtime": _read_json(directory / "runtime.json"),
        "max_action_logit": max(float(row["action_final_logit_abs_max"]) for row in epoch_rows),
        "foundation_grad_ema": max(float(row.get("foundation_grad_ema", 0.0)) for row in epoch_rows),
        "action_emergency_cap_rate": sum(map(float, action_cap_rates)) / len(epoch_rows),
    }


def validate_heca_pilot_recomputation(
    directory: str | Path, *, expected_git_head: str
) -> list[str]:
    """Recompute A-G from raw four-epoch evidence instead of trusting self-reports."""
    root = Path(directory)
    try:
        epoch_dirs = sorted(path for path in root.glob("epoch_*") if path.is_dir())
        if len(epoch_dirs) != 4:
            return ["pilot_recomputation:expected_exactly_four_epochs"]
        recomputed = evaluate_heca_pilot(
            epochs=[_epoch_payload(path) for path in epoch_dirs],
            implementation_audit=_read_json(
                root / "heca_implementation_audit_input.json"
            ),
            ontology_manifest=_read_json(root / "heca_ontology_manifest_input.json"),
            tau_stats=_read_json(root / "heca_tau_stats_input.json"),
            gradient_rows=_read_jsonl(root / "heca_gradient_ownership.jsonl"),
            git_head=expected_git_head,
        )
        saved = _read_json(root / "HECA_PILOT_PASS.json")
        saved_gates = {
            letter: _read_json(root / f"HECA_GATE_{letter}.json")
            for letter in "ABCDEFG"
        }
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        return [f"pilot_recomputation:error:{type(error).__name__}"]
    failures = []
    if recomputed.get("pass") is not True:
        failures.append("pilot_recomputation:gates_failed")
    if recomputed.get("git_head") != expected_git_head:
        failures.append("pilot_recomputation:git_head")
    if saved.get("gates") != recomputed.get("gates"):
        failures.append("pilot_recomputation:gates_mismatch")
    if saved.get("pass") is not True or saved.get("git_head") != expected_git_head:
        failures.append("pilot_recomputation:saved_status_or_head")
    if saved.get("gate_payloads") != recomputed.get("gate_payloads"):
        failures.append("pilot_recomputation:payload_mismatch")
    for letter, payload in recomputed.get("gate_payloads", {}).items():
        if saved_gates.get(letter) != payload:
            failures.append(f"pilot_recomputation:gate_{letter}_mismatch")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_dir", required=True)
    parser.add_argument("--implementation_audit", required=True)
    parser.add_argument("--ontology_manifest", required=True)
    parser.add_argument("--tau_stats", required=True)
    args = parser.parse_args()
    root = Path(args.pilot_dir)
    epoch_dirs = sorted(path for path in root.glob("epoch_*") if path.is_dir())
    epochs = [_epoch_payload(path) for path in epoch_dirs]
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    audit = _read_json(args.implementation_audit)
    ontology = _read_json(args.ontology_manifest)
    tau = _read_json(args.tau_stats)
    gradients = _read_jsonl(root / "heca_gradient_ownership.jsonl")
    result = evaluate_heca_pilot(
        epochs=epochs,
        implementation_audit=audit,
        ontology_manifest=ontology,
        tau_stats=tau,
        gradient_rows=gradients,
        git_head=git_head,
    )
    write_json(root / "heca_implementation_audit_input.json", audit)
    write_json(root / "heca_ontology_manifest_input.json", ontology)
    write_json(root / "heca_tau_stats_input.json", tau)
    payload = {
        "ontology_manifest": ontology,
        "tau_stats": tau,
        "gradient_ownership": gradients,
        "loss_wiring": _read_json(root / "heca_loss_wiring.json"),
        "component_call_counters": _read_json(root / "heca_component_call_counters.json"),
        "contribution_conservation": _read_jsonl(root / "heca_contribution_conservation.jsonl"),
        "schedule_state": _read_json(root / "heca_schedule_state.json"),
        "ablation_manifest": {
            "cheap_same_forward": list(CHEAP_SAME_FORWARD_MODES),
            "independent_runs": INDEPENDENT_HECA_ABLATIONS,
        },
        "gates": result["gate_payloads"],
    }
    write_heca_artifact_sidecar(root, payload)
    write_heca_pilot_evidence_manifest(root, git_head=git_head)
    result["evidence_manifest_sha256"] = file_hash(
        root / "heca_pilot_evidence_manifest.json"
    )
    write_json(root / "HECA_PILOT_PASS.json", result)
    failures = validate_heca_pilot_bundle(root, expected_git_head=git_head)
    failures.extend(
        validate_heca_pilot_recomputation(root, expected_git_head=git_head)
    )
    result["artifact_failures"] = failures
    result["pass"] = result["pass"] and not failures
    write_json(root / "HECA_PILOT_PASS.json", result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
