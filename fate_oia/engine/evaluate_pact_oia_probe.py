from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from fate_oia.utils.aie_metrics import aie_branch_metrics
from fate_oia.utils.aie_calibration import apply_posthoc_threshold
from fate_oia.utils.pact_artifacts import sha256, validate_epoch_artifacts, write_json
from fate_oia.utils.pact_bootstrap import paired_bootstrap


def _metric(rows):
    metrics = aie_branch_metrics(torch.from_numpy(rows["action"]), torch.from_numpy(rows["reason"]),
                                 torch.from_numpy(rows["action_target"]), torch.from_numpy(rows["reason_target"]))
    return {key: metrics[key] for key in ("Act_mF1", "Act_oF1", "Act_mAP", "Exp_mF1", "Exp_oF1", "Exp_mAP", "joint")}


def _load_epoch(root: str, epoch: int) -> dict:
    directory = Path(root) / f"epoch_{epoch:03d}"
    packed = directory / "test_outputs.pt"
    if packed.exists():
        result = torch.load(packed, map_location="cpu")
    else:
        result = {
        "action_final": torch.load(directory / "action_logits_final_test.pt", map_location="cpu"),
        "reason_final": torch.load(directory / "reason_logits_final_test.pt", map_location="cpu"),
        "action_target": torch.load(directory / "labels_action_test.pt", map_location="cpu"),
        "reason_target": torch.load(directory / "labels_reason_test.pt", map_location="cpu"),
        "file_name": __import__("json").loads((directory / "file_names_test.json").read_text(encoding="utf-8")),
        }
    metrics = json.loads((directory / "branch_metrics.json").read_text(encoding="utf-8"))
    thresholds = metrics.get("thresholds_train_calib")
    if thresholds is None:
        thresholds = metrics["calibration_thresholds"]["threshold_prob"]
    result["thresholds_train_calib"] = torch.as_tensor(thresholds)
    result["branch_metrics"] = metrics
    return result


def _output_hash(root: str) -> str:
    digest = hashlib.sha256()
    for epoch in range(3):
        directory = Path(root) / f"epoch_{epoch:03d}"
        for name in ("action_logits_final_test.pt", "reason_logits_final_test.pt", "labels_action_test.pt",
                     "labels_reason_test.pt", "file_names_test.json"):
            with (directory / name).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_external_bindings(audit_path: str | Path, selected_path: str | Path, git_head: str,
                               config_hash: str, checkpoint_hash: str) -> tuple[dict, dict]:
    audit = _json(Path(audit_path))
    selected = _json(Path(selected_path))
    errors = []
    if not audit.get("pass", False):
        errors.append("implementation audit did not pass")
    if audit.get("git_head") != git_head:
        errors.append("implementation audit git_head mismatch")
    if audit.get("config_hash") != config_hash:
        errors.append("implementation audit config_hash mismatch")
    if audit.get("checkpoint_hash") != checkpoint_hash:
        errors.append("implementation audit checkpoint_hash mismatch")
    if not selected:
        errors.append("selected hyperparameters are empty")
    if errors:
        raise RuntimeError("; ".join(errors))
    return audit, selected


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--control-dir", required=True); parser.add_argument("--method-dir", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--resamples", type=int, default=2000); parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--config", required=True); parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--implementation-audit", required=True); parser.add_argument("--selected-hparams", required=True)
    args = parser.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config_hash, checkpoint_hash = sha256(args.config), sha256(args.source_checkpoint)
    audit, selected_hparams = validate_external_bindings(
        args.implementation_audit, args.selected_hparams, git_head, config_hash, checkpoint_hash)
    decisions = []
    for epoch in range(3):
        control = _load_epoch(args.control_dir, epoch)
        method = _load_epoch(args.method_dir, epoch)
        if control["file_name"] != method["file_name"]:
            raise RuntimeError("CONTROL and PACT test sample order differs")
        control_action = apply_posthoc_threshold(control["action_final"], control["thresholds_train_calib"][:4])
        control_reason = apply_posthoc_threshold(control["reason_final"], control["thresholds_train_calib"][4:])
        method_action = apply_posthoc_threshold(method["action_final"], method["thresholds_train_calib"][:4])
        method_reason = apply_posthoc_threshold(method["reason_final"], method["thresholds_train_calib"][4:])
        c = {"action": control_action.numpy(), "reason": control_reason.numpy(),
             "action_target": control["action_target"].numpy(), "reason_target": control["reason_target"].numpy()}
        m = {"action": method_action.numpy(), "reason": method_reason.numpy(),
             "action_target": method["action_target"].numpy(), "reason_target": method["reason_target"].numpy()}
        bootstrap = paired_bootstrap(c, m, _metric, args.resamples, args.seed + epoch)
        write_json(out / f"paired_bootstrap_epoch_{epoch}.json", bootstrap)
        write_json(Path(args.control_dir) / f"epoch_{epoch:03d}" / "paired_bootstrap.json", bootstrap)
        write_json(Path(args.method_dir) / f"epoch_{epoch:03d}" / "paired_bootstrap.json", bootstrap)
        decisions.append({"epoch": epoch, "bootstrap": bootstrap, "control": _metric(c), "method": _metric(m)})
    write_json(out / "paired_results.json", {"epochs": decisions, "metric_view": "deploy_train_calib_threshold"})

    pareto_epoch_pass = []
    for row in decisions:
        c, m, b = row["control"], row["method"], row["bootstrap"]
        point = (m["Act_mF1"] >= c["Act_mF1"] + 0.002 and m["Exp_mF1"] >= c["Exp_mF1"] + 0.004 and
                 m["Act_mAP"] >= c["Act_mAP"] - 0.001 and m["Exp_mAP"] >= c["Exp_mAP"] + 0.002 and
                 m["joint"] >= c["joint"] + 0.004)
        ci = b["joint"]["p2_5"] > 0 and b["Act_mF1"]["p2_5"] >= -0.001 and b["Exp_mF1"]["p2_5"] > 0
        pareto_epoch_pass.append(point and ci)
    two_consecutive = any(pareto_epoch_pass[i] and pareto_epoch_pass[i + 1] for i in range(2))
    old_line = any(row["method"]["Act_mF1"] >= 0.724 and row["method"]["Exp_mF1"] >= 0.385 and
                   row["method"]["Act_mAP"] >= 0.788 and row["method"]["Exp_mAP"] >= 0.375 for row in decisions)

    mechanism_rows, missing_artifacts = [], []
    for epoch in range(3):
        directory = Path(args.method_dir) / f"epoch_{epoch:03d}"
        missing_artifacts.extend(f"epoch_{epoch:03d}/{name}" for name in validate_epoch_artifacts(directory))
        role = _json(directory / "role_gradient_stats.json"); reason = _json(directory / "reason_rank_coverage.json")
        action = _json(directory / "action_rank_stats.json"); predicate = _json(directory / "predicate_agreement_stats.json")
        cf = _json(directory / "counterfactual_summary.json"); branch = _json(directory / "branch_metrics.json")
        mechanism_rows.append({"role": role, "reason": reason, "action": action, "predicate": predicate, "cf": cf,
                               "branch": branch})
    license_rows = []
    license_path = Path(args.method_dir) / "pareto_license_stats.jsonl"
    if license_path.exists():
        license_rows = [json.loads(line) for line in license_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    choices = {row["selected_lambda"] for row in license_rows}
    licenses = [row["license_after"] for row in license_rows]
    mechanism = {
        "owner_firewall": all(row["role"]["illegal_cross_owner_grad_max"] < 1e-8 for row in mechanism_rows),
        "license_nontrivial": bool(licenses) and not all(value == 0 for value in licenses) and
                              not all(value == 0.75 for value in licenses) and len(choices) >= 2,
        "reason_bound": all(row["reason"]["reason_delta_to_budget_max"] <= 1.0001 for row in mechanism_rows),
        "pair_coverage": all(row["reason"]["labels_with_pairs"] == 21 for row in mechanism_rows),
        "action_map_noninferiority": all(row["branch"]["final_raw"]["Act_mAP"] >= row["branch"]["primary"]["Act_mAP"] - 0.0005 for row in mechanism_rows),
        "forward_stop_ap_guard": all(all(row["branch"]["final_raw"]["Act_per_label_ap"][label] >=
                                          row["branch"]["primary"]["Act_per_label_ap"][label] - 0.01 for label in (0, 1))
                                     for row in mechanism_rows),
        "predicate_gate_nontrivial": all(row["predicate"]["gate_mean"] > 0 and row["predicate"]["gate_p90"] < 0.25 for row in mechanism_rows),
        "counterfactual_lcb_positive": all(row["cf"].get("selected_minus_control_bootstrap_lcb", float("-inf")) > 0 for row in mechanism_rows),
        "naming_coverage": all(row["predicate"]["named_coverage"] > 0 for row in mechanism_rows),
    }
    layers = {"code_and_numerics": bool(audit["pass"]) and not missing_artifacts,
              "paired_pareto_two_consecutive": two_consecutive, "old_pareto_line": old_line,
              "mechanism": all(mechanism.values())}
    passed = all(layers.values())
    method_manifest = _json(Path(args.method_dir) / "run_manifest.json")
    gate = {"pass": passed, "layers": layers, "pareto_epoch_pass": pareto_epoch_pass, "mechanism": mechanism,
            "missing_artifacts": missing_artifacts, "source_head": method_manifest["source_head"],
            "probe_head": git_head,
            "config_hash": config_hash, "checkpoint_hash": checkpoint_hash,
            "split_hash": method_manifest["split_hash"], "selected_hyperparameters": selected_hparams,
            "control_output_hash": _output_hash(args.control_dir), "pact_output_hash": _output_hash(args.method_dir),
            "paired_results": decisions}
    write_json(out / "probe_gate.json", gate)
    pass_path, fail_path = out / "PACT_FAST_VALIDATION_PASS.json", out / "PACT_FAST_VALIDATION_FAIL.json"
    if passed:
        write_json(pass_path, gate); fail_path.unlink(missing_ok=True)
    else:
        write_json(fail_path, gate); pass_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
