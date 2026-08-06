from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from fate_oia.utils.aie_artifacts import validate_run_artifacts, write_json
from fate_oia.utils.aie_hashes import file_sha256


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def probe_health_gate(evidence: list[dict], epochs: list[dict]) -> bool:
    """Judge late probe health; initialization is not final collapse evidence."""
    if not evidence or not epochs:
        return False
    final_epoch_ids = {int(row["epoch"]) for row in epochs[-2:]}
    late_rows = [row for row in evidence if int(row.get("epoch", -1)) in final_epoch_ids]
    maximum_entropy = math.log(3600.0) - 0.01
    return any(
        row.get("dominant_probe_over_0p9_rate", 1.0) < 0.8
        and row.get("probe_pairwise_overlap", 1.0) < 0.9
        and row.get("probe_effective_count", 0.0) > 1.2
        and 0.1 < row.get("probe_map_entropy", maximum_entropy + 1.0) < maximum_entropy
        for row in late_rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--pilot-dir", required=True); parser.add_argument("--config", required=True)
    args = parser.parse_args(); root = Path(args.pilot_dir); epochs = read_jsonl(root / "metrics_summary.jsonl"); evidence = read_jsonl(root / "evidence_components.jsonl")
    preflight = root.parent / "aie_oia_v1_preflight"
    review = json.loads((preflight / "AIE_IMPLEMENTATION_REVIEW.json").read_text(encoding="utf-8"))
    runtime_path = preflight / "AIE_RUNTIME_PROFILE.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {"pass": False}
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    loss_rows = read_jsonl(root / "loss_components.jsonl")
    final_two = epochs[-2:]
    gates = {
        "A_foundation_primary": bool(review.get("functional_checks", {}).get("foundation_equivalence")) and bool(review.get("functional_checks", {}).get("primary_trajectory_isolation")) and all(row.get("model_state_hash_unchanged", True) for row in epochs),
        "B_evidence_active": any(row.get("raw_contribution_std", 0) > 1e-3 and row.get("action_evidence_grad", 0) > 0 and row.get("action_contribution_grad", 0) > 0 for row in loss_rows[:50]),
        "C_action_direction": any(row["final"]["Act_mAP"] >= row["primary"]["Act_mAP"] + 0.003 and row["final"]["Act_mF1"] >= row["primary"]["Act_mF1"] - 0.002 for row in final_two),
        "D_reason_direction": any(row["final"]["Exp_mAP"] >= row["primary"]["Exp_mAP"] + 0.003 and row["final"]["Exp_mF1"] >= row["primary"]["Exp_mF1"] - 0.003 for row in final_two),
        "F_probe_health": probe_health_gate(evidence, epochs),
        "I_artifacts": bool(runtime.get("pass")) and not validate_run_artifacts(root),
    }
    cf_rows = [row for epoch in sorted(root.glob("epoch_*")) for row in (read_jsonl(epoch / "counterfactual_metrics.jsonl") if (epoch / "counterfactual_metrics.jsonl").exists() else [])]
    eligible_cf = [row for row in cf_rows if row.get("available")]
    gates["E_counterfactual"] = any(
        row.get("cf_valid_count", 0) > 0
        and row.get("selected_minus_control_mean", float("-inf")) > 0
        and row.get("positive_action_directions", 0) >= 3
        and row.get("contribution_effect_spearman", float("-inf")) > 0.30
        and row.get("max_selected_control_overlap", float("inf")) <= 0.20
        for row in eligible_cf
    )
    naming_rows = [row for epoch in sorted(root.glob("epoch_*")) for row in (read_jsonl(epoch / "naming_metrics.jsonl") if (epoch / "naming_metrics.jsonl").exists() else [])]
    gates["G_naming_honesty"] = any(0.05 < row.get("named_coverage", 0) < 0.90 and row.get("quality_gt_random_rate", 0) > 0.5 for row in naming_rows)
    gates["H_firewall"] = True  # Bound to implementation review's dynamic owner-gradient proof below.
    gates["H_firewall"] = bool(review.get("functional_checks", {}).get("firewalls", False))
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    gates["J_hash_binding"] = (
        review.get("git_head") == current_head == manifest.get("git_head")
        and review.get("config_hash") == file_sha256(args.config)
        and review.get("source_tree_hash") == manifest.get("source_tree_hash")
        and runtime.get("git_head") == current_head
        and runtime.get("config_hash") == file_sha256(args.config)
    )
    passed = all(gates.values())
    last_epoch = root / f"epoch_{int(epochs[-1]['epoch']):03d}"
    binding = {
        "git_head": current_head,
        "source_head": review.get("source_head"),
        "source_tree_hash": manifest.get("source_tree_hash"),
        "config_hash": file_sha256(args.config),
        "predicate_schema_hash": manifest.get("predicate_schema_hash"),
        "counter_evidence_schema_hash": manifest.get("counter_evidence_schema_hash"),
        "split_hash": file_sha256(root / "split_manifest.json"),
        "checkpoint_hash": file_sha256(root / "checkpoint_latest.pth"),
        "logits_hash": file_sha256(last_epoch / "action_logits_final_test.pt"),
        "labels_hash": file_sha256(last_epoch / "labels_action_test.pt"),
        "file_order_hash": file_sha256(last_epoch / "file_names_test.json"),
    }
    payload = {"pass": passed, "gates": gates, "artifact_failures": validate_run_artifacts(root), **binding, "pilot_dir": str(root)}
    write_json(root / "AIE_PILOT_RAW_EVIDENCE.json", {"epochs": epochs, "loss_updates": loss_rows, "evidence_updates": evidence, "counterfactual": cf_rows, "naming": naming_rows})
    write_json(root / "AIE_PILOT_GATES.json", payload)
    if passed: write_json(root / "AIE_PILOT_PASS.json", payload); write_json(root / "AIE_FULL_TRAIN_READY.json", payload)
    print(json.dumps(payload, indent=2)); raise SystemExit(0 if passed else 1)


if __name__ == "__main__": main()
