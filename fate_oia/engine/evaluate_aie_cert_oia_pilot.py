from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from fate_oia.utils.aie_cert_artifacts import write_json


def _rows(path: Path) -> list[dict]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(root: Path) -> dict:
    metrics_rows = _rows(root / "metrics_summary.jsonl")
    mechanism_rows = _rows(root / "mechanism_stats.jsonl")
    epoch_dirs = sorted(path for path in root.glob("epoch_*") if path.is_dir())
    cf_rows, ecpo_rows, evidence_rows, owner_rows, dual_rows = [], [], [], [], []
    for epoch in epoch_dirs:
        cf_rows.append(json.loads((epoch / "counterfactual_certificate.json").read_text(encoding="utf-8")))
        ecpo_rows.append(json.loads((epoch / "ecpo_stats.json").read_text(encoding="utf-8")))
        evidence_rows.append(json.loads((epoch / "mechanism_summary.json").read_text(encoding="utf-8")))
        owner_rows.append(json.loads((epoch / "owner_gradients.json").read_text(encoding="utf-8")))
        dual_rows.append(json.loads((epoch / "dual_constraints.json").read_text(encoding="utf-8")))
    finite = True
    for collection in (metrics_rows, mechanism_rows, cf_rows, ecpo_rows, evidence_rows):
        text = json.dumps(collection, allow_nan=True)
        finite &= "NaN" not in text and "Infinity" not in text
    last_metric = metrics_rows[-1] if metrics_rows else {}
    last_evidence = evidence_rows[-1] if evidence_rows else {}
    owner_names = ("primary_core", "predicate_visual", "action_evidence", "action_contribution", "reason_private", "naming_readout")
    owner_nonzero = {}
    for owner in owner_names:
        seen_grad = any(row.get("owner_gradients", {}).get("before_clip", {}).get(owner, {}).get("grad_norm", 0) > 0
                        for row in mechanism_rows)
        seen_update = any(row.get("owner_gradients", {}).get("update_rms", {}).get(owner, 0) > 0
                          for row in mechanism_rows)
        owner_nonzero[owner] = seen_grad and seen_update
    dino_zero = all(row.get("owner_gradients", {}).get("before_clip", {}).get("dino_grad_max", 0) == 0
                    for row in mechanism_rows)
    cf_valid = sum(int(row.get("valid_events", 0)) for row in cf_rows)
    ecpo_pairs = sum(int(row.get("valid_pairs", 0)) for row in ecpo_rows)
    labels_with_pairs = max((int(row.get("labels_with_pairs", 0)) for row in ecpo_rows), default=0)
    max_queue_age = max((int(row.get("queue_max_age", 0)) for row in ecpo_rows), default=0)
    lambda_values = []
    for row in dual_rows:
        state = row.get("state", {})
        lambda_values.extend(float(value) for key, value in state.items() if key.startswith("lambda_"))
    cf_correlations = [row.get("contribution_certificate_pearson") for row in cf_rows
                       if row.get("contribution_certificate_pearson") is not None]
    gates = {
        "three_epochs": len(metrics_rows) == 3,
        "all_values_finite": finite,
        "all_owner_gradients_and_updates_nonzero": all(owner_nonzero.values()),
        "dino_gradient_zero": dino_zero,
        "predicate_mixture_active": last_evidence.get("predicate_mixture_active_rate", 0) > 0,
        "predicate_fallback_not_all": last_evidence.get("predicate_fallback_rate", 1) < 1,
        "predicate_effective_count_finite": math.isfinite(last_evidence.get("predicate_effective_count", float("nan"))),
        "local_global_ratio": 0.05 <= last_evidence.get("local_global_token_rms_ratio", -1) <= 2.0,
        "transport_token_nonzero": last_evidence.get("transport_token_delta_rms", 0) > 0,
        "transport_map_nonzero": last_evidence.get("transport_map_delta_rms", 0) > 0,
        "cotransport_identity": last_evidence.get("cotransport_matrix_discrepancy", 1) == 0,
        "contribution_exact": last_evidence.get("contribution_reconstruction_error", 1) < 1e-6,
        "counterfactual_events": cf_valid >= 64,
        "counterfactual_control_types": max((row.get("control_types_observed", 0) for row in cf_rows), default=0) >= 3,
        "certificate_positive_rate": (sum(row.get("certificate_positive_rate", 0) * row.get("valid_events", 0) for row in cf_rows)
                                      / max(cf_valid, 1)) >= 0.40,
        "contribution_certificate_positive_correlation": bool(cf_correlations) and max(cf_correlations) > 0,
        "dual_finite_not_saturated": bool(lambda_values) and all(math.isfinite(value) and value < 10.0 for value in lambda_values),
        "ecpo_pairs": ecpo_pairs >= 100,
        "ecpo_label_coverage": labels_with_pairs >= 8,
        "queue_age": max_queue_age <= 64,
        "reason_budget_not_min": last_evidence.get("reason_budget_max", 0) > last_evidence.get("reason_budget_min", 0) + 1e-5,
        "reason_budget_not_max": last_evidence.get("reason_budget_min", 1) < last_evidence.get("reason_budget_max", 1) - 1e-5,
        "reason_delta_bounded": last_evidence.get("reason_delta_rms", 99) < 1.5,
        "action_map_guard": last_metric.get("Act_mAP", 0) >= last_metric.get("primary", {}).get("Act_mAP", 0) - 0.02,
        "reason_map_guard": last_metric.get("Exp_mAP", 0) >= last_metric.get("primary", {}).get("Exp_mAP", 0) - 0.02,
        "naming_quality_reported": "naming_quality_mean" in last_evidence,
    }
    return {"pass": all(gates.values()), "gates": gates, "owner_status": owner_nonzero,
            "aggregate": {"cf_valid_events": cf_valid, "ecpo_valid_pairs": ecpo_pairs,
                          "ecpo_labels_with_pairs": labels_with_pairs, "queue_max_age": max_queue_age},
            "epochs": metrics_rows}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--pilot-dir", required=True)
    parser.add_argument("--config", default="configs/fate_oia_train_360x640_aie_cert_oia_v1.yaml")
    parser.add_argument("--review-dir", default=".review/aie_cert_oia_v1")
    args = parser.parse_args(); root = Path(args.pilot_dir); review_dir = Path(args.review_dir); review_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(root)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    binding = {"git_head": head, "config_hash": _sha256(Path(args.config)), "pilot_dir": str(root.resolve())}
    result.update(binding)
    write_json(root / "AIE_CERT_PILOT_GATE.json", result)
    pass_path = review_dir / "PILOT_PASS_AIE_CERT_OIA_V1.json"
    ready_path = review_dir / "AIE_CERT_FULL_TRAIN_READY.json"
    for path in (pass_path, ready_path):
        if path.exists(): path.unlink()
    if result["pass"]:
        write_json(pass_path, result)
        profile = json.loads((review_dir / "AIE_CERT_RUNTIME_PROFILE.json").read_text(encoding="utf-8"))
        implementation = json.loads((review_dir / "REVIEW_PASS_AIE_CERT_OIA_V1.json").read_text(encoding="utf-8"))
        bindings_match = profile.get("git_head") == head == implementation.get("git_head") and profile.get("config_hash") == binding["config_hash"] == implementation.get("config_hash")
        ready = {"pass": bool(bindings_match), **binding, "pilot_gate": str(pass_path),
                 "review_pass": str(review_dir / "REVIEW_PASS_AIE_CERT_OIA_V1.json"),
                 "runtime_profile": str(review_dir / "AIE_CERT_RUNTIME_PROFILE.json")}
        write_json(ready_path, ready)
        if not bindings_match: raise SystemExit(1)
    if not result["pass"]: raise SystemExit(1)


if __name__ == "__main__": main()
