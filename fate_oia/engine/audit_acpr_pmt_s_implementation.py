from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml


REQUIRED = [
    "configs/fate_oia_train_360x640_acpr_pmt_s_v1.yaml",
    "configs/acpr_pmt_action_predicate_grammar.yaml",
    "fate_oia/models/acpr_predicate_patch_targets.py",
    "fate_oia/models/acpr_predicate_transport_alignment.py",
    "fate_oia/models/acpr_triadic_mediator.py",
    "fate_oia/models/acpr_predicate_conditioned_threshold.py",
    "fate_oia/losses/acpr_pmt_losses.py",
    "fate_oia/utils/acpr_pmt_artifacts.py",
    "fate_oia/utils/acpr_pmt_phase_schedule.py",
    "fate_oia/utils/acpr_pmt_visualization.py",
    "fate_oia/engine/audit_acpr_pmt_s_implementation.py",
    "fate_oia/engine/export_acpr_pmt_visuals.py",
    "fate_oia/engine/supervise_acpr_pmt_s_foreground.py",
    "scripts/FATE_OIA_acpr_pmt_s_v1_foreground.ps1",
]

FORBIDDEN = [
    "frozen_run_c", "FrozenRunC", "run_c_logits", "cached_logits", "tail_residual_adapter",
    "feature_cache_enabled: true", "token_compression: keep_merge", "Start-Process", "Start-Job",
    "nohup", "scheduled task", "test_threshold_teacher", "update threshold from test", "update gate from test",
]


def run_static_audit(root: Path) -> dict:
    checked = {p: (root / p).exists() for p in REQUIRED}
    scan_files = [p for p in REQUIRED if (root / p).exists() and not p.endswith("audit_acpr_pmt_s_implementation.py")]
    text = "\n".join((root / p).read_text(encoding="utf-8", errors="ignore") for p in scan_files)
    forbidden = {pat: (pat not in text) for pat in FORBIDDEN}
    cfg_ok = False
    cfg_path = root / "configs/fate_oia_train_360x640_acpr_pmt_s_v1.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        cfg_ok = bool(cfg.get("pmt", {}).get("enabled")) and cfg.get("feature_cache_enabled") is False and str(cfg.get("token_compression")) == "none"
    functional = {
        "required_files": all(checked.values()),
        "forbidden_patterns_absent": all(forbidden.values()),
        "config_no_cache_no_compression_test": cfg_ok,
        "triadic_mediator_present": "class ACPRTriadicMediator" in text and "predicate-only action delta is impossible" in text,
        "predicate_patch_targets_present": "class ACPRPredicatePatchTargetBuilder" in text,
        "predicate_conditioned_threshold_present": "class ACPRPredicateConditionedThreshold" in text,
        "phase_schedule_present": "pmt_phase_for_epoch" in text,
        "visual_chain_present": "patch_coordinates" in text and "reason_name" in text,
    }
    return {
        "pass": all(functional.values()),
        "git_head": "",
        "checked_files": checked,
        "forbidden_pattern_results": forbidden,
        "functional_checks": functional,
        "missing_items": [k for k, v in checked.items() if not v],
        "warnings": [],
    }


def dynamic_audit(device: str = "cpu") -> dict:
    from fate_oia.models.acpr_oia_model import ACPROIAModel
    model = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, pmt_kwargs={"enabled": True})
    model.eval().to(device)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 360, 640, device=device), epoch=0)
    return {
        "action_shape": list(out["action_logits_final_raw"].shape),
        "reason_shape": list(out["reason_logits_final_raw"].shape),
        "triadic_delta_zero": float(out["triadic_action_delta"].abs().max().detach().cpu()) < 1e-6,
        "branch_logits": sorted(out["branch_logits"].keys()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--write_review_pass", action="store_true")
    args = ap.parse_args()
    root = Path(".")
    result = run_static_audit(root)
    try:
        result["smoke_result"] = dynamic_audit("cpu")
        result["pass"] = bool(result["pass"] and result["smoke_result"]["triadic_delta_zero"])
    except Exception as e:
        result["pass"] = False
        result["smoke_result"] = {"error": repr(e)}
    try:
        import subprocess
        result["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        result["git_head"] = ""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit_path = out / "implementation_audit_ACPR_PMT_S_V1.json"
    audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if result["pass"] and args.write_review_pass:
        (out / "REVIEW_PASS_ACPR_PMT_S_V1.txt").write_text("REVIEW_PASS_ACPR_PMT_S_V1\n" + result["git_head"] + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
