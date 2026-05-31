from __future__ import annotations
import argparse, json, os, tempfile
from pathlib import Path
import torch
from fate_oia.datasets.dino_token_cache import DinoTokenCache
from fate_oia.engine.export_trace_visuals import export_trace_case
from fate_oia.models.trace_oia_model import TraceOIAModel
from fate_oia.utils.config_io import load_yaml_config
from fate_oia.utils.trace_artifacts import write_json
from fate_oia.utils.trace_review_gates import FORBIDDEN_PROXY_TERMS, FORBIDDEN_SUPERVISOR_TERMS, scan_forbidden


def _fake_grounding():
    return {"sample_0.jpg": {"objects": [{"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 180, "y2": 120}}]}, "sample_1.jpg": {"objects": [{"category": "traffic light", "box2d": {"x1": 200, "y1": 20, "x2": 240, "y2": 80}}]}}, {0: {"car"}, 1: {"traffic light"}}


def run_dynamic_smoke(device="cpu"):
    dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(7); model = TraceOIAModel(dim=384).to(dev).train()
    tokens = torch.randn(2, 3601, 384, device=dev); labels = torch.zeros(2, 21, device=dev); labels[0, 0] = 1; labels[1, 1] = 1
    cache, rules = _fake_grounding()
    out = model(tokens, batch={"file_name": ["sample_0.jpg", "sample_1.jpg"]}, grounding_cache=cache, reason_rules=rules, return_cf=True, cf_targets=labels)
    out["transport"]["evidence_reason_logits"].sum().backward()
    ev2 = dict(out["evidence"]); ev2["evidence_tokens"] = ev2["evidence_tokens"] + 0.5
    t2 = model.transport(out["label_tokens"][:, 4:], out["base_reason_logits"], ev2)
    with tempfile.TemporaryDirectory() as td:
        c = DinoTokenCache(td); c.put("a.jpg", torch.randn(4, 8)); c.get("a.jpg"); cache_hit_rate = c.stats()["cache_hit_rate"]
    return {"T_shape": list(out["transport"]["T"].shape), "T_mass_error_max": float(out["transport"]["transport_mass_error_max"].detach().cpu()), "T_sparse_fraction": float(out["transport"]["T_sparse_fraction"].detach().cpu()), "prototype_grad_norm": float(model.transport.prototypes.grad.norm().detach().cpu()), "evidence_reason_delta_after_token_perturb": float((t2["evidence_reason_logits"] - out["transport"]["evidence_reason_logits"]).abs().mean().detach().cpu()), "action_protection_max_abs_diff": float((out["action_logits"] - out["base_action_logits"]).abs().max().detach().cpu()), "cf_valid_count": int(out["cf"]["cf_valid_mask"].sum().detach().cpu()), "target_deleted_drop_mean": float(out["cf"]["target_deleted_drop_mean"].detach().cpu()), "non_target_deleted_drop_mean": float(out["cf"]["non_target_deleted_drop_mean"].detach().cpu()), "cf_is_proxy": bool(out["cf"]["cf_is_proxy"]), "cache_hit_rate": cache_hit_rate}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--output_dir", default=".background_runs/trace_oia_v1_preflight"); ap.add_argument("--device", default="cuda"); args = ap.parse_args(argv)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml_config(args.config); failures = []; checks = {"config_version": cfg.get("config_version")}
    cwd = Path.cwd(); branch = os.popen("git branch --show-current").read().strip(); head = os.popen("git rev-parse HEAD").read().strip(); dirty = os.popen("git status --short").read().strip()
    checks.update({"worktree_path": str(cwd), "branch": branch, "git_head": head, "dirty_status": dirty})
    if str(cwd).replace("/", "\\").lower() != "e:\\sbw\\fate_drive\\fate_oia_trace_oia_v1_worktree": failures.append("wrong_worktree_path")
    if branch != "trace_oia_v1_proto_transport": failures.append("wrong_branch")
    if cfg.get("config_version") != "trace_oia_v1_proto_transport": failures.append("wrong_config_version")
    if cfg.get("model", {}).get("token_compression") != "none": failures.append("token_compression_not_none")
    if cfg.get("model", {}).get("action_final_mode") != "base_only": failures.append("action_not_base_only")
    if scan_forbidden(["fate_oia/models/transport_counterfactual.py"], FORBIDDEN_PROXY_TERMS): failures.append("forbidden_proxy_terms")
    if scan_forbidden(["fate_oia/engine/supervise_trace_oia_foreground.py", "scripts/FATE_OIA_trace_oia_v1_foreground.ps1"], FORBIDDEN_SUPERVISOR_TERMS): failures.append("forbidden_supervisor_terms")
    smoke = run_dynamic_smoke(args.device); checks["dynamic_smoke"] = smoke
    if smoke["T_shape"][:3] != [2, 21, 6]: failures.append("bad_T_shape")
    if smoke["T_mass_error_max"] > 1e-4: failures.append("T_mass_not_normalized")
    if smoke["T_sparse_fraction"] <= 0.30: failures.append("T_not_sparse")
    if smoke["prototype_grad_norm"] <= 0: failures.append("no_prototype_grad")
    if smoke["evidence_reason_delta_after_token_perturb"] <= 0: failures.append("evidence_logits_do_not_change")
    if smoke["action_protection_max_abs_diff"] >= 1e-7: failures.append("action_not_protected")
    if smoke["cf_valid_count"] <= 0 or smoke["cf_is_proxy"]: failures.append("bad_counterfactual")
    if smoke["cache_hit_rate"] < 0.99: failures.append("cache_smoke_failed")
    export_trace_case(out, 0, {"sample_id": "audit_smoke", "reason_idx": 0, "prototype_id": 0, "drop": smoke["target_deleted_drop_mean"], "top_evidence": [{"source_type": "object", "transport_mass": 1.0}]})
    report = {"passed": not failures, "git_head": head, "dirty_status": dirty, "checks": checks, "failures": failures, "required_next_action": "train" if not failures else "fix_failures"}
    write_json(out / "review_report.json", report)
    if not failures:
        (out / "REVIEW_PASS_TRACE_OIA.txt").write_text(json.dumps({"git_head": head, "T_shape": smoke["T_shape"], "T_sparse_fraction": smoke["T_sparse_fraction"], "action_protection_max_abs_diff": smoke["action_protection_max_abs_diff"], "cache_hit_rate": smoke["cache_hit_rate"]}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures: raise SystemExit(2)


if __name__ == "__main__":
    main()
