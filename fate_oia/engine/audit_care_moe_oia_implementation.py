from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.datasets.bdd100k_structured import BDD100KStructuredIndex
from fate_oia.models.care_moe_oia_model import CAREMoEOIAModel
from fate_oia.losses.care_moe_losses import care_moe_training_loss


PASS_TOKEN = "REVIEW_PASS_CARE_MOE_OIA_V1"


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True, errors="ignore").strip()


def main() -> None:
    root = Path.cwd()
    out_dir = root / ".background_runs" / "care_moe_oia_v1_preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    notes: list[str] = []
    branch = _git(["branch", "--show-current"])
    if branch != "care_moe_oia_v1_direct_image":
        failed.append(f"wrong branch: {branch}")
    if "fate_oia_care_moe_oia_v1_worktree" not in str(root):
        failed.append(f"wrong worktree path: {root}")
    cfg_path = root / "configs" / "fate_oia_train_360x640_care_moe_oia_v1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    checks = {
        "config_version": cfg.get("config_version") == "care_moe_oia_v1_direct_image",
        "feature_cache_enabled": cfg.get("feature_cache_enabled") is False,
        "token_compression": cfg.get("token_compression") == "none",
        "test_only_evaluation": cfg.get("test_only_evaluation") is True,
        "best_selection_split": cfg.get("best_selection_split") == "test",
        "action_cap": float(cfg.get("action_cap_max", 1.0)) <= 0.04,
        "reason_dim": int(cfg.get("reason_dim", 0)) == 21,
        "action_dim": int(cfg.get("action_dim", 0)) == 4,
        "include_confuse": cfg.get("include_confuse") is False,
    }
    failed.extend([f"config check failed: {k}" for k, ok in checks.items() if not ok])
    ds_train = BDDOIAMultiTaskDataset(data_root=cfg["data_root"], raw_root=cfg["bdd_oia_root"], split="train", action_dim=4, reason_dim=21)
    ds_test = BDDOIAMultiTaskDataset(data_root=cfg["data_root"], raw_root=cfg["bdd_oia_root"], split="test", action_dim=4, reason_dim=21)
    idx = BDD100KStructuredIndex(cfg["bdd100k_root"])
    audit_train = idx.audit_split([s.file_name for s in ds_train.samples], "train", sample_limit=512)
    audit_test = idx.audit_split([s.file_name for s in ds_test.samples], "test", sample_limit=512)
    for name, audit in [("train", audit_train), ("test", audit_test)]:
        if audit["match_rate"] < 0.90:
            failed.append(f"BDD100K match rate below 0.90 on {name}: {audit['match_rate']}")
        if audit["object_count"] <= 0 or audit["lane_count"] <= 0 or audit["drivable_count"] <= 0:
            failed.append(f"BDD100K structured counts zero on {name}: {audit}")
    model = CAREMoEOIAModel()
    model.train()
    tokens = torch.randn(2, 3601, 384)
    reason = torch.zeros(2, 21)
    reason[0, [5, 9, 12]] = 1
    reason[1, [6, 11, 14]] = 1
    action = torch.zeros(2, 4)
    action[:, 1] = 1
    structured = [idx.resolve(ds_train.samples[0].file_name, "train").to_dict(), idx.resolve(ds_train.samples[1].file_name, "train").to_dict()]
    out = model(tokens, batch={"reason": reason}, structured=structured, epoch=3)
    if float(out["active_reason_recall_train"]) < 0.999:
        failed.append("GT-positive active reason coverage failed")
    if int((out["expert_route_mask"].sum(-1) > 2).sum().item()) != 0:
        failed.append("top-2 expert routing violation")
    if len(out["expert_usage"]) != 5 or not all(k in out["expert_usage"] for k in ["object", "lane", "drivable", "traffic_control", "global_context"]):
        failed.append("missing evidence experts")
    if float(out["action_delta"].abs().max()) > 0.040001:
        failed.append("action residual cap exceeded")
    args = SimpleNamespace(asl_gamma_pos=0.0, asl_gamma_neg=4.0, asl_clip=0.05, loss_evidence_bag=0.1, loss_reason_delta_reg=0.001, loss_action_delta_reg=0.001)
    loss, _ = care_moe_training_loss(out, action, reason, args)
    loss.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.expert_router.parameters())
    if not grad_ok:
        failed.append("bag/task loss gradient did not reach router/expert parameters")
    if bool(out["diagnostics"].get("primary_test_uses_bdd100k_gt")):
        failed.append("primary test path indicates BDD100K GT usage")
    if failed:
        review = {"review": "FAIL", "review_pass_token": "", "failed_checks": failed, "notes": notes, "bdd100k_train": audit_train, "bdd100k_test": audit_test}
    else:
        review = {"review": "PASS", "review_pass_token": PASS_TOKEN, "failed_checks": [], "notes": notes, "bdd100k_train": audit_train, "bdd100k_test": audit_test}
        (out_dir / "REVIEW_PASS_CARE_MOE_OIA_V1.txt").write_text(PASS_TOKEN + "\n", encoding="utf-8")
    (out_dir / "review_report.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(review, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
