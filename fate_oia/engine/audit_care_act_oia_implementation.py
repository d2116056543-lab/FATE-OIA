from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from fate_oia.datasets.bdd100k_structured import BDD100KStructuredIndex
from fate_oia.engine.eval_care_act_oia import select_guarded_action_branch
from fate_oia.models.care_act_model import CAREActOIAModel
from fate_oia.models.care_action_evidence_experts import ActionEvidenceExpertBank
from fate_oia.models.care_action_set_head import ActionSetConsistencyHead
from fate_oia.utils.care_act_artifacts import write_json
from fate_oia.utils.care_act_review_gates import PASS_TOKEN


FORBIDDEN = ["Start-Process", "Start-Job", "Win32_Process", "Invoke-WmiMethod", "nohup", "hidden cmd", "detached process"]


def _check(name: str, ok: bool, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _git_branch(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    except Exception:
        return ""


def run_static_audit(root: Path, config_path: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    checks.append(_check("branch", _git_branch(root) == "care_moe_oia_v1_direct_image", _git_branch(root)))
    checks.append(_check("config_version", cfg.get("config_version") == "care_act_oia_v1_direct_image", cfg.get("config_version")))
    checks.append(_check("feature_cache_disabled", cfg.get("feature_cache_enabled") is False, cfg.get("feature_cache_enabled")))
    checks.append(_check("token_compression_none", cfg.get("token_compression") == "none", cfg.get("token_compression")))
    checks.append(_check("test_only", cfg.get("test_only_evaluation") is True and cfg.get("best_selection_split") == "test", {"test_only": cfg.get("test_only_evaluation"), "best": cfg.get("best_selection_split")}))
    required = [
        "fate_oia/models/care_action_evidence_experts.py",
        "fate_oia/models/care_action_set_head.py",
        "fate_oia/models/care_act_model.py",
        "fate_oia/losses/care_act_losses.py",
        "fate_oia/engine/train_care_act_oia.py",
        "fate_oia/engine/eval_care_act_oia.py",
        "fate_oia/engine/supervise_care_act_oia_foreground.py",
    ]
    checks.append(_check("required_files", all((root / x).exists() for x in required), [x for x in required if not (root / x).exists()]))
    fg_files = ["fate_oia/engine/supervise_care_act_oia_foreground.py", "scripts/FATE_OIA_care_act_oia_v1_foreground.ps1"]
    bad: list[str] = []
    for rel in fg_files:
        p = root / rel
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="ignore")
            bad.extend([f"{rel}:{tok}" for tok in FORBIDDEN if tok in text])
    checks.append(_check("foreground_no_detach_tokens", not bad, bad))
    return {"review": "PASS" if all(c["ok"] for c in checks) else "FAIL", "checks": checks}


def run_semantic_audit(root: Path, cfg: dict[str, Any], device: str = "cpu") -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    tokens = torch.randn(2, 45 * 80 + 1, 384, device=dev)
    action_tokens = torch.randn(2, 4, 384, device=dev)
    base_action = torch.randn(2, 4, device=dev)
    bank = ActionEvidenceExpertBank().to(dev)
    out1 = bank(action_tokens, tokens, base_action)
    tokens2 = tokens.clone()
    tokens2[:, 100:120] += 5.0
    out2 = bank(action_tokens, tokens2, base_action)
    checks.append(_check("action_evidence_perturbation_changes_logits", not torch.allclose(out1["action_evidence_delta_raw"], out2["action_evidence_delta_raw"])))
    checks.append(_check("action_expert_top2", out1["expert_route_mask"].sum(-1).max().item() <= 2))
    head = ActionSetConsistencyHead().to(dev)
    set_out = head(base_action, out1["action_evidence_context"], torch.rand_like(base_action))
    set_out["action_set_logits"].sum().backward()
    checks.append(_check("action_set_head_gradient", any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())))
    model = CAREActOIAModel().to(dev)
    model.train()
    reason = torch.zeros(2, 21, device=dev)
    reason[:, [5, 6, 9]] = 1
    mout = model(tokens, batch={"reason": reason}, structured=[None, None], epoch=8)
    checks.append(_check("model_required_outputs", all(k in mout for k in ["action_evidence_logits", "action_set_logits", "action_final_candidate_logits", "reason_logits"])))
    checks.append(_check("caps", mout["action_evidence_delta"].abs().max().item() <= 0.12001 and mout["action_set_delta"].abs().max().item() <= 0.06001 and mout["action_total_delta"].abs().max().item() <= 0.15001))
    labels = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32)
    good = torch.where(labels > 0.5, torch.full_like(labels, 3.0), torch.full_like(labels, -3.0))
    bad = -good
    selected = select_guarded_action_branch({"base": bad, "evidence": good, "action_set": bad, "candidate": bad}, labels)
    checks.append(_check("metric_level_selector", selected["selected_branch"] == "evidence"))
    index = BDD100KStructuredIndex(cfg.get("bdd100k_root", "E:/sbw/BDD100K"))
    checks.append(_check("bdd100k_index_nonempty", len(index.label_map) > 0 and len(index.drivable_map) > 0, {"labels": len(index.label_map), "drivable": len(index.drivable_map)}))
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_care_act_oia_v1.yaml")
    ap.add_argument("--output_dir", default=".background_runs/care_act_oia_v1_preflight")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    root = Path.cwd()
    config_path = root / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    result = run_static_audit(root, config_path)
    semantic = run_semantic_audit(root, cfg, args.device)
    result["checks"].extend(semantic)
    result["review"] = "PASS" if all(c["ok"] for c in result["checks"]) else "FAIL"
    result["review_pass_token"] = PASS_TOKEN if result["review"] == "PASS" else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "care_act_implementation_audit.json", result)
    if result["review"] == "PASS":
        (out_dir / "REVIEW_PASS_CARE_ACT_OIA_V1.txt").write_text(PASS_TOKEN + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["review"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
