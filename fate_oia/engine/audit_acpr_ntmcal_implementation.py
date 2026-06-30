from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
import yaml

from fate_oia.models.acpr_ntmcal_model import ACPRNTMCalModel
from fate_oia.utils.acpr_ntmcal_tensor_asserts import assert_deploy_equation, assert_shape


REQUIRED = [
    "configs/acpr_ntmcal_native_text_predicates.yaml", "configs/acpr_ntmcal_reason_formulas.yaml",
    "fate_oia/models/acpr_ntmcal_text_atoms.py", "fate_oia/models/acpr_ntmcal_predicate_bank.py",
    "fate_oia/models/acpr_ntmcal_topk_predicate_measurement.py", "fate_oia/models/acpr_ntmcal_observation_builder.py",
    "fate_oia/models/acpr_ntmcal_pu_state.py", "fate_oia/models/acpr_ntmcal_reason_residual.py", "fate_oia/models/acpr_ntmcal_action_predicate_head.py",
    "fate_oia/models/acpr_ntmcal_threshold_head.py", "fate_oia/models/acpr_ntmcal_pair_memory.py", "fate_oia/models/acpr_ntmcal_model.py",
    "fate_oia/losses/acpr_ntmcal_losses.py", "fate_oia/engine/train_acpr_ntmcal_oia.py", "fate_oia/engine/eval_acpr_ntmcal_oia.py",
    "fate_oia/engine/supervise_acpr_ntmcal_foreground.py", "scripts/FATE_OIA_acpr_ntmcal_v1_memory_probe.ps1", "scripts/FATE_OIA_acpr_ntmcal_v1_foreground.ps1",
]
FORBIDDEN = ["hashlib", "open_clip", "CLIPModel", "AutoTokenizer", "AutoModel", "BertModel", "SentenceTransformer", "sentence_transformers", "frozen_run_c", "FrozenRunC", "run_c_logits", "cached_logits", "tail_residual_adapter", "checkpoint distillation", "teacher checkpoint", "expert", "Expert", "moe", "MoE", "specialist", "Specialist", "router", "Router", "pmi", "cooccur", "co_occurrence", "label_correlation", "feature_cache_enabled: true", "token_compression: keep_merge", "checkpoint_best_val", "best_selection_split: val", "eval_splits: val", "Start-Job", "nohup", "scheduled task", "daemon"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--output_dir", required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--write_review_pass", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    missing = [p for p in REQUIRED if not Path(p).exists()]
    forbidden = {}
    for path in REQUIRED:
        if Path(path).suffix in {".py", ".yaml", ".ps1"} and Path(path).exists():
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            hits = [x for x in FORBIDDEN if x in text]
            if hits:
                forbidden[path] = hits
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    functional = {}
    pass_flag = not missing and not forbidden
    try:
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        model = ACPRNTMCalModel(
            selected_layers=tuple(cfg.get("model", {}).get("selected_layers", [3, 7, 11])),
            pretrained_weights=str(cfg.get("pretrained_weights", "ckp/reference/dino_deitsmall8_pretrain.pth")),
            use_mock_dino=True,
            predicate_topk=8,
        ).to(device)
        images = torch.randn(2, 3, 360, 640, device=device)
        reason = torch.zeros(2, 21, device=device); reason[:, 0] = 1
        out = model(images, epoch=8, reason_labels=reason, split="train")
        assert_shape(out["action_logits_deploy"], (2, 4), "action_logits_deploy")
        assert_shape(out["reason_logits_deploy"], (2, 21), "reason_logits_deploy")
        assert_shape(out["predicate_q"], (2, None), "predicate_q")
        assert_shape(out["predicate_rho"], (2, None), "predicate_rho")
        assert_deploy_equation(out["action_logits_ntmcal"], out["theta_action"], out["action_logits_deploy"], "action")
        assert_deploy_equation(out["reason_logits_ntmcal"], out["theta_reason"], out["reason_logits_deploy"], "reason")
        out_zero = model(images, epoch=8, reason_labels=reason, force_zero_reason_delta=True)
        action_diff = (out["action_logits_ntmcal"] - out_zero["action_logits_ntmcal"]).abs().max().item()
        if action_diff > 1e-6:
            raise AssertionError(f"reason residual changed action logits: {action_diff}")
        functional["full_model_forward"] = {"pass": True, "action_independence_error": action_diff, "predicate_count": len(model.predicate_bank.specs)}
    except Exception as exc:
        pass_flag = False
        functional["full_model_forward"] = {"pass": False, "error": repr(exc)}
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
    result = {"pass": bool(pass_flag), "git_head": head.stdout.strip(), "branch": branch.stdout.strip(), "worktree": str(Path.cwd()), "checked_files": REQUIRED, "forbidden_pattern_results": forbidden, "functional_checks": functional, "memory_probe_result": {}, "smoke_result": {}, "review_pass_path": str(out_dir / "REVIEW_PASS_ACPR_NTMCAL_V1.txt"), "missing_items": missing, "warnings": []}
    (out_dir / "implementation_audit_ACPR_NTMCAL_V1.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if pass_flag and args.write_review_pass:
        (out_dir / "REVIEW_PASS_ACPR_NTMCAL_V1.txt").write_text("REVIEW_PASS_ACPR_NTMCAL_V1\n", encoding="utf-8")
    if not pass_flag:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
