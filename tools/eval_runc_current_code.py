from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import fate_oia.engine.train_fate_oia as t
from fate_oia.models.fate_oia_model import FATEOIAFeatureModel

DEFAULT_DATA_ROOT = r"E:\sbw\FATE_Drive\fate_oia_worktree\dataset\BDD-OIA"
DEFAULT_RAW_ROOT = r"E:\sbw\FATE_Drive\fate_oia_worktree\raw_data\BDD-OIA"
DEFAULT_PRETRAINED = r"E:\sbw\FATE_Drive\fate_oia_worktree\ckp\reference\dino_deitsmall8_pretrain.pth"

def ensure(args: Namespace, name: str, value: Any) -> None:
    if not hasattr(args, name):
        setattr(args, name, value)

def main() -> int:
    ap = argparse.ArgumentParser(description="Strict current-code Run C test reproduction; no training.")
    ap.add_argument("--artifacts_dir", default=str(ROOT / "run_c_artifacts"))
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    ap.add_argument("--pretrained_weights", default=DEFAULT_PRETRAINED)
    ap.add_argument("--output", default=str(ROOT / "runc_outputs" / "current_code_eval_repro.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_test_samples", type=int, default=0)
    cli = ap.parse_args()

    art = Path(cli.artifacts_dir)
    ckpt = art / "checkpoint_best_test.pth"
    args_json = art / "args.json"
    config_yaml = art / "training_config_resolved.yaml"
    manifest_json = art / "run_manifest.json"
    target_json = art / "metrics_best_test.json"
    for path in [ckpt, args_json, config_yaml, manifest_json, target_json, Path(cli.data_root), Path(cli.raw_root), Path(cli.pretrained_weights)]:
        if not path.exists():
            raise FileNotFoundError(path)

    args = Namespace(**json.loads(args_json.read_text(encoding="utf-8-sig")))
    target = json.loads(target_json.read_text(encoding="utf-8-sig"))
    args.data_root = cli.data_root
    args.raw_root = cli.raw_root
    args.pretrained_weights = cli.pretrained_weights
    args.device = cli.device
    args.resume = str(ckpt)
    args.output_dir = str(ROOT / "runc_outputs")
    args.max_test_samples = int(cli.max_test_samples)

    overrides = {
        "num_workers": 0,
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 32,
        "max_train_samples": 0,
        "max_val_samples": 0,
        "max_saved_token_stats": 0,
        "log_every": 1000000000,
        "threshold_mode": "fixed",
        "eval_threshold": 0.5,
        "reason_loss": "asl",
        "reason_loss_weight": 1.0,
        "reason_prior_path": "",
        "reason_logit_adjust_tau": 0.3,
        "reason_logit_adjustment": None,
        "label_correlation_bias_path": "",
        "label_correlation_bias": "none",
        "label_correlation_bias_weight": 0.0,
        "label_correlation_residual_init": 1.0,
        "label_correlation_residual_learnable": True,
        "fusion_mode": "learned_gate",
        "fusion_fixed_alpha": 0.0,
        "fusion_gate_floor": 0.0,
        "loss_gate_balance": 0.0,
        "loss_gate_entropy": 0.0,
        "fusion_gate_target": 0.5,
        "task_balance": "none",
        "loss_counterfactual": 0.0,
        "counterfactual_eval": False,
        "counterfactual_start_epoch": 0,
        "counterfactual_topk_ratio": 0.05,
        "cf_mask_fill": "mean",
        "loss_grounding": 0.0,
        "grounding_mode": "none",
        "grounding_cache_jsonl": "",
        "reason_grounding_rules": "",
        "token_score_mode": "norm",
        "scheduler": "cosine",
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "resume_strict": True,
        "resume_optimizer": False,
    }
    for name, value in overrides.items():
        setattr(args, name, value)

    if args.label_correlation == "self_attn" and t.checkpoint_uses_legacy_label_correlation(ckpt):
        args.label_correlation = "self_attn_legacy"
        args.label_correlation_legacy_detected = True
    else:
        args.label_correlation_legacy_detected = False

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    label_bias_matrix = t.load_label_bias_matrix(args.label_correlation_bias_path, args.label_correlation_bias, args.action_dim + args.reason_dim)
    args.reason_logit_adjustment = None
    backbone, dim = t.build_backbone(args, device)
    model = FATEOIAFeatureModel(
        dim=dim,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        use_label_query=args.use_label_query,
        label_correlation=args.label_correlation,
        label_correlation_layers=args.label_correlation_layers,
        label_correlation_heads=args.label_correlation_heads,
        label_correlation_dropout=args.label_correlation_dropout,
        label_correlation_bias=args.label_correlation_bias,
        label_correlation_bias_matrix=label_bias_matrix,
        label_correlation_bias_weight=args.label_correlation_bias_weight,
        label_correlation_residual_init=args.label_correlation_residual_init,
        label_correlation_residual_learnable=args.label_correlation_residual_learnable,
        fusion_mode=args.fusion_mode,
        fusion_fixed_alpha=args.fusion_fixed_alpha,
        fusion_gate_floor=args.fusion_gate_floor,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    resume_state = t.load_resume_checkpoint(ckpt, model, optimizer, device=device, resume_optimizer=False, strict=True)
    criterion = t.make_multilabel_criterion(args)
    loader = t.make_loader(args, "test", shuffle=False)
    epoch = int(target.get("epoch", 0))
    with torch.no_grad():
        stats = t.run_epoch(args, backbone, model, loader, criterion, optimizer, device, train=False, grounding_cache={}, epoch=epoch, task_balancer=None)
    metrics = stats["metrics"]
    actual = {
        "joint_test_score": 0.5 * float(metrics["Act_mF1"]) + 0.5 * float(metrics["Exp_mF1"]),
        "Act_mF1": float(metrics["Act_mF1"]),
        "Exp_mF1": float(metrics["Exp_mF1"]),
        "Exp_mAP": float(metrics["Exp_mAP"]),
        "Act_oF1": float(metrics["Act_oF1"]),
        "Exp_oF1": float(metrics["Exp_oF1"]),
    }
    tm = target["test_metrics"]
    expected = {
        "joint_test_score": float(target["joint_test_score"]),
        "Act_mF1": float(tm["Act_mF1"]),
        "Exp_mF1": float(tm["Exp_mF1"]),
        "Exp_mAP": float(tm["Exp_mAP"]),
        "Act_oF1": float(tm["Act_oF1"]),
        "Exp_oF1": float(tm["Exp_oF1"]),
    }
    diff = {k: abs(actual[k] - expected[k]) for k in expected}
    acceptance_keys = ["joint_test_score", "Act_mF1", "Exp_mF1", "Exp_mAP"]
    passed = all(diff[k] < 1e-5 for k in acceptance_keys)
    result = {
        "passed": passed,
        "acceptance_threshold": 1e-5,
        "acceptance_keys": acceptance_keys,
        "actual": actual,
        "expected": expected,
        "diff": diff,
        "count": int(stats.get("count", 0)),
        "checkpoint": str(ckpt),
        "args_json": str(args_json),
        "training_config_resolved": str(config_yaml),
        "run_manifest": str(manifest_json),
        "label_correlation_mode_used": args.label_correlation,
        "resume_state": {
            "start_epoch": int(resume_state.start_epoch),
            "missing_keys": resume_state.missing_keys or [],
            "unexpected_keys": resume_state.unexpected_keys or [],
        },
    }
    out = Path(cli.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if passed else 2

if __name__ == "__main__":
    raise SystemExit(main())


