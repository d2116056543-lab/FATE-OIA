from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

import fate_oia.engine.train_fate_oia as t
from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.engine.eval_snna25 import evaluate_snna25
from fate_oia.losses.specialist_losses import (
    action_preserve_loss,
    delta_l2_loss,
    evidence_distill_loss,
    hard_reason_ranking_loss,
    non_tail_distillation_loss,
    reason_asl_loss,
    sigmoid_f1_loss,
)
from fate_oia.models.action_set_head import action_set_accuracy, assign_action_patterns, build_action_patterns
from fate_oia.models.fate_oia_model import FATEOIAFeatureModel
from fate_oia.models.runc_integrated_specialist import RunCIntegratedSpecialist

ROOT = Path(__file__).resolve().parents[2]
RUNC_ART = ROOT / "run_c_artifacts"
RUN_C = {
    "joint": 0.5478436350822449,
    "act": 0.714386522769928,
    "exp": 0.38130074739456177,
    "ap": 0.36782199286279227,
}

RUNC_HARD_SETTINGS = {
    "image_height": 360,
    "image_width": 640,
    "patch_size": 8,
    "action_dim": 4,
    "reason_dim": 21,
    "token_compression": "keep_merge",
    "compression_keep_ratio_final": 0.70,
    "num_summary_tokens": 4,
}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(t._json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(t._json_safe(row), ensure_ascii=False) + "\n")


def _nearly_equal(a: Any, b: Any, tol: float = 1e-8) -> bool:
    if isinstance(b, float):
        try:
            return abs(float(a) - b) <= tol
        except (TypeError, ValueError):
            return False
    return a == b


def assert_no_runc_config_drift(args: Namespace, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Hard-stop if a run is no longer comparable to the Run C base protocol.

    The integrated-specialist experiment is meant to be an additive specialist
    on top of Run C, not a silent config rewrite. This guard runs before any
    expensive training setup and writes a machine-readable diff when possible.
    """

    diffs: list[dict[str, Any]] = []
    for key, expected in RUNC_HARD_SETTINGS.items():
        actual = getattr(args, key, None)
        if not _nearly_equal(actual, expected):
            diffs.append({"field": key, "expected": expected, "actual": actual})

    required_paths = {
        "runc_checkpoint": getattr(args, "runc_checkpoint", None),
        "runc_args": getattr(args, "runc_args", None),
        "runc_config": getattr(args, "runc_config", None),
    }
    for key, value in required_paths.items():
        if not value or not Path(value).exists():
            diffs.append({"field": key, "expected": "existing path", "actual": value})

    report = {
        "passed": not diffs,
        "hard_settings": RUNC_HARD_SETTINGS,
        "diffs": diffs,
        "runc_checkpoint": str(required_paths["runc_checkpoint"]),
        "runc_args": str(required_paths["runc_args"]),
        "runc_config": str(required_paths["runc_config"]),
    }
    if output_dir is not None:
        write_json(Path(output_dir) / "config_drift_report.json", report)
    if diffs:
        raise ValueError(f"Run C config drift detected: {diffs}")
    return report


def make_args(cli: argparse.Namespace) -> Namespace:
    base_args = read_json(cli.runc_args)
    args = Namespace(**base_args)
    overrides = vars(cli)
    for k, v in overrides.items():
        if v is not None:
            setattr(args, k, v)
    args.resume = str(Path(cli.runc_checkpoint))
    args.output_dir = cli.output_dir
    args.label_correlation = "self_attn_legacy" if t.checkpoint_uses_legacy_label_correlation(args.resume) else getattr(args, "label_correlation", "self_attn")
    args.label_correlation_legacy_detected = args.label_correlation == "self_attn_legacy"
    args.reason_logit_adjustment = None
    args.reason_logit_adjustment_tensor = None
    args.grounding_cache_jsonl = ""
    args.loss_grounding = 0.0
    args.grounding_mode = "none"
    args.loss_counterfactual = 0.0
    args.counterfactual_eval = False
    args.threshold_mode = "fixed"
    args.eval_threshold = 0.5
    args.max_saved_token_stats = int(getattr(args, "max_saved_token_stats", 8))
    args.effective_batch_size = int(args.batch_size) * int(args.gradient_accumulation_steps)
    return args


def build_base_model(args: Namespace, dim: int, device: torch.device) -> FATEOIAFeatureModel:
    label_bias_matrix = t.load_label_bias_matrix("", "none", args.action_dim + args.reason_dim)
    return FATEOIAFeatureModel(
        dim=dim,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        use_label_query=args.use_label_query,
        label_correlation=args.label_correlation,
        label_correlation_layers=args.label_correlation_layers,
        label_correlation_heads=args.label_correlation_heads,
        label_correlation_dropout=args.label_correlation_dropout,
        label_correlation_bias="none",
        label_correlation_bias_matrix=label_bias_matrix,
        label_correlation_bias_weight=0.0,
        fusion_mode="learned_gate",
        fusion_fixed_alpha=0.0,
        fusion_gate_floor=0.0,
    ).to(device)


def collect_train_actions(args: Namespace) -> torch.Tensor:
    ds = BDDOIAMultiTaskDataset(args.data_root, args.raw_root, split="train", action_dim=args.action_dim, reason_dim=args.reason_dim, load_image=False)
    if args.max_train_samples and args.max_train_samples > 0:
        samples = ds.samples[: int(args.max_train_samples)]
    else:
        samples = ds.samples
    return torch.stack([torch.tensor(s.action, dtype=torch.float32) for s in samples])


def compress(args: Namespace, original_tokens: torch.Tensor, epoch: int):
    keep_ratio = float(getattr(args, "compression_keep_ratio_final", 0.70))
    return t.compress_tokens(original_tokens, keep_ratio, args.num_summary_tokens, args.min_tokens, args.token_compression)


def logits_cat(action_logits: torch.Tensor, reason_logits: torch.Tensor) -> torch.Tensor:
    return torch.cat([action_logits, reason_logits], dim=1)


def branch_eval(logits: torch.Tensor, labels: torch.Tensor, action_dim: int) -> dict[str, Any]:
    return evaluate_snna25(logits.detach().cpu(), labels.detach().cpu(), action_dim, threshold_mode="fixed", fixed_threshold=0.5)["metrics"]


def make_optimizer(args: Namespace, model: RunCIntegratedSpecialist) -> torch.optim.Optimizer:
    groups = []
    if not args.freeze_base:
        groups.append({"params": list(model.base_fate_head.parameters()), "lr": args.lr_base, "name": "base"})
    else:
        for p in model.base_fate_head.parameters():
            p.requires_grad = False
    specialist_params = []
    for module in [model.reason_specialist, model.action_set_head, model.evidence_aux]:
        if module is not None:
            specialist_params += list(module.parameters())
    specialist_params.append(model.action_alpha_raw)
    groups.append({"params": specialist_params, "lr": args.lr_specialist, "name": "specialist"})
    groups.append({"params": [model.reason_bias], "lr": args.lr_bias, "name": "bias"})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def run_epoch(args, backbone, model, loader, optimizer, device, pattern_matrix, train: bool, epoch: int) -> dict[str, Any]:
    model.train(train)
    if train:
        optimizer.zero_grad(set_to_none=True)
    accum = max(1, int(args.gradient_accumulation_steps))
    total_loss = 0.0
    count = 0
    final_logits_rows = []
    base_logits_rows = []
    base_action_rows = []
    final_action_rows = []
    base_reason_rows = []
    final_reason_rows = []
    labels_rows = []
    loss_rows = []
    diag_rows = []
    token_rows = []
    pattern_ids_all = []
    pattern_logits_all = []
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = t.labels_from_batch(batch).to(device, non_blocking=True)
        action_gt = labels[:, : args.action_dim]
        reason_gt = labels[:, args.action_dim :]
        with torch.no_grad():
            original_tokens = t.extract_tokens(backbone, images, args.n_last_blocks)
        tokens, provenance, token_stats = compress(args, original_tokens, epoch)
        out = model(tokens)
        pattern_ids = assign_action_patterns(action_gt, pattern_matrix.to(device))
        action_asl = reason_asl_loss(out["final_action_logits"], action_gt) * float(args.loss_final_action)
        reason_asl = reason_asl_loss(out["final_reason_logits"], reason_gt) * float(args.loss_reason_asl)
        ranking = hard_reason_ranking_loss(out["final_reason_logits"], reason_gt, margin=args.ranking_margin, hard_k=args.ranking_hard_k) * float(args.loss_reason_ranking)
        f1_loss = sigmoid_f1_loss(out["final_reason_logits"], reason_gt) * float(args.loss_sigmoid_f1)
        pattern_loss = F.cross_entropy(out["pattern_logits"], pattern_ids) * float(args.loss_action_pattern)
        preserve = action_preserve_loss(out["final_action_logits"], out["base_action_logits"]) * float(args.loss_action_preserve)
        non_tail = non_tail_distillation_loss(out["final_reason_logits"], out["base_reason_logits"]) * float(args.loss_non_tail_distill)
        delta_l2 = delta_l2_loss(out["reason_delta_logits"]) * float(args.loss_delta_l2)
        ev_loss = evidence_distill_loss(out["final_reason_logits"], out.get("evidence_reason_logits")) * float(args.loss_evidence_distill)
        loss = action_asl + reason_asl + ranking + f1_loss + pattern_loss + preserve + non_tail + delta_l2 + ev_loss
        if train:
            (loss / float(accum)).backward()
            if ((step + 1) % accum == 0) or ((step + 1) == len(loader)):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        bs = int(images.shape[0])
        total_loss += float(loss.detach().item()) * bs
        count += bs
        final_logits_rows.append(logits_cat(out["final_action_logits"], out["final_reason_logits"]).detach().cpu())
        base_logits_rows.append(logits_cat(out["base_action_logits"], out["base_reason_logits"]).detach().cpu())
        base_action_rows.append(out["base_action_logits"].detach().cpu())
        final_action_rows.append(out["final_action_logits"].detach().cpu())
        base_reason_rows.append(out["base_reason_logits"].detach().cpu())
        final_reason_rows.append(out["final_reason_logits"].detach().cpu())
        labels_rows.append(labels.detach().cpu())
        pattern_ids_all.append(pattern_ids.detach().cpu())
        pattern_logits_all.append(out["pattern_logits"].detach().cpu())
        loss_rows.append({
            "train": train, "epoch": epoch, "step": step,
            "total_loss": float(loss.detach().item()),
            "action_asl": float(action_asl.detach().item()),
            "reason_asl": float(reason_asl.detach().item()),
            "ranking": float(ranking.detach().item()),
            "sigmoid_f1": float(f1_loss.detach().item()),
            "pattern_loss": float(pattern_loss.detach().item()),
            "action_preserve": float(preserve.detach().item()),
            "non_tail_distill": float(non_tail.detach().item()),
            "delta_l2": float(delta_l2.detach().item()),
            "evidence_distill": float(ev_loss.detach().item()),
            "lr_base": float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0,
            "effective_batch_size": int(args.effective_batch_size),
            "loss_divided_by_accumulation": True,
        })
        diag = out.get("diagnostics", {})
        diag_rows.append({"train": train, "epoch": epoch, "step": step, **diag})
        if len(token_rows) < int(getattr(args, "max_saved_token_stats", 8)):
            token_rows.append({"epoch": epoch, "step": step, **token_stats})
        if step % int(args.log_every) == 0:
            print(json.dumps({"event":"runc_integrated_batch", "train":train, "epoch":epoch, "step":step, "loss":float(loss.detach().item()), "batch_size":bs, "diag":diag, "token_stats":token_stats}, ensure_ascii=False), flush=True)
    final_logits = torch.cat(final_logits_rows, 0)
    base_logits = torch.cat(base_logits_rows, 0)
    labels_tensor = torch.cat(labels_rows, 0)
    pattern_ids_tensor = torch.cat(pattern_ids_all, 0)
    pattern_logits_tensor = torch.cat(pattern_logits_all, 0)
    final_metrics = branch_eval(final_logits, labels_tensor, args.action_dim)
    base_metrics = branch_eval(base_logits, labels_tensor, args.action_dim)
    pattern_stats = action_set_accuracy(pattern_logits_tensor, pattern_ids_tensor)
    return {
        "loss": total_loss / max(count, 1), "count": count,
        "metrics": final_metrics, "base_metrics": base_metrics,
        "branch_metrics": {"full_integrated": final_metrics, "runc_base": base_metrics, **pattern_stats},
        "logits": final_logits, "base_logits": base_logits,
        "base_action_logits": torch.cat(base_action_rows, 0),
        "final_action_logits": torch.cat(final_action_rows, 0),
        "base_reason_logits": torch.cat(base_reason_rows, 0),
        "final_reason_logits": torch.cat(final_reason_rows, 0),
        "labels": labels_tensor,
        "pattern_ids": pattern_ids_tensor,
        "pattern_logits": pattern_logits_tensor,
        "loss_components": loss_rows,
        "diagnostics": diag_rows,
        "token_stats": token_rows,
    }


def save_epoch(out_dir: Path, epoch: int, train_stats: dict, val_stats: dict, test_stats: dict, args: Namespace) -> dict[str, float]:
    epoch_dir = out_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "epoch": epoch,
        "train_loss": train_stats["loss"], "val_loss": val_stats["loss"], "test_loss": test_stats["loss"],
        "joint_test_score": 0.5 * test_stats["metrics"]["Act_mF1"] + 0.5 * test_stats["metrics"]["Exp_mF1"],
        "joint_val_score": 0.5 * val_stats["metrics"]["Act_mF1"] + 0.5 * val_stats["metrics"]["Exp_mF1"],
        "test_metrics": test_stats["metrics"], "val_metrics": val_stats["metrics"],
        "test_base_metrics": test_stats["base_metrics"], "val_base_metrics": val_stats["base_metrics"],
        "branch_metrics": test_stats["branch_metrics"],
        "runc_baseline": RUN_C,
    }
    write_json(epoch_dir / "metrics_summary.json", metrics)
    write_json(epoch_dir / "branch_metrics.json", test_stats["branch_metrics"])
    write_json(epoch_dir / "reason_delta_stats.json", {"train": train_stats["diagnostics"], "test": test_stats["diagnostics"][:8]})
    write_json(epoch_dir / "action_set_stats.json", {k: v for k, v in test_stats["branch_metrics"].items() if k.startswith("action_set")})
    write_json(epoch_dir / "evidence_stats.json", {"evidence_available": 0, "mode": "disabled_or_no_true_evidence"})
    write_json(epoch_dir / "per_label_reason_metrics.json", {"note": "per-label table not expanded in v1; logits are saved for offline audit"})
    write_jsonl(epoch_dir / "loss_components.jsonl", train_stats["loss_components"] + val_stats["loss_components"] + test_stats["loss_components"])
    write_jsonl(epoch_dir / "failure_cases.jsonl", [])
    for name, tensor in {
        "logits_base_action.pt": test_stats["base_action_logits"],
        "logits_final_action.pt": test_stats["final_action_logits"],
        "logits_base_reason.pt": test_stats["base_reason_logits"],
        "logits_final_reason.pt": test_stats["final_reason_logits"],
        "labels_action.pt": test_stats["labels"][:, : args.action_dim],
        "labels_reason.pt": test_stats["labels"][:, args.action_dim :],
    }.items():
        torch.save(tensor, epoch_dir / name)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Run C-preserved integrated specialist training.")
    ap.add_argument("--runc_checkpoint", default=str(RUNC_ART / "checkpoint_best_test.pth"))
    ap.add_argument("--runc_args", default=str(RUNC_ART / "args.json"))
    ap.add_argument("--runc_config", default=str(RUNC_ART / "training_config_resolved.yaml"))
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_root", default=r"E:\sbw\FATE_Drive\fate_oia_worktree\dataset\BDD-OIA")
    ap.add_argument("--raw_root", default=r"E:\sbw\FATE_Drive\fate_oia_worktree\raw_data\BDD-OIA")
    ap.add_argument("--pretrained_weights", default=r"E:\sbw\FATE_Drive\fate_oia_worktree\ckp\reference\dino_deitsmall8_pretrain.pth")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--lr_base", type=float, default=1e-5)
    ap.add_argument("--lr_specialist", type=float, default=1e-4)
    ap.add_argument("--lr_bias", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--freeze_base", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--freeze_dino", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reason_specialist", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--action_set_head", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--evidence_aux", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--loss_reason_asl", type=float, default=1.0)
    ap.add_argument("--loss_reason_ranking", type=float, default=0.15)
    ap.add_argument("--loss_sigmoid_f1", type=float, default=0.05)
    ap.add_argument("--loss_action_pattern", type=float, default=0.10)
    ap.add_argument("--loss_action_preserve", type=float, default=0.02)
    ap.add_argument("--loss_non_tail_distill", type=float, default=0.02)
    ap.add_argument("--loss_delta_l2", type=float, default=0.01)
    ap.add_argument("--loss_evidence_distill", type=float, default=0.05)
    ap.add_argument("--loss_final_action", type=float, default=0.25)
    ap.add_argument("--ranking_margin", type=float, default=0.5)
    ap.add_argument("--ranking_hard_k", type=int, default=5)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--max_saved_token_stats", type=int, default=8)
    args = make_args(ap.parse_args())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drift_report = assert_no_runc_config_drift(args, output_dir=out_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    backbone, dim = t.build_backbone(args, device)
    base = build_base_model(args, dim, device)
    dummy_optimizer = torch.optim.AdamW(base.parameters(), lr=args.lr_base)
    resume_state = t.load_resume_checkpoint(args.resume, base, dummy_optimizer, device=device, resume_optimizer=False, strict=True)
    if resume_state.missing_keys or resume_state.unexpected_keys:
        raise RuntimeError(f"Run C base load mismatch: missing={resume_state.missing_keys} unexpected={resume_state.unexpected_keys}")
    train_actions = collect_train_actions(args)
    pattern_matrix, pattern_meta = build_action_patterns(train_actions, top_k=16)
    write_json(out_dir / "action_patterns.json", pattern_meta)
    model = RunCIntegratedSpecialist(
        base_fate_head=base,
        dim=dim,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        pattern_matrix=pattern_matrix.to(device),
        enable_reason_specialist=args.reason_specialist,
        enable_action_set_head=args.action_set_head,
        enable_evidence_aux=args.evidence_aux,
    ).to(device)
    optimizer = make_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - resume_state.start_epoch), eta_min=args.min_lr) if args.scheduler == "cosine" else None
    train_loader = t.make_loader(args, "train", True)
    val_loader = t.make_loader(args, "val", False)
    test_loader = t.make_loader(args, "test", False)
    manifest = {"repo":"FATE-OIA", "mode":"runc_integrated_specialist_v1", "git_head": os.popen("git rev-parse HEAD").read().strip(), "hostname": socket.gethostname(), "args": vars(args), "resume_state": resume_state.__dict__, "run_c_baseline": RUN_C, "pattern_meta": pattern_meta, "config_drift_report": drift_report}
    write_json(out_dir / "run_manifest.json", manifest)
    write_json(out_dir / "args.json", vars(args))
    history = []
    best_test = -1.0
    best_val = -1.0
    start_epoch = int(resume_state.start_epoch)
    for epoch in range(start_epoch, int(args.epochs)):
        train_stats = run_epoch(args, backbone, model, train_loader, optimizer, device, pattern_matrix, True, epoch)
        val_stats = run_epoch(args, backbone, model, val_loader, optimizer, device, pattern_matrix, False, epoch)
        test_stats = run_epoch(args, backbone, model, test_loader, optimizer, device, pattern_matrix, False, epoch)
        metrics = save_epoch(out_dir, epoch, train_stats, val_stats, test_stats, args)
        history.append(metrics)
        if scheduler is not None:
            scheduler.step()
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None, "args": vars(args), "dim": dim, "best_test_score": best_test, "best_val_score": best_val, "pattern_matrix": pattern_matrix}
        torch.save(ckpt, out_dir / "checkpoint_latest.pth")
        if metrics["joint_test_score"] >= best_test:
            best_test = metrics["joint_test_score"]
            ckpt["best_test_score"] = best_test
            torch.save(ckpt, out_dir / "checkpoint_best_test.pth")
            write_json(out_dir / "metrics_best_test.json", metrics)
        if metrics["joint_val_score"] >= best_val:
            best_val = metrics["joint_val_score"]
            ckpt["best_val_score"] = best_val
            torch.save(ckpt, out_dir / "checkpoint_best_val.pth")
            write_json(out_dir / "metrics_best_val.json", metrics)
        write_json(out_dir / "metrics_latest.json", metrics)
        with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(t._json_safe(metrics), ensure_ascii=False) + "\n")
        print(json.dumps({"event":"runc_integrated_epoch", **t._json_safe(metrics)}, ensure_ascii=False), flush=True)
    write_json(out_dir / "history.json", history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
