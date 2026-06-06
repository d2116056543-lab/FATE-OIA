from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from fate_oia.datasets.diva_caf_oia_dataset import build_diva_caf_dataset, collate_diva_caf
from fate_oia.datasets.bdd100k_scene_state_proxy import BDD100KSceneStateProxy
from fate_oia.engine.eval_diva_caf_oia import evaluate_diva_caf
from fate_oia.losses.diva_caf_gradient_budget import apply_gradient_budget
from fate_oia.losses.diva_caf_losses import diva_caf_loss
from fate_oia.models.diva_caf_oia_model import DIVACAFOIAModel
from fate_oia.models.diva_multilayer_dino import build_dino_extractor
from fate_oia.utils.diva_caf_artifacts import write_json, append_jsonl, write_required_smoke_artifacts
from fate_oia.utils.diva_caf_manifest import write_run_manifest


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DIVA-CAF-OIA V2 direct-image model")
    parser.add_argument("--config", default="configs/fate_oia_train_360x640_diva_caf_oia_v2.yaml")
    parser.add_argument("--data_root", default=r"E:\sbw\BDD-OIA\data")
    parser.add_argument("--raw_root", default=r"E:\sbw\BDD-OIA")
    parser.add_argument("--bdd100k_root", default=r"E:\sbw\BDD100K")
    parser.add_argument("--pretrained_weights", default="")
    parser.add_argument("--checkpoint_key", default="")
    parser.add_argument("--dino_arch", default="vit_small")
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_feature_cache", action="store_true", default=True)
    parser.add_argument("--test_only", action="store_true", default=True)
    parser.add_argument("--print_every", type=int, default=200)
    parser.add_argument("--require_review_pass", action="store_true")
    return parser.parse_args()


def _lr_lambda(epoch: int, warmup: int, total: int, min_ratio: float) -> float:
    if warmup > 0 and epoch < warmup:
        return float(epoch + 1) / float(warmup)
    if total <= warmup:
        return 1.0
    progress = (epoch - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
    return min_ratio + (1.0 - min_ratio) * cosine


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.require_review_pass and not (Path(".background_runs") / "diva_caf_oia_v2_preflight" / "REVIEW_PASS_DIVA_CAF_OIA_V2.txt").exists():
        raise RuntimeError("RequireReviewPass enabled but REVIEW_PASS_DIVA_CAF_OIA_V2.txt is missing")
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    write_run_manifest(output_dir / "run_manifest.json", args, {"config": cfg, "feature_cache": False, "test_only": True})

    train_ds = build_diva_caf_dataset(args.data_root, args.raw_root, "train", max_samples=args.max_train_samples or None)
    test_ds = build_diva_caf_dataset(args.data_root, args.raw_root, "test", max_samples=args.max_test_samples or None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_diva_caf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_diva_caf)

    extractor = build_dino_extractor(
        arch=args.dino_arch,
        patch_size=args.patch_size,
        pretrained_weights=args.pretrained_weights or None,
        checkpoint_key=args.checkpoint_key or None,
        dim=args.dim,
        frozen=True,
    )
    effective_dim = getattr(extractor, "dim", args.dim)
    if hasattr(extractor, "backbone") and hasattr(extractor.backbone, "embed_dim"):
        effective_dim = int(extractor.backbone.embed_dim)
    model = DIVACAFOIAModel(dim=effective_dim, action_dim=4, reason_dim=21, dino_extractor=extractor).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.05)
    # CosineAnnealing-style schedule with explicit warmup.
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: _lr_lambda(e, args.warmup_epochs, args.epochs, args.min_lr / max(args.lr, 1e-12)))
    scene_proxy = BDD100KSceneStateProxy(args.bdd100k_root)
    best_joint = -1.0
    history: list[dict[str, Any]] = []
    last_out: dict[str, Any] | None = None
    last_grad_stats: dict[str, Any] = {}
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device)
            y_action = batch["action"].to(device)
            y_reason = batch["reason"].to(device)
            proxy = scene_proxy.for_file_names(batch["file_name"], device=device)
            out = model(images=images, labels={"action": y_action, "reason": y_reason}, train_mode=True, scene_state_proxy=proxy)
            _, terms = diva_caf_loss(out, y_action, y_reason)
            scaled, grad_stats = apply_gradient_budget(terms["main_loss"], terms["aux_loss"], list(model.parameters()), rho=0.15)
            loss = scaled / max(1, args.gradient_accumulation_steps)
            loss.backward()
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            model.factor_router.update_reliability(out["selected_vs_random_stats"].get("selected_vs_random_action_loss_drop", 0.0))
            last_out = out
            last_grad_stats = grad_stats
            append_jsonl(output_dir / "loss_components.jsonl", {"epoch": epoch, "batch": step, "lr": opt.param_groups[0]["lr"], "main_loss": terms["main_loss"], "aux_loss": terms["aux_loss"], "total_loss": terms["total_loss"], **grad_stats})
            if args.print_every > 0 and (step == 1 or step % args.print_every == 0):
                print(f"epoch={epoch} batch={step}/{len(train_loader)} lr={opt.param_groups[0]['lr']:.6g} total={float(terms['total_loss'].detach().cpu()):.4f} main={float(terms['main_loss'].detach().cpu()):.4f} aux={float(terms['aux_loss'].detach().cpu()):.4f}", flush=True)
        if step % args.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        scheduler.step()
        metrics, tensors = evaluate_diva_caf(model, test_loader, device=device)
        epoch_row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], **metrics}
        history.append(epoch_row)
        write_json(output_dir / "metrics_latest.json", metrics)
        write_json(output_dir / f"metrics_epoch_{epoch}.json", metrics)
        append_jsonl(output_dir / "metrics_summary.jsonl", epoch_row)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics, "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict()}, output_dir / "checkpoint_latest.pth")
        if metrics["joint"] > best_joint:
            best_joint = float(metrics["joint"])
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, output_dir / "checkpoint_best_test.pth")
            write_json(output_dir / "metrics_best_test.json", metrics)
        print(f"eval epoch={epoch} joint={metrics['joint']:.6f} Act_mF1={metrics['Act_mF1']:.6f} Exp_mF1={metrics['Exp_mF1']:.6f} Exp_mAP={metrics['Exp_mAP']:.6f}", flush=True)
        if last_out is not None:
            branch = {"epoch": epoch, "Act_mF1": metrics["Act_mF1"], "Act_fate_mF1": metrics.get("Act_fate_mF1"), "Act_eva_mF1": metrics.get("Act_eva_mF1"), "Act_actor_mF1": metrics.get("Act_actor_mF1"), "Exp_base_mF1": metrics.get("Exp_base_mF1"), "Exp_factor_mF1": metrics.get("Exp_factor_mF1"), "Exp_mF1": metrics["Exp_mF1"]}
            write_required_smoke_artifacts(output_dir, metrics, last_out, last_grad_stats, branch)
            write_json(output_dir / f"branch_metrics_epoch_{epoch}.json", branch)
    if args.epochs >= 32:
        write_json(output_dir / "GOAL_COMPLETED_DIVA_CAF_OIA_V2.json", {"completed_epochs": args.epochs, "best_joint": best_joint})


if __name__ == "__main__":
    main()
