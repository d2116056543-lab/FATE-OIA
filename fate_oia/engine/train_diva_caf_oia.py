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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DIVA-CAF-OIA V2 direct-image model")
    parser.add_argument("--config", default="configs/fate_oia_train_360x640_diva_caf_oia_v2.yaml")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--raw_root", default=None)
    parser.add_argument("--bdd100k_root", default=None)
    parser.add_argument("--pretrained_weights", default=None)
    parser.add_argument("--checkpoint_key", default=None)
    parser.add_argument("--dino_arch", default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--delta_cap", type=float, default=None)
    parser.add_argument("--reason_cap", type=float, default=None)
    parser.add_argument("--factor_topk", type=int, default=None)
    parser.add_argument("--group_topk", type=int, default=None)
    parser.add_argument("--layer_indices", default=None)
    parser.add_argument("--image_height", type=int, default=None)
    parser.add_argument("--image_width", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_feature_cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--test_only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--print_every", type=int, default=200)
    parser.add_argument("--require_review_pass", action="store_true")
    parser.add_argument("--resume_checkpoint", default=None, help="Resume model/optimizer/scheduler from checkpoint_latest.pth")
    return parser.parse_args(argv)


def _cfg_get(cfg: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    resolved = {
        "data": {
            "data_root": args.data_root or _cfg_get(cfg, ("data", "data_root"), r"E:\sbw\BDD-OIA\data"),
            "raw_root": args.raw_root or _cfg_get(cfg, ("data", "raw_root"), r"E:\sbw\BDD-OIA"),
            "bdd100k_root": args.bdd100k_root or _cfg_get(cfg, ("data", "bdd100k_root"), r"E:\sbw\BDD100K"),
            "image_height": int(args.image_height or _cfg_get(cfg, ("data", "image_height"), 360)),
            "image_width": int(args.image_width or _cfg_get(cfg, ("data", "image_width"), 640)),
            "patch_size": int(args.patch_size or _cfg_get(cfg, ("data", "patch_size"), _cfg_get(cfg, ("model", "patch_size"), 8))),
            "eval_splits": _cfg_get(cfg, ("data", "eval_splits"), ["test"]),
            "no_feature_cache": bool(args.no_feature_cache if args.no_feature_cache is not None else _cfg_get(cfg, ("data", "no_feature_cache"), True)),
        },
        "model": {
            "dim": int(args.dim or _cfg_get(cfg, ("model", "dim"), 384)),
            "layer_indices": [int(x) for x in (args.layer_indices.split(",") if isinstance(args.layer_indices, str) else _cfg_get(cfg, ("model", "layer_indices"), [3, 6, 9, 12]))],
            "delta_cap": float(args.delta_cap or _cfg_get(cfg, ("model", "delta_cap"), _cfg_get(cfg, ("diva", "visual_delta_cap"), 0.08))),
            "reason_cap": float(args.reason_cap or _cfg_get(cfg, ("model", "reason_cap"), _cfg_get(cfg, ("reason", "reason_cap"), 0.25))),
            "action_dim": int(_cfg_get(cfg, ("model", "action_dim"), 4)),
            "reason_dim": int(_cfg_get(cfg, ("model", "reason_dim"), 21)),
        },
        "backbone": {
            "pretrained_weights": args.pretrained_weights or _cfg_get(cfg, ("backbone", "pretrained_weights"), ""),
            "checkpoint_key": args.checkpoint_key if args.checkpoint_key is not None else _cfg_get(cfg, ("backbone", "checkpoint_key"), ""),
            "dino_arch": args.dino_arch or _cfg_get(cfg, ("backbone", "dino_arch"), _cfg_get(cfg, ("backbone", "dino_variant"), "vit_small")),
            "dino_frozen": bool(_cfg_get(cfg, ("backbone", "dino_frozen"), True)),
        },
        "caf": {
            "factor_topk": int(args.factor_topk or _cfg_get(cfg, ("caf", "factor_topk"), 3)),
            "group_topk": int(args.group_topk or _cfg_get(cfg, ("caf", "factor_group_topk"), _cfg_get(cfg, ("caf", "group_topk"), 3))),
        },
        "training": {
            "epochs": int(args.epochs or _cfg_get(cfg, ("training", "epochs"), 1)),
            "batch_size": int(args.batch_size or _cfg_get(cfg, ("training", "batch_size"), 2)),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps or _cfg_get(cfg, ("training", "gradient_accumulation_steps"), 2)),
            "lr": float(args.lr or _cfg_get(cfg, ("training", "lr"), 3e-4)),
            "min_lr": float(args.min_lr or _cfg_get(cfg, ("training", "min_lr"), 1e-5)),
            "warmup_epochs": int(args.warmup_epochs if args.warmup_epochs is not None else _cfg_get(cfg, ("training", "warmup_epochs"), 2)),
            "weight_decay": float(args.weight_decay if args.weight_decay is not None else _cfg_get(cfg, ("training", "weight_decay"), 0.05)),
            "test_only": bool(args.test_only if args.test_only is not None else True),
        },
        "raw_config": cfg,
    }
    return resolved


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
    resolved = resolve_config(args)
    cfg = resolved["raw_config"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.require_review_pass and not (Path(".background_runs") / "diva_caf_oia_v2_1_preflight" / "REVIEW_PASS_DIVA_CAF_OIA_V2_1.txt").exists():
        raise RuntimeError("RequireReviewPass enabled but REVIEW_PASS_DIVA_CAF_OIA_V2_1.txt is missing")
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    write_run_manifest(output_dir / "run_manifest.json", args, {"config": cfg, "resolved_config": resolved, "feature_cache": False, "test_only": resolved["training"]["test_only"]})

    train_ds = build_diva_caf_dataset(resolved["data"]["data_root"], resolved["data"]["raw_root"], "train", height=resolved["data"]["image_height"], width=resolved["data"]["image_width"], max_samples=args.max_train_samples or None)
    test_ds = build_diva_caf_dataset(resolved["data"]["data_root"], resolved["data"]["raw_root"], "test", height=resolved["data"]["image_height"], width=resolved["data"]["image_width"], max_samples=args.max_test_samples or None)
    train_loader = DataLoader(train_ds, batch_size=resolved["training"]["batch_size"], shuffle=True, num_workers=0, collate_fn=collate_diva_caf)
    test_loader = DataLoader(test_ds, batch_size=resolved["training"]["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_diva_caf)

    extractor = build_dino_extractor(
        arch=resolved["backbone"]["dino_arch"],
        patch_size=resolved["data"]["patch_size"],
        pretrained_weights=resolved["backbone"]["pretrained_weights"] or None,
        checkpoint_key=resolved["backbone"]["checkpoint_key"] or None,
        layer_indices=tuple(resolved["model"]["layer_indices"]),
        dim=resolved["model"]["dim"],
        frozen=resolved["backbone"]["dino_frozen"],
    )
    effective_dim = getattr(extractor, "dim", resolved["model"]["dim"])
    if hasattr(extractor, "backbone") and hasattr(extractor.backbone, "embed_dim"):
        effective_dim = int(extractor.backbone.embed_dim)
    model = DIVACAFOIAModel(
        dim=effective_dim,
        action_dim=resolved["model"]["action_dim"],
        reason_dim=resolved["model"]["reason_dim"],
        dino_extractor=extractor,
        layer_indices=tuple(resolved["model"]["layer_indices"]),
        delta_cap=resolved["model"]["delta_cap"],
        reason_cap=resolved["model"]["reason_cap"],
        factor_topk=resolved["caf"]["factor_topk"],
    ).to(device)
    model.factor_router.group_topk = resolved["caf"]["group_topk"]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=resolved["training"]["lr"], weight_decay=resolved["training"]["weight_decay"])
    # CosineAnnealing-style schedule with explicit warmup.
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: _lr_lambda(e, resolved["training"]["warmup_epochs"], resolved["training"]["epochs"], resolved["training"]["min_lr"] / max(resolved["training"]["lr"], 1e-12)))
    scene_proxy = BDD100KSceneStateProxy(resolved["data"]["bdd100k_root"])
    best_joint = -1.0
    history: list[dict[str, Any]] = []
    start_epoch = 0
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        hist_path = output_dir / "history.json"
        if hist_path.exists():
            history = json.loads(hist_path.read_text(encoding="utf-8"))
            if history:
                best_joint = max(float(row.get("joint", -1.0)) for row in history)
        elif "metrics" in ckpt:
            best_joint = float(ckpt["metrics"].get("joint", -1.0))
        write_json(output_dir / "resume_state.json", {
            "resume_checkpoint": str(resume_path),
            "resume_epoch": int(ckpt.get("epoch", -1)),
            "start_epoch": start_epoch,
            "best_joint_before_resume": best_joint,
        })
    last_out: dict[str, Any] | None = None
    last_grad_stats: dict[str, Any] = {}
    for epoch in range(start_epoch, resolved["training"]["epochs"]):
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
            loss = scaled / max(1, resolved["training"]["gradient_accumulation_steps"])
            loss.backward()
            if step % resolved["training"]["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            sv = out["selected_vs_random_stats"]
            model.factor_router.update_reliability(
                selected_vs_random_per_action_group=sv.get("per_action_group_selected_minus_random", torch.zeros_like(model.factor_router.faith_ema)),
                help_delta_per_action_group=torch.relu(sv.get("per_action_group_selected_minus_random", torch.zeros_like(model.factor_router.faith_ema))),
                hurt_delta_per_action_group=torch.relu(-sv.get("per_action_group_selected_minus_random", torch.zeros_like(model.factor_router.faith_ema))),
            )
            last_out = out
            last_grad_stats = grad_stats
            append_jsonl(output_dir / "loss_components.jsonl", {"epoch": epoch, "batch": step, "lr": opt.param_groups[0]["lr"], "main_loss": terms["main_loss"], "aux_loss": terms["aux_loss"], "total_loss": terms["total_loss"], **grad_stats})
            if args.print_every > 0 and (step == 1 or step % args.print_every == 0):
                print(f"epoch={epoch} batch={step}/{len(train_loader)} lr={opt.param_groups[0]['lr']:.6g} total={float(terms['total_loss'].detach().cpu()):.4f} main={float(terms['main_loss'].detach().cpu()):.4f} aux={float(terms['aux_loss'].detach().cpu()):.4f}", flush=True)
        if step % resolved["training"]["gradient_accumulation_steps"] != 0:
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
    if resolved["training"]["epochs"] >= 32:
        write_json(output_dir / "GOAL_COMPLETED_DIVA_CAF_OIA_V2_1.json", {"completed_epochs": resolved["training"]["epochs"], "best_joint": best_joint})


if __name__ == "__main__":
    main()
