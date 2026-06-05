from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.engine.eval_egcaf_oia import evaluate_logits
from fate_oia.losses.egcaf_losses import EGCafLoss
from fate_oia.models.egcaf_oia_model import EGCafOIAModel
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.utils.egcaf_artifacts import write_epoch_factor_artifacts, write_json


def _load_yaml_flat(path: str) -> dict[str, Any]:
    if not path:
        return {}
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
    return data if isinstance(data, dict) else {}


def _apply_defaults(args: argparse.Namespace, cfg: dict[str, Any]) -> argparse.Namespace:
    cli = set(sys.argv[1:])
    for k, v in cfg.items():
        if hasattr(args, k) and f"--{k}" not in cli:
            setattr(args, k, v)
    if isinstance(getattr(args, "image_size", None), list):
        args.image_height, args.image_width = int(args.image_size[0]), int(args.image_size[1])
    if isinstance(getattr(args, "hook_layers", None), str):
        args.hook_layers = [int(x) for x in args.hook_layers.replace("[","").replace("]","").split(",") if x.strip()]
    return args


def _limited(ds, n: int):
    return Subset(ds, list(range(min(n, len(ds))))) if n and n > 0 else ds


def make_loader(args: argparse.Namespace, split: str, shuffle: bool) -> DataLoader:
    tfm = AspectRatioLetterboxTransform(args.image_height, args.image_width, patch_size=args.patch_size, return_meta=True)
    ds = BDDOIAMultiTaskDataset(args.data_root, args.raw_root, split=split, action_dim=args.action_dim, reason_dim=args.reason_dim, load_image=True, transform=tfm)
    max_n = args.max_train_samples if split == "train" else args.max_test_samples
    ds = _limited(ds, max_n)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    return batch["image"].to(device), batch["action"].float().to(device), batch["reason"].float().to(device), list(batch["file_name"])


@torch.no_grad()
def evaluate(model: EGCafOIAModel, loader: DataLoader, device: torch.device, out_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    model.eval()
    ac, af, ag, rr, ya, yr, names = [], [], [], [], [], [], []
    last_outputs: dict[str, Any] | None = None
    for batch in loader:
        images, y_action, y_reason, file_names = _batch_to_device(batch, device)
        outputs = model(images, return_artifacts=True, mode="test")
        ac.append(outputs["action_core_logits"].detach().cpu())
        af.append(outputs["action_final_logits"].detach().cpu())
        ag.append(outputs["guarded_action_logits"].detach().cpu())
        rr.append(outputs["reason_logits"].detach().cpu())
        ya.append(y_action.detach().cpu())
        yr.append(y_reason.detach().cpu())
        names.extend(file_names)
        if last_outputs is None:
            last_outputs = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in outputs.items() if k not in {"reason_attention", "reason_memory_tokens"}}
    tensors = {
        "action_core": torch.cat(ac), "action_final": torch.cat(af), "guarded_action": torch.cat(ag),
        "reason": torch.cat(rr), "labels_action": torch.cat(ya), "labels_reason": torch.cat(yr),
    }
    result = evaluate_logits(tensors["action_core"], tensors["action_final"], tensors["guarded_action"], tensors["reason"], tensors["labels_action"], tensors["labels_reason"])
    if out_dir:
        logit_dir = out_dir / "logits"; logit_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tensors["action_core"], logit_dir / "action_core_test.pt")
        torch.save(tensors["action_final"], logit_dir / "action_final_test.pt")
        torch.save(tensors["guarded_action"], logit_dir / "guarded_action_test.pt")
        torch.save(tensors["reason"], logit_dir / "reason_test.pt")
        torch.save(tensors["labels_action"], logit_dir / "labels_action_test.pt")
        torch.save(tensors["labels_reason"], logit_dir / "labels_reason_test.pt")
        (logit_dir / "file_names_test.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    return result, tensors, last_outputs or {}


def _lr_for_epoch(base: float, epoch: int, epochs: int, warmup: int, min_lr: float) -> float:
    if epoch < warmup:
        return base * float(epoch + 1) / float(max(warmup, 1))
    t = (epoch - warmup) / max(epochs - warmup, 1)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_manifest.json", {
        "repo": "FATE-OIA", "method": "EG-CAF-OIA V1", "command": " ".join(sys.argv),
        "hostname": socket.gethostname(), "python": sys.executable, "device": str(device),
        "direct_image_training": True, "no_feature_cache": bool(args.no_feature_cache), "test_only_eval": bool(args.test_only),
        "best_selection_split": "test", "batch_size": args.batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "image_height": args.image_height, "image_width": args.image_width, "pretrained_weights": args.pretrained_weights,
    })
    train_loader = make_loader(args, "train", True)
    test_loader = make_loader(args, "test", False)
    model = EGCafOIAModel(
        action_dim=args.action_dim, reason_dim=args.reason_dim, hidden_dim=args.hidden_dim,
        pretrained_weights=args.pretrained_weights, patch_size=args.patch_size, hook_layers=args.hook_layers,
        lightweight_backbone=args.lightweight_backbone, residual_cap=args.residual_cap,
    ).to(device)
    criterion = EGCafLoss(rho=args.auxiliary_budget_rho, sufficiency_weight=args.sufficiency_weight, comprehensiveness_weight=args.comprehensiveness_weight)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))
    best = -1e9
    history = []
    for epoch in range(args.epochs):
        model.train()
        lr = _lr_for_epoch(args.lr, epoch, args.epochs, args.warmup_epochs, args.min_lr)
        _set_lr(opt, lr)
        running = []
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            images, y_action, y_reason, _ = _batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                outputs = model(images, return_artifacts=(step == 1), mode="train")
                loss, stats = criterion(outputs, y_action, y_reason, shared_params=list(model.actor.parameters()) + list(model.reason_decoder.parameters()))
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            running.append(stats["total_loss"])
            if step % args.print_every == 0:
                print(f"epoch={epoch} batch={step}/{len(train_loader)} loss={sum(running)/len(running):.5f} lr={lr:.8f}", flush=True)
            if args.max_train_batches and step >= args.max_train_batches:
                break
        epoch_dir = out / f"epoch_{epoch:03d}"
        metrics, _, eval_outputs = evaluate(model, test_loader, device, epoch_dir)
        flat = {
            "epoch": epoch,
            "train_loss": float(sum(running) / max(len(running), 1)),
            "lr": lr,
            "action_core_mF1": metrics["action_core"]["Act_mF1"],
            "action_final_mF1": metrics["action_final"]["Act_mF1"],
            "guarded_action_mF1": metrics["guarded_action"]["Act_mF1"],
            "Exp_mF1": metrics["reason"]["Exp_mF1"],
            "Exp_mAP": metrics["reason"]["Exp_mAP"],
            "joint_core": metrics["joint_core"],
            "joint_final": metrics["joint_final"],
            "joint_guarded": metrics["joint_guarded"],
        }
        eval_outputs["gradient_budget_stats"] = {"available": True, "last_batch": stats}
        write_epoch_factor_artifacts(epoch_dir, eval_outputs, flat)
        write_json(epoch_dir / "metrics.json", flat)
        with (out / "metrics_summary.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(flat, ensure_ascii=False) + "\n")
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": flat}, out / "checkpoint_latest.pth")
        score = flat["joint_guarded"]
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": flat}, out / "checkpoint_best_test.pth")
            write_json(out / "best_metrics.json", flat)
        history.append(flat)
        write_json(out / "history.json", history)
        print(f"epoch={epoch} TEST joint_guarded={flat['joint_guarded']:.6f} action_core={flat['action_core_mF1']:.6f} action_final={flat['action_final_mF1']:.6f} guarded={flat['guarded_action_mF1']:.6f} Exp_mF1={flat['Exp_mF1']:.6f} Exp_mAP={flat['Exp_mAP']:.6f}", flush=True)
    return history[-1] if history else {}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--data_root", default=r"E:\sbw\BDD-OIA\data")
    ap.add_argument("--raw_root", default=r"E:\sbw\BDD-OIA")
    ap.add_argument("--output_dir", default=r".background_runs\egcaf_oia_v1_full")
    ap.add_argument("--pretrained_weights", default=r"ckp\reference\dino_deitsmall8_pretrain.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", "--grad_accum", dest="gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--max_train_batches", type=int, default=0)
    ap.add_argument("--image_height", type=int, default=360)
    ap.add_argument("--image_width", type=int, default=640)
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--hook_layers", type=int, nargs="*", default=[3,6,9,12])
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--auxiliary_budget_rho", type=float, default=0.10)
    ap.add_argument("--sufficiency_weight", type=float, default=0.10)
    ap.add_argument("--comprehensiveness_weight", type=float, default=0.10)
    ap.add_argument("--residual_cap", type=float, default=0.03)
    ap.add_argument("--print_every", type=int, default=200)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_feature_cache", action="store_true")
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--lightweight_backbone", action="store_true")
    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = _load_yaml_flat(args.config)
    args = _apply_defaults(args, cfg)
    run(args)


if __name__ == "__main__":
    main()
