from __future__ import annotations

import argparse
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
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_feature_cache", action="store_true", default=True)
    parser.add_argument("--test_only", action="store_true", default=True)
    parser.add_argument("--print_every", type=int, default=200)
    parser.add_argument("--require_review_pass", action="store_true")
    return parser.parse_args()


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

    model = DIVACAFOIAModel(dim=args.dim, action_dim=4, reason_dim=21).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scene_proxy = BDD100KSceneStateProxy(args.bdd100k_root)
    best_joint = -1.0
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
                opt.step()
                opt.zero_grad(set_to_none=True)
            model.factor_router.update_reliability(out["factor_groups"], out["selected_factor_weights"].detach())
            last_out = out
            last_grad_stats = grad_stats
            append_jsonl(output_dir / "loss_components.jsonl", {"epoch": epoch, "batch": step, "main_loss": terms["main_loss"], "aux_loss": terms["aux_loss"], "total_loss": terms["total_loss"], **grad_stats})
            if args.print_every > 0 and (step == 1 or step % args.print_every == 0):
                print(f"epoch={epoch} batch={step}/{len(train_loader)} total={float(terms['total_loss'].detach().cpu()):.4f} main={float(terms['main_loss'].detach().cpu()):.4f} aux={float(terms['aux_loss'].detach().cpu()):.4f}", flush=True)
        metrics, tensors = evaluate_diva_caf(model, test_loader, device=device)
        write_json(output_dir / "metrics_latest.json", metrics)
        append_jsonl(output_dir / "metrics_summary.jsonl", {"epoch": epoch, **metrics})
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, output_dir / "checkpoint_latest.pth")
        if metrics["joint"] > best_joint:
            best_joint = float(metrics["joint"])
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, output_dir / "checkpoint_best_test.pth")
        print(f"eval epoch={epoch} joint={metrics['joint']:.6f} Act_mF1={metrics['Act_mF1']:.6f} Exp_mF1={metrics['Exp_mF1']:.6f} Exp_mAP={metrics['Exp_mAP']:.6f}", flush=True)
        if last_out is not None:
            branch = {"epoch": epoch, "Act_mF1": metrics["Act_mF1"], "Act_fate_mF1": metrics.get("Act_fate_mF1"), "Act_eva_mF1": metrics.get("Act_eva_mF1"), "Exp_mF1": metrics["Exp_mF1"]}
            write_required_smoke_artifacts(output_dir, metrics, last_out, last_grad_stats, branch)


if __name__ == "__main__":
    main()
