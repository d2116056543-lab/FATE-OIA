from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from fate_oia.engine.train_acpr_oia import (
    build_model,
    collect_base_logits,
    collect_threshold_teacher,
    dataset_label_rates,
    evaluate,
    load_config,
    make_dataset,
    make_loader,
)
from fate_oia.losses import acpr_threshold_losses as TL
from fate_oia.utils.acpr_artifacts import append_jsonl, write_json
from fate_oia.utils.acpr_train_calib_split import make_train_calib_indices


def _load_model_state(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported checkpoint payload: {path}")


def _threshold_loss_on_logits(model, action_logits, reason_logits, action_targets, reason_targets, cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    th = cfg.get("threshold", {})
    thresholded = model.threshold_head(action_logits.detach(), reason_logits.detach())
    losses = TL.calalign_loss_bundle(
        thresholded["action_logits_deploy"],
        thresholded["reason_logits_deploy"],
        action_targets,
        reason_targets,
        thresholded["threshold_logit"],
        model.threshold_head.theta_teacher,
        model.threshold_head.train_prior_theta,
        model.threshold_head.teacher_pred_rate,
        tau=float(th.get("soft_f1_tau_final", 0.20)),
        threshold_prob=thresholded["threshold_prob"],
        min_prob=torch.sigmoid(model.threshold_head.threshold_min_logit),
        max_prob=torch.sigmoid(model.threshold_head.threshold_max_logit),
        weights={
            "soft_f1_action": float(th.get("soft_f1_action_weight", 0.03)),
            "soft_f1_reason": float(th.get("soft_f1_reason_weight", 0.08)),
            "rate": float(th.get("rate_weight", 0.03)),
            "action_cardinality": float(th.get("action_cardinality_weight", 0.02)),
            "teacher": float(th.get("teacher_weight", 0.05)),
            "prior": float(th.get("prior_weight", 0.02)),
            "range": float(th.get("range_weight", 0.0)),
        },
    )
    return losses["total"], {k: float(v.detach().cpu()) for k, v in losses.items() if k != "total"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--freeze_backbone_and_trunk", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("threshold", {})["enabled"] = True
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size or int(cfg.get("training", {}).get("batch_size", 6))
    train_dataset = make_dataset(cfg, "train")
    train_main_indices, train_calib_indices = make_train_calib_indices(
        train_dataset,
        calib_fraction=float(cfg.get("threshold", {}).get("train_calib_fraction", 0.10)),
        seed=int(cfg.get("threshold", {}).get("split_seed", 20260615)),
    )
    train_calib_loader = make_loader(cfg, "train", batch_size, None, False, int(cfg.get("data", {}).get("num_workers", 0)), indices=train_calib_indices)
    test_loader = make_loader(cfg, "test", batch_size, None, False, int(cfg.get("data", {}).get("num_workers", 0)))

    model = build_model(cfg, device)
    missing, unexpected = model.load_state_dict(_load_model_state(args.checkpoint), strict=False)
    if args.freeze_backbone_and_trunk:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("threshold_head.")

    action_rate, reason_rate = dataset_label_rates(torch.utils.data.Subset(train_dataset, train_calib_indices))
    model.threshold_head.initialize_from_label_stats(action_rate.to(device), reason_rate.to(device), cfg.get("grammar", {}).get("tail_indices"))
    opt = torch.optim.AdamW(
        [p for p in model.threshold_head.parameters() if p.requires_grad],
        lr=float(cfg.get("threshold", {}).get("lr_threshold", 7e-4)),
        weight_decay=float(cfg.get("threshold", {}).get("weight_decay_threshold", 0.0)),
    )
    write_json(
        out_dir / "threshold_only_manifest.json",
        {
            "checkpoint": str(args.checkpoint),
            "config": args.config,
            "train_main_count": len(train_main_indices),
            "train_calib_count": len(train_calib_indices),
            "missing_checkpoint_keys": list(missing),
            "unexpected_checkpoint_keys": list(unexpected),
            "teacher_source": "train_calib",
            "test_threshold_usage": "oracle_diagnostic_only",
        },
    )

    best = -1.0
    for epoch in range(int(args.epochs)):
        teacher = collect_threshold_teacher(model, train_calib_loader, device, epoch, cfg)
        model.threshold_head.update_teacher(
            teacher["threshold_logit"].to(device),
            pred_rate_teacher=teacher["pred_rate"].to(device),
            ema=float(cfg.get("threshold", {}).get("teacher_ema", 0.20)),
            copy_to_params=True,
        )
        action_logits, reason_logits, action_labels, reason_labels = collect_base_logits(model, train_calib_loader, device, epoch)
        action_logits = action_logits.to(device)
        reason_logits = reason_logits.to(device)
        action_labels = action_labels.to(device)
        reason_labels = reason_labels.to(device)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, parts = _threshold_loss_on_logits(model, action_logits, reason_logits, action_labels, reason_labels, cfg)
        loss.backward()
        opt.step()
        metrics = evaluate(model, test_loader, device, epoch, out_dir)
        row = {
            "event": "acpr_threshold_only_epoch",
            "epoch": epoch,
            "loss_threshold_total": float(loss.detach().cpu()),
            **parts,
            **metrics,
        }
        append_jsonl(out_dir / "threshold_only_metrics.jsonl", row)
        write_json(
            out_dir / f"threshold_table_epoch_{epoch:03d}.json",
            {
                "source": "train_calib",
                "threshold_prob": teacher["threshold_prob"].tolist(),
                "teacher_best_f1": teacher["best_f1"].tolist(),
                "teacher_pred_rate": teacher["pred_rate"].tolist(),
                "learned_threshold_prob": torch.sigmoid(model.threshold_head.compose_theta()).detach().cpu().tolist(),
            },
        )
        score = float(metrics.get("final_raw_joint", 0.0))
        if score >= best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "checkpoint_threshold_head_best.pth")
        print(json.dumps({"epoch": epoch, "threshold_only_joint": score, "best": best}), flush=True)


if __name__ == "__main__":
    main()
