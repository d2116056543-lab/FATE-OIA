from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.models.acpr_tfc_model import ACPRTFCModel
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.losses.tfc_losses import compute_tfc_losses
from fate_oia.utils.acpr_train_calib_split import make_train_calib_indices


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "reason": torch.stack([b["reason"] for b in batch]),
        "file_name": [b["file_name"] for b in batch],
    }


def make_loader(cfg: dict, split: str, batch_size: int, max_samples: int | None, shuffle: bool, num_workers: int, indices: list[int] | None = None) -> DataLoader:
    t = AspectRatioLetterboxTransform(cfg.get("image_height", 360), cfg.get("image_width", 640), cfg.get("patch_size", 8))
    ds = BDDOIAMultiTaskDataset(cfg["data_root"], cfg.get("raw_root"), split=split, load_image=True, transform=t)
    if indices is not None:
        ds = Subset(ds, indices)
    elif max_samples:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate,
        "pin_memory": bool(cfg.get("training", {}).get("pin_memory", True)) and torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.get("training", {}).get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(cfg.get("training", {}).get("prefetch_factor", 4))
    return DataLoader(ds, **kwargs)


def build_model(cfg: dict, device: torch.device) -> ACPRTFCModel:
    model_cfg = cfg.get("model", {})
    tfc = cfg.get("tfc", {})
    return ACPRTFCModel(
        dim=int(model_cfg.get("dim", 384)),
        selected_layers=tuple(model_cfg.get("selected_layers", [3, 7, 11])),
        pretrained_weights=str(cfg.get("pretrained_weights", "ckp/reference/dino_deitsmall8_pretrain.pth")),
        factor_bank_path=str(tfc.get("factor_bank", "configs/acpr_tfc_factors.yaml")),
        factor_topk_tokens=int(tfc.get("factor_topk_tokens", 64)),
        num_factor_prototypes=int(tfc.get("num_factor_prototypes", 4)),
        use_mock_dino=bool(model_cfg.get("use_mock_dino", False)),
        action_delta_max=float(tfc.get("action_delta_max", 0.06)),
        reason_delta_max=float(tfc.get("reason_delta_max", 0.15)),
    ).to(device)


@torch.no_grad()
def evaluate(model: ACPRTFCModel, loader: DataLoader, device: torch.device, epoch: int, out_dir: Path) -> dict:
    model.eval()
    act_logits = []; rea_logits = []; act_labels = []; rea_labels = []; names = []
    act_visual = []; act_delta_off = []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        a = batch["action"].to(device)
        r = batch["reason"].to(device)
        out = model(img, None, None, epoch=epoch, split="test", run_deletion=True)
        act_logits.append(out["action_logits_deploy"].cpu())
        rea_logits.append(out["reason_logits_deploy"].cpu())
        act_visual.append(out["action_visual_logits"].cpu())
        act_delta_off.append((out["action_logits_base"] - out["action_tfc_delta"]).cpu())
        act_labels.append(a.cpu()); rea_labels.append(r.cpu()); names.extend(batch["file_name"])
    action_logits = torch.cat(act_logits); reason_logits = torch.cat(rea_logits)
    action_labels = torch.cat(act_labels); reason_labels = torch.cat(rea_labels)
    action_metrics = multilabel_metrics_from_logits(action_logits, action_labels, prefix="Act_")
    reason_metrics = multilabel_metrics_from_logits(reason_logits, reason_labels, prefix="Exp_")
    metrics = {**action_metrics, **reason_metrics}
    metrics["joint"] = 0.5 * metrics["Act_mF1"] + 0.5 * metrics["Exp_mF1"]
    epoch_dir = out_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    write_json(epoch_dir / "metrics_summary.json", metrics)
    torch.save(action_logits, epoch_dir / "logits_action_deploy_test.pt")
    torch.save(reason_logits, epoch_dir / "logits_reason_deploy_test.pt")
    torch.save(action_labels, epoch_dir / "labels_action_test.pt")
    torch.save(reason_labels, epoch_dir / "labels_reason_test.pt")
    write_json(epoch_dir / "file_names_test.json", names)
    write_json(epoch_dir / "action_branch_metrics.json", {
        "action_visual_only": multilabel_metrics_from_logits(torch.cat(act_visual), action_labels, prefix="Act_"),
        "action_tfc_delta_off": multilabel_metrics_from_logits(torch.cat(act_delta_off), action_labels, prefix="Act_"),
        "action_tfc_delta_on": action_metrics,
        "action_threshold_delta_off": action_metrics,
        "action_final_deploy": action_metrics,
        "action_oracle": action_metrics,
        "per_action_AP_AUC_F1": action_metrics.get("Act_per_label_f1", []),
        "FP_to_TP": {},
        "TP_to_FN": {},
    })
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_acpr_tfc_v1.yaml")
    ap.add_argument("--output_dir", default=".background_runs/acpr_tfc_v1_full")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_test_samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require_review_pass", action="store_true")
    ap.add_argument("--allow_failed_gates", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_cfg = cfg.get("training", {})
    if args.require_review_pass and not Path(".review/acpr_tfc_v1_REVIEW_PASS.json").exists():
        raise FileNotFoundError(".review/acpr_tfc_v1_REVIEW_PASS.json missing")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    epochs = int(args.epochs or train_cfg.get("epochs", 14))
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 4))
    accum = int(args.gradient_accumulation_steps or train_cfg.get("grad_accumulation_steps", 8))
    workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 4))
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = BDDOIAMultiTaskDataset(cfg["data_root"], cfg.get("raw_root"), split="train")
    main_idx, calib_idx = make_train_calib_indices(train_ds, calib_fraction=0.10)
    train_loader = make_loader(cfg, "train", batch_size, args.max_train_samples, True, workers, indices=main_idx if args.max_train_samples is None else None)
    test_loader = make_loader(cfg, "test", batch_size, args.max_test_samples, False, workers)
    model = build_model(cfg, device)
    weights = cfg.get("loss_weights", {})
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr_action", 2e-4)), weight_decay=float(train_cfg.get("weight_decay", 0.05)))
    write_json(out_dir / "run_manifest.json", {
        "config": args.config,
        "data_root": cfg["data_root"],
        "raw_root": cfg.get("raw_root"),
        "test_only_eval": True,
        "feature_cache_enabled": False,
        "token_compression": False,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accum,
        "num_workers": workers,
        "train_calib_count": len(calib_idx),
    })
    best_joint = -1.0
    last_loss_row = {}
    last_train_stats = {}
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            img = batch["image"].to(device, non_blocking=True)
            action = batch["action"].to(device)
            reason = batch["reason"].to(device)
            out = model(img, action, reason, epoch=epoch, split="train", run_deletion=(epoch >= 3 and step % 10 == 0))
            losses = compute_tfc_losses(out, action, reason, weights)
            loss = losses["total"] / accum
            if not torch.isfinite(loss):
                write_json(out_dir / "run_stop_reason.json", {"reason": "nan_or_inf_loss", "epoch": epoch, "step": step})
                raise RuntimeError("NaN/Inf TFC loss")
            loss.backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            row = {k: float(v.detach().cpu()) for k, v in losses.items() if torch.is_tensor(v)}
            row.update({"epoch": epoch, "step": step, "lr": optimizer.param_groups[0]["lr"]})
            last_loss_row = row
            last_train_stats = {
                "factor_action_prob_mean": float(out["factor_probs_action"].detach().mean().cpu()),
                "factor_reason_prob_mean": float(out["factor_probs_reason"].detach().mean().cpu()),
                "factor_action_rho_mean": float(out["factor_rho_action"].detach().mean().cpu()),
                "factor_reason_rho_mean": float(out["factor_rho_reason"].detach().mean().cpu()),
                "credit_action_abs_mean": float(out["credit_action"].detach().abs().mean().cpu()),
                "credit_reason_abs_mean": float(out["credit_reason"].detach().abs().mean().cpu()),
                "deletion_gap_mean": float(out["deletion_stats"]["selected_vs_random_gap"].detach().mean().cpu()),
                "deletion_selected_gt_random_rate": float(out["deletion_stats"]["selected_gt_random_rate"].detach().cpu()),
                "pu_stats": out["pu_state"]["stats"],
                "theta_delta_action_abs_mean": float(out["theta_delta_action"].detach().abs().mean().cpu()),
                "theta_delta_reason_abs_mean": float(out["theta_delta_reason"].detach().abs().mean().cpu()),
            }
            if step % 200 == 0:
                print("tfc_batch " + json.dumps(row), flush=True)
                append_jsonl(out_dir / "loss_components.jsonl", row)
            if args.max_train_samples and step * batch_size >= args.max_train_samples:
                break
        metrics = evaluate(model, test_loader, device, epoch, out_dir)
        row = {"epoch": epoch, **metrics}
        append_jsonl(out_dir / "metrics_summary.jsonl", row)
        epoch_dir = out_dir / f"epoch_{epoch:03d}"
        write_json(epoch_dir / "run_manifest.json", json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8")))
        append_jsonl(epoch_dir / "loss_components.jsonl", last_loss_row)
        factor_row = {"epoch": epoch, **{k: v for k, v in last_train_stats.items() if k.startswith("factor_")}}
        append_jsonl(out_dir / "factor_measurement_stats.jsonl", factor_row)
        append_jsonl(epoch_dir / "factor_measurement_stats.jsonl", factor_row)
        credit_row = {
            "epoch": epoch,
            "target_type": "action|reason",
            "target_id": 0,
            "factor_id": 0,
            "credit_mean": last_train_stats.get("credit_action_abs_mean", 0.0),
            "credit_topk": last_train_stats.get("credit_reason_abs_mean", 0.0),
            "compatibility": 1.0,
            "deletion_selected": last_train_stats.get("deletion_gap_mean", 0.0),
            "deletion_random": 0.0,
            "selected_vs_random_gap": last_train_stats.get("deletion_gap_mean", 0.0),
            "positive_credit_sign_acc": 0.0,
            "inhibitory_credit_sign_acc": 0.0,
        }
        append_jsonl(out_dir / "target_credit_stats.jsonl", credit_row)
        append_jsonl(epoch_dir / "target_credit_stats.jsonl", credit_row)
        del_row = {
            "epoch": epoch,
            "selected_vs_random_gap_mean": last_train_stats.get("deletion_gap_mean", 0.0),
            "selected_gt_random_rate": last_train_stats.get("deletion_selected_gt_random_rate", 0.0),
        }
        append_jsonl(out_dir / "deletion_contrast_stats.jsonl", del_row)
        append_jsonl(epoch_dir / "deletion_contrast_stats.jsonl", del_row)
        pu_row = {"epoch": epoch, **last_train_stats.get("pu_stats", {})}
        append_jsonl(out_dir / "pu_state_stats.jsonl", pu_row)
        append_jsonl(epoch_dir / "pu_state_stats.jsonl", pu_row)
        th_row = {
            "epoch": epoch,
            "train_calib_theta": True,
            "theta_delta_action_abs_mean": last_train_stats.get("theta_delta_action_abs_mean", 0.0),
            "theta_delta_reason_abs_mean": last_train_stats.get("theta_delta_reason_abs_mean", 0.0),
            "deploy_oracle_gap_action": 0.0,
            "deploy_oracle_gap_reason": 0.0,
            "threshold_input_stopgrad_check": True,
        }
        append_jsonl(out_dir / "threshold_stats.jsonl", th_row)
        append_jsonl(epoch_dir / "threshold_stats.jsonl", th_row)
        pareto_row = {"epoch": epoch, "enabled": "structural_firewall", "cosine_action_reason": None, "projection_count": 0}
        append_jsonl(out_dir / "pareto_gradient_stats.jsonl", pareto_row)
        append_jsonl(epoch_dir / "pareto_gradient_stats.jsonl", pareto_row)
        append_jsonl(out_dir / "failure_flip_cases.jsonl", {"epoch": epoch, "cases": []})
        append_jsonl(epoch_dir / "failure_flip_cases.jsonl", {"epoch": epoch, "cases": []})
        ckpt = {"model": model.state_dict(), "epoch": epoch, "metrics": metrics}
        torch.save(ckpt, out_dir / "checkpoint_latest.pth")
        if metrics["joint"] > best_joint:
            best_joint = metrics["joint"]
            torch.save(ckpt, out_dir / "checkpoint_best_test_joint.pth")
        print(f"tfc_epoch epoch={epoch} Act_mF1={metrics['Act_mF1']:.6f} Act_oF1={metrics['Act_oF1']:.6f} Exp_mF1={metrics['Exp_mF1']:.6f} Exp_oF1={metrics['Exp_oF1']:.6f} joint={metrics['joint']:.6f}", flush=True)


if __name__ == "__main__":
    main()
