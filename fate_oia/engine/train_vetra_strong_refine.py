from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from fate_oia.datasets.aie_splits import stable_split_ids, write_split_manifest
from fate_oia.engine.train_aie_oia import (
    build_model,
    canonical_model_state_dict,
    make_dataset,
    make_loader,
)
from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.losses.vetra_strong_rank_losses import (
    action_pairwise_ap_loss,
    action_smooth_ap_loss,
    base_margin_trust_loss,
    residual_energy_loss,
)
from fate_oia.models.vetra_strong_refiner import (
    SelectiveActionPathRefiner,
    SelectiveVisualActionRankRefiner,
)
from fate_oia.utils.acpr_threshold_search import search_best_thresholds_for_f1
from fate_oia.utils.aie_calibration import apply_posthoc_threshold
from fate_oia.utils.aie_metrics import aie_branch_metrics


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_base(cfg: dict, checkpoint_path: str, device: torch.device):
    model = build_model(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(canonical_model_state_dict(checkpoint["model"]), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_refiner(base, cfg: dict):
    refiner_cfg = cfg["refiner"]
    refiner_type = str(refiner_cfg.get("type", "path"))
    if refiner_type == "rank":
        return SelectiveVisualActionRankRefiner(
            dim=int(cfg["primary"]["dim"]),
            rank=int(refiner_cfg["rank"]),
            action_dim=4,
            max_delta=float(refiner_cfg["max_delta"]),
        )
    if refiner_type == "path":
        return SelectiveActionPathRefiner(
            base.action_evidence,
            base.action_contribution,
            action_dim=4,
            max_delta=float(refiner_cfg["max_delta"]),
        )
    raise ValueError(f"unsupported refiner.type: {refiner_type}")


def run_refiner(refiner, source, action_scale: float, gain=None):
    if isinstance(refiner, SelectiveVisualActionRankRefiner):
        return refiner(
            source["action_logits_final"].detach(),
            source["reason_logits_final"],
            source["action_nodes_primary"],
            source["evidence_token"],
            gain=gain,
        )
    return refiner(source, action_scale=action_scale, gain=gain)


@torch.no_grad()
def base_forward(base, images, cfg):
    field = base.encode_images(images)
    return base.decode_from_field(
        field,
        action_scale=float(cfg["evidence"]["action_scale"]),
        reason_scale=float(cfg["reason_private"]["reason_scale"]),
    )


@torch.no_grad()
def collect(base, refiner, loader, device, cfg, gain=None):
    store = {key: [] for key in (
        "base_action", "final_action", "reason", "delta", "action_target", "reason_target"
    )}
    names = []
    base.eval(); refiner.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = base_forward(base, images, cfg)
            refined = run_refiner(
                refiner, output, action_scale=float(cfg["evidence"]["action_scale"]), gain=gain
            )
        mapping = {
            "base_action": output["action_logits_final"],
            "final_action": refined["action_logits_final"],
            "reason": refined["reason_logits_final"],
            "delta": refined["action_delta_unscaled"],
            "action_target": batch["action"],
            "reason_target": batch["reason"],
        }
        for key, value in mapping.items():
            store[key].append(value.detach().float().cpu())
        names.extend(batch["file_name"])
    return {key: torch.cat(value) for key, value in store.items()} | {"file_name": names}


def label_f1(logits, targets, threshold):
    prediction = torch.sigmoid(logits) >= float(threshold)
    target = targets.bool()
    tp = (prediction & target).sum().float()
    fp = (prediction & ~target).sum().float()
    fn = (~prediction & target).sum().float()
    return float(2 * tp / (2 * tp + fp + fn).clamp_min(1))


def select_gains(calib, audit, candidates):
    gains = []
    rows = []
    grid = torch.arange(0.01, 0.9501, 0.005)
    for action in range(4):
        action_rows = []
        for gain in candidates:
            calib_logits = calib["base_action"][:, action] + float(gain) * calib["delta"][:, action]
            result = search_best_thresholds_for_f1(
                calib_logits[:, None], calib["action_target"][:, action, None], grid=grid
            )
            threshold = float(result["threshold_prob"][0])
            audit_logits = audit["base_action"][:, action] + float(gain) * audit["delta"][:, action]
            score = label_f1(audit_logits, audit["action_target"][:, action], threshold)
            action_rows.append({"gain": float(gain), "threshold": threshold, "audit_f1": score})
        best = max(action_rows, key=lambda row: (row["audit_f1"], -row["gain"]))
        gains.append(best["gain"])
        rows.append(action_rows)
    return torch.tensor(gains), rows


def fit_locked_thresholds(calib, audit, gain, reason_offset):
    action_logits = torch.cat((
        calib["base_action"] + calib["delta"] * gain,
        audit["base_action"] + audit["delta"] * gain,
    ))
    action_target = torch.cat((calib["action_target"], audit["action_target"]))
    reason_logits = torch.cat((calib["reason"], audit["reason"]))
    reason_target = torch.cat((calib["reason_target"], audit["reason_target"]))
    grid = torch.arange(0.01, 0.9501, 0.005)
    action_threshold = search_best_thresholds_for_f1(action_logits, action_target, grid=grid)["threshold_prob"]
    reason_threshold = search_best_thresholds_for_f1(reason_logits, reason_target, grid=grid)["threshold_prob"]
    reason_threshold = (reason_threshold + float(reason_offset)).clamp(0.01, 0.95)
    return torch.cat((action_threshold, reason_threshold))


def evaluate(base, refiner, calib_loader, audit_loader, test_loader, device, cfg, epoch_dir):
    calib = collect(base, refiner, calib_loader, device, cfg)
    audit = collect(base, refiner, audit_loader, device, cfg)
    gain, gain_rows = select_gains(calib, audit, cfg["refiner"]["gain_candidates"])
    refiner.set_deployment_gain(gain.to(device))
    thresholds = fit_locked_thresholds(
        calib, audit, gain, cfg["calibration"]["reason_probability_offset"]
    )
    test = collect(base, refiner, test_loader, device, cfg, gain=gain.to(device))
    base_metrics = aie_branch_metrics(
        test["base_action"], test["reason"], test["action_target"], test["reason_target"]
    )
    deploy = aie_branch_metrics(
        apply_posthoc_threshold(test["final_action"], thresholds[:4]),
        apply_posthoc_threshold(test["reason"], thresholds[4:]),
        test["action_target"],
        test["reason_target"],
    )
    raw = aie_branch_metrics(
        test["final_action"], test["reason"], test["action_target"], test["reason_target"]
    )
    epoch_dir.mkdir(parents=True, exist_ok=True)
    torch.save(test, epoch_dir / "test_outputs.pt")
    write_json(epoch_dir / "gain_selection.json", {"deployment_gain": gain.tolist(), "candidates": gain_rows})
    write_json(epoch_dir / "locked_thresholds.json", {"threshold_prob": thresholds.tolist()})
    write_json(epoch_dir / "metrics.json", {"base_raw": base_metrics, "refined_raw": raw, "deploy": deploy})
    return {"base_raw": base_metrics, "refined_raw": raw, "deploy": deploy}, gain, thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["data"]["split_seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(int(cfg["runtime"]["cpu_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    base = load_base(cfg, args.source_checkpoint, device)
    refiner = build_refiner(base, cfg).to(device)
    optimizer = torch.optim.AdamW(
        refiner.parameters(), lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    train = make_dataset(cfg, "train"); test = make_dataset(cfg, "test")
    names = [sample.file_name for sample in train.samples]
    split = stable_split_ids(
        names, seed, float(cfg["data"]["train_calib_fraction"]), int(cfg["data"]["train_audit_count"])
    )
    held_out = set(split["train_calib"]) | set(split["train_audit"])
    train_main = [name for name in names if name not in held_out]
    if held_out & set(train_main):
        raise RuntimeError("train_main overlaps train calibration/audit splits")
    index = {sample.file_name: i for i, sample in enumerate(train.samples)}
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    workers = args.num_workers if args.num_workers is not None else int(cfg["data"]["num_workers"])
    accumulation = args.gradient_accumulation_steps or int(cfg["training"]["gradient_accumulation_steps"])
    loaders = {
        "train": make_loader(Subset(train, [index[n] for n in train_main]), batch_size, True, workers, cfg),
        "calib": make_loader(Subset(train, [index[n] for n in split["train_calib"]]), batch_size, False, workers, cfg),
        "audit": make_loader(Subset(train, [index[n] for n in split["train_audit"]]), batch_size, False, workers, cfg),
        "test": make_loader(test, batch_size, False, workers, cfg),
    }
    write_split_manifest(out / "split_manifest.json", names, seed, float(cfg["data"]["train_calib_fraction"]), int(cfg["data"]["train_audit_count"]))
    write_json(out / "run_manifest.json", {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command_line": [sys.executable, *sys.argv],
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "direct_image": True, "feature_cache_enabled": False, "token_compression": "none",
        "base_frozen": True, "reason_identity": True,
        "batch_size": batch_size, "gradient_accumulation_steps": accumulation, "num_workers": workers,
    })
    epochs = args.epochs or int(cfg["training"]["epochs"])
    update = 0
    for epoch in range(epochs):
        refiner.train(); optimizer.zero_grad(set_to_none=True); window = []
        for micro, batch in enumerate(loaders["train"]):
            images = batch["image"].to(device, non_blocking=True)
            target = batch["action"].to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                base_output = base_forward(base, images, cfg)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                refined = run_refiner(
                    refiner,
                    base_output,
                    action_scale=float(cfg["evidence"]["action_scale"]),
                    gain=torch.ones(4, device=device),
                )
            window.append((refined, target))
            if (micro + 1) % accumulation and micro + 1 != len(loaders["train"]):
                continue
            logits = torch.cat([row[0]["action_logits_final"] for row in window])
            base_logits = torch.cat([row[0]["action_logits_base"] for row in window])
            delta = torch.cat([row[0]["action_delta"] for row in window])
            targets = torch.cat([row[1] for row in window])
            losses = {
                "action_asl": asymmetric_loss_with_logits(
                    logits, targets, gamma_pos=float(cfg["loss"]["gamma_pos"]),
                    gamma_neg=float(cfg["loss"]["gamma_neg"]), clip=float(cfg["loss"]["clip"]),
                ),
                "action_pairwise_ap": action_pairwise_ap_loss(
                    logits, targets, temperature=float(cfg["loss"]["pair_temperature"])
                ),
                "action_smooth_ap": action_smooth_ap_loss(
                    logits, targets, temperature=float(cfg["loss"]["smooth_ap_temperature"])
                ),
                "base_margin_trust": base_margin_trust_loss(
                    logits, base_logits, targets, tolerance=float(cfg["loss"]["trust_tolerance"])
                ),
                "residual_energy": residual_energy_loss(delta),
            }
            total = sum(float(cfg["loss_weights"][name]) * value for name, value in losses.items())
            total.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(refiner.parameters(), float(cfg["training"]["grad_clip"])))
            optimizer.step(); optimizer.zero_grad(set_to_none=True); update += 1; window = []
            if update == 1 or update % int(cfg["training"]["print_every_updates"]) == 0:
                row = {"event": "strong_refine_batch", "epoch": epoch, "update": update,
                       "loss_total": float(total.detach()), "grad_norm": grad_norm,
                       "delta_rms": float(delta.detach().float().square().mean().sqrt()),
                       **{f"loss_{name}": float(value.detach()) for name, value in losses.items()}}
                print(json.dumps(row), flush=True); append_jsonl(out / "loss_components.jsonl", row)
        torch.save(
            {"epoch": epoch, "update": update, "refiner": refiner.state_dict(),
             "optimizer": optimizer.state_dict(), "stage": "pre_eval"},
            out / f"checkpoint_pre_eval_epoch_{epoch:03d}.pth",
        )
        metrics, gain, thresholds = evaluate(
            base, refiner, loaders["calib"], loaders["audit"], loaders["test"], device, cfg, out / f"epoch_{epoch:03d}"
        )
        checkpoint = {"epoch": epoch, "update": update, "refiner": refiner.state_dict(),
                      "optimizer": optimizer.state_dict(), "deployment_gain": gain, "thresholds": thresholds,
                      "metrics": metrics}
        torch.save(checkpoint, out / "checkpoint_latest.pth")
        torch.save(checkpoint, out / f"checkpoint_epoch_{epoch:03d}.pth")
        append_jsonl(out / "metrics_summary.jsonl", {"epoch": epoch, **metrics})
        print(json.dumps({"event": "strong_refine_epoch", "epoch": epoch,
                          "Act_mF1": metrics["deploy"]["Act_mF1"], "Act_oF1": metrics["deploy"]["Act_oF1"],
                          "Exp_mF1": metrics["deploy"]["Exp_mF1"], "Exp_oF1": metrics["deploy"]["Exp_oF1"],
                          "Act_mAP": metrics["refined_raw"]["Act_mAP"], "Exp_mAP": metrics["refined_raw"]["Exp_mAP"],
                          "gain": gain.tolist()}), flush=True)


if __name__ == "__main__":
    main()
