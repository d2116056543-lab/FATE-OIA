from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from fate_oia.engine.train_acpr_tfc_oia import (
    build_flip_cases,
    build_model,
    make_loader,
    oracle_threshold_metrics,
    per_label_ranking_metrics,
)
from fate_oia.metrics import multilabel_metrics_from_logits


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.no_grad()
def collect_logits(model, loader, device: torch.device, epoch: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, list[str]]:
    buckets: dict[str, list[torch.Tensor]] = {
        "action_visual_only": [],
        "action_tfc_delta_off": [],
        "action_tfc_delta_on": [],
        "action_threshold_delta_off": [],
        "action_final_deploy": [],
        "reason_visual_only": [],
        "reason_tfc_delta_off": [],
        "reason_tfc_delta_on": [],
        "reason_threshold_delta_off": [],
        "reason_final_deploy": [],
    }
    action_labels: list[torch.Tensor] = []
    reason_labels: list[torch.Tensor] = []
    file_names: list[str] = []
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        out = model(images, None, None, epoch=epoch, split="test", run_deletion=False)
        buckets["action_visual_only"].append(out["action_visual_logits"].cpu())
        buckets["action_tfc_delta_off"].append((out["action_logits_base"] - out["action_tfc_delta"]).cpu())
        buckets["action_tfc_delta_on"].append(out["action_logits_base"].cpu())
        buckets["action_threshold_delta_off"].append(out["action_logits_base"].cpu())
        buckets["action_final_deploy"].append(out["action_logits_deploy"].cpu())
        buckets["reason_visual_only"].append(out["reason_visual_logits"].cpu())
        buckets["reason_tfc_delta_off"].append((out["reason_logits_base"] - out["reason_tfc_delta"]).cpu())
        buckets["reason_tfc_delta_on"].append(out["reason_logits_base"].cpu())
        buckets["reason_threshold_delta_off"].append(out["reason_logits_base"].cpu())
        buckets["reason_final_deploy"].append(out["reason_logits_deploy"].cpu())
        action_labels.append(batch["action"].cpu())
        reason_labels.append(batch["reason"].cpu())
        file_names.extend(batch["file_name"])
    stacked = {key: torch.cat(value, dim=0) for key, value in buckets.items()}
    return stacked, torch.cat(action_labels, dim=0), torch.cat(reason_labels, dim=0), file_names


def flip_counts(reference_logits: torch.Tensor, candidate_logits: torch.Tensor, labels: torch.Tensor) -> dict[str, list[int]]:
    ref_pred = torch.sigmoid(reference_logits) >= 0.5
    cand_pred = torch.sigmoid(candidate_logits) >= 0.5
    labels_bool = labels > 0.5
    return {
        "FP_to_TP": ((~ref_pred) & cand_pred & labels_bool).sum(0).to(torch.int64).tolist(),
        "TP_to_FN": (ref_pred & (~cand_pred) & labels_bool).sum(0).to(torch.int64).tolist(),
        "TN_to_FP": ((~ref_pred) & cand_pred & (~labels_bool)).sum(0).to(torch.int64).tolist(),
        "FN_to_TN": (ref_pred & (~cand_pred) & (~labels_bool)).sum(0).to(torch.int64).tolist(),
    }


def delta_by_label(before_logits: torch.Tensor, after_logits: torch.Tensor, prefix: str) -> list[dict[str, Any]]:
    delta = after_logits - before_logits
    rows: list[dict[str, Any]] = []
    for label_id in range(delta.shape[1]):
        values = delta[:, label_id]
        rows.append({
            "label_id": label_id,
            f"{prefix}_mean": float(values.mean().cpu()),
            f"{prefix}_abs_mean": float(values.abs().mean().cpu()),
            f"{prefix}_max": float(values.max().cpu()),
            f"{prefix}_min": float(values.min().cpu()),
        })
    return rows


def summarize(
    logits: dict[str, torch.Tensor],
    action_labels: torch.Tensor,
    reason_labels: torch.Tensor,
    file_names: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in [
        "action_visual_only",
        "action_tfc_delta_off",
        "action_tfc_delta_on",
        "action_threshold_delta_off",
        "action_final_deploy",
    ]:
        summary[key] = multilabel_metrics_from_logits(logits[key], action_labels, prefix="Act_")
    for key in [
        "reason_visual_only",
        "reason_tfc_delta_off",
        "reason_tfc_delta_on",
        "reason_threshold_delta_off",
        "reason_final_deploy",
    ]:
        summary[key] = multilabel_metrics_from_logits(logits[key], reason_labels, prefix="Exp_")
    summary["action_oracle_diagnostic"] = oracle_threshold_metrics(logits["action_final_deploy"], action_labels, prefix="Act_")
    summary["reason_oracle_diagnostic"] = oracle_threshold_metrics(logits["reason_final_deploy"], reason_labels, prefix="Exp_")
    summary["per_action_AP_AUC_F1"] = per_label_ranking_metrics(
        logits["action_final_deploy"],
        action_labels,
        summary["action_final_deploy"].get("Act_per_label_f1", []),
    )
    summary["wrong_flip_counts_vs_visual"] = flip_counts(logits["action_visual_only"], logits["action_final_deploy"], action_labels)
    summary["wrong_flip_counts_vs_tfc_delta_off"] = flip_counts(logits["action_tfc_delta_off"], logits["action_final_deploy"], action_labels)
    summary["deploy_vs_base_delta"] = {
        "action": delta_by_label(logits["action_tfc_delta_on"], logits["action_final_deploy"], "deploy_minus_base"),
        "reason": delta_by_label(logits["reason_tfc_delta_on"], logits["reason_final_deploy"], "deploy_minus_base"),
    }
    summary["failure_flip_cases"] = build_flip_cases(
        -1,
        file_names,
        logits["action_visual_only"],
        logits["action_final_deploy"],
        action_labels,
    )
    summary["delta_stats"] = {
        "action_tfc_delta_abs_mean": float((logits["action_tfc_delta_on"] - logits["action_tfc_delta_off"]).abs().mean()),
        "action_threshold_delta_abs_mean": float((logits["action_tfc_delta_on"] - logits["action_final_deploy"]).abs().mean()),
        "reason_tfc_delta_abs_mean": float((logits["reason_tfc_delta_on"] - logits["reason_tfc_delta_off"]).abs().mean()),
        "reason_threshold_delta_abs_mean": float((logits["reason_tfc_delta_on"] - logits["reason_final_deploy"]).abs().mean()),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fate_oia_train_360x640_acpr_tfc_v1.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=".background_runs/tfc_branch_ablation.json")
    parser.add_argument("--epoch", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    loader = make_loader(cfg, "test", args.batch_size, args.max_test_samples, False, args.num_workers)
    logits, action_labels, reason_labels, file_names = collect_logits(model, loader, device, args.epoch)
    summary = summarize(logits, action_labels, reason_labels, file_names)
    summary["checkpoint"] = str(args.checkpoint)
    summary["checkpoint_epoch"] = int(checkpoint.get("epoch", args.epoch)) if isinstance(checkpoint, dict) else args.epoch
    summary["num_samples"] = len(file_names)
    write_json(Path(args.output), summary)


if __name__ == "__main__":
    main()
