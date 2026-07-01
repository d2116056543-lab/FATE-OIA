from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from fate_oia.engine.train_acpr_ntmcal_oia import build_model, make_loader
from fate_oia.utils.acpr_thresholds import acpr_metric_views, standard_joint


def load_config(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    scores = scores.detach().float().cpu()
    labels = labels.detach().float().cpu()
    aucs = []
    for j in range(scores.shape[1]):
        s = scores[:, j]
        y = labels[:, j]
        pos = int(y.sum().item())
        neg = int((1 - y).sum().item())
        if pos == 0 or neg == 0:
            aucs.append(float("nan"))
            continue
        order = torch.argsort(s)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float32)
        pos_rank_sum = ranks[y.bool()].sum()
        auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
        aucs.append(float(auc.item()))
    return torch.tensor(aucs)


def state_name(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.bool()
    target = target.bool()
    # TP=0, FN=1, TN=2, FP=3
    out = torch.empty_like(pred, dtype=torch.long)
    out[pred & target] = 0
    out[(~pred) & target] = 1
    out[(~pred) & (~target)] = 2
    out[pred & (~target)] = 3
    return out


def flip_counts(base_logits: torch.Tensor, variant_logits: torch.Tensor, labels: torch.Tensor) -> dict[str, int]:
    base_pred = base_logits >= 0
    var_pred = variant_logits >= 0
    base_state = state_name(base_pred, labels)
    var_state = state_name(var_pred, labels)
    names = ["TP", "FN", "TN", "FP"]
    result: dict[str, int] = {}
    for i, src in enumerate(names):
        for j, dst in enumerate(names):
            result[f"{src}_to_{dst}"] = int(((base_state == i) & (var_state == j)).sum().item())
    result["beneficial_wrong_to_right"] = result["FN_to_TP"] + result["FP_to_TN"]
    result["harmful_right_to_wrong"] = result["TP_to_FN"] + result["TN_to_FP"]
    # These user-requested labels are logically impossible under fixed ground truth; keep them explicit.
    result["FP_to_TP"] = result.get("FP_to_TP", 0)
    result["FN_to_TN"] = result.get("FN_to_TN", 0)
    return result


def summarize_variant(
    checkpoint_name: str,
    variant: str,
    action_logits: torch.Tensor,
    reason_logits: torch.Tensor,
    base_action_logits: torch.Tensor,
    labels_action: torch.Tensor,
    labels_reason: torch.Tensor,
) -> dict[str, Any]:
    views = acpr_metric_views(action_logits, reason_logits, labels_action, labels_reason)
    fixed = views["metrics_raw_fixed"]
    auc = binary_auc(action_logits, labels_action)
    delta = action_logits - base_action_logits
    flips = flip_counts(base_action_logits, action_logits, labels_action)
    return {
        "checkpoint": checkpoint_name,
        "variant": variant,
        "Act_mF1": float(fixed["Act_mF1"]),
        "Act_oF1": float(fixed["Act_oF1"]),
        "Act_mAP": float(fixed["Act_mAP"]),
        "Act_AUC_mean": float(torch.nanmean(auc).item()),
        "Act_per_label_auc": [None if torch.isnan(x) else float(x) for x in auc],
        "Act_per_label_f1": [float(x) for x in fixed.get("Act_per_label_f1", [])],
        "Act_per_label_ap": [float(x) for x in fixed.get("Act_per_label_ap", [])],
        "Exp_mF1": float(fixed["Exp_mF1"]),
        "Exp_oF1": float(fixed["Exp_oF1"]),
        "Exp_mAP": float(fixed["Exp_mAP"]),
        "standard_joint": standard_joint(fixed),
        "deploy_vs_base_action_delta_mean_by_label": [float(x) for x in delta.mean(0)],
        "deploy_vs_base_action_delta_abs_mean_by_label": [float(x) for x in delta.abs().mean(0)],
        "deploy_vs_base_action_delta_abs_mean": float(delta.abs().mean().item()),
        "flip_counts_vs_base_fixed": flips,
    }


def compose_from_trunk(model, trunk: dict[str, torch.Tensor], pred: dict[str, torch.Tensor], epoch: int, *, force_zero_reason_delta: bool = False) -> dict[str, torch.Tensor]:
    base_action = trunk["action_logits_direct"]
    base_reason = trunk["reason_logits_visual"]
    pu = model.pu_builder(None, pred["predicate_q"], pred["predicate_rho"], epoch)
    reason_res = model.reason_residual(
        base_reason,
        trunk["label_nodes"][:, model.action_dim :],
        pu["support_score"],
        pu["contra_score"],
        pu["reason_rho"],
        epoch=epoch,
    )
    reason_delta = torch.zeros_like(reason_res["reason_delta"]) if force_zero_reason_delta else reason_res["reason_delta"]
    reason_logits = base_reason + reason_delta
    action_pred = model.action_predicate_head(base_action, pred["predicate_q"], pred["predicate_rho"], pred["predicate_tokens"], epoch=epoch)
    action_logits = base_action + action_pred["action_predicate_delta"]
    cal = model.ntmcal_threshold(
        action_logits,
        reason_logits,
        pu["support_score"],
        pu["contra_score"],
        pu["reason_rho"],
        base_reason,
        pred["predicate_q"],
        pred["predicate_rho"],
        epoch=epoch,
    )
    return {
        **trunk,
        **pred,
        **reason_res,
        **action_pred,
        **cal,
        "action_logits_base": base_action,
        "reason_logits_base": base_reason,
        "action_logits_ntmcal": action_logits,
        "reason_logits_ntmcal": reason_logits,
        "pu_state": pu,
        "support_score": pu["support_score"],
        "contra_score": pu["contra_score"],
    }


def forward_components(model, images, epoch: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    field = model.dino(images)
    patch = field["patch_tokens_by_layer"]
    patch0, _ego_features, region_masks, _ego_stats = model.ego(patch[:, 0])
    patch = patch.clone()
    patch[:, 0] = patch0
    pred = model.predicate_measurement(patch, region_masks=region_masks)
    trunk_full = model.trunk(patch, predicate_tokens=pred["predicate_tokens"])
    trunk_no_pred = model.trunk(patch, predicate_tokens=None)
    return pred, trunk_full, trunk_no_pred


@torch.no_grad()
def run_checkpoint(args, checkpoint_path: Path, checkpoint_name: str, cfg: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    model = build_model(cfg, device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    load_result = model.load_state_dict(ckpt["model"], strict=False)
    epoch = int(ckpt.get("epoch", args.epoch_override if args.epoch_override is not None else 0))
    model.eval()
    loader = make_loader(cfg, "test", args.batch_size, args.max_test_samples, False, args.num_workers)
    buckets: dict[str, list[torch.Tensor]] = {}
    labels_action = []
    labels_reason = []

    def add(name: str, tensor: torch.Tensor) -> None:
        buckets.setdefault(name, []).append(tensor.detach().cpu())

    for batch in loader:
        images = batch["image"].to(device)
        pred, trunk_full, trunk_no_pred = forward_components(model, images, epoch)
        out = compose_from_trunk(model, trunk_full, pred, epoch)
        no_reason = compose_from_trunk(model, trunk_full, pred, epoch, force_zero_reason_delta=True)
        no_pred = compose_from_trunk(model, trunk_no_pred, pred, epoch)

        # V1: disable action predicate delta and recompute action threshold on base action.
        cal_no_action_pred = model.ntmcal_threshold(
            out["action_logits_base"],
            out["reason_logits_ntmcal"],
            out["support_score"],
            out["contra_score"],
            out["pu_state"]["reason_rho"],
            out["reason_logits_base"],
            out["predicate_q"],
            out["predicate_rho"],
            epoch=epoch,
        )
        # V5: action visual-only branch, threshold recomputed for that action branch.
        cal_visual_only = model.ntmcal_threshold(
            out["action_visual_logits"],
            out["reason_logits_ntmcal"],
            out["support_score"],
            out["contra_score"],
            out["pu_state"]["reason_rho"],
            out["reason_logits_base"],
            out["predicate_q"],
            out["predicate_rho"],
            epoch=epoch,
        )

        add("base_action", out["action_logits_base"])
        add("base_reason", out["reason_logits_base"])
        add("v0_action", out["action_logits_deploy"])
        add("v0_reason", out["reason_logits_deploy"])
        add("v1_action", cal_no_action_pred["action_logits_deploy"])
        add("v1_reason", out["reason_logits_deploy"])
        add("v2_action", out["action_logits_deploy"] + out["threshold_delta_action"])
        add("v2_reason", out["reason_logits_deploy"])
        add("v3_action", no_reason["action_logits_deploy"])
        add("v3_reason", no_reason["reason_logits_deploy"])
        add("v4_action", no_pred["action_logits_deploy"])
        add("v4_reason", no_pred["reason_logits_deploy"])
        add("v5_action", cal_visual_only["action_logits_deploy"])
        add("v5_reason", out["reason_logits_deploy"])
        add("v6_action", out["action_logits_base"])
        add("v6_reason", out["reason_logits_base"])
        add("v6b_action", cal_no_action_pred["action_logits_deploy"])
        add("v6b_reason", cal_no_action_pred["reason_logits_deploy"])
        labels_action.append(batch["action"])
        labels_reason.append(batch["reason"])

    tensors = {k: torch.cat(v) for k, v in buckets.items()}
    ya = torch.cat(labels_action)
    yr = torch.cat(labels_reason)
    variants = {
        "V0_full_deploy": ("v0_action", "v0_reason"),
        "V1_no_action_predicate_delta": ("v1_action", "v1_reason"),
        "V2_no_threshold_delta_action": ("v2_action", "v2_reason"),
        "V3_no_reason_delta": ("v3_action", "v3_reason"),
        "V4_no_predicate_tokens_in_trunk": ("v4_action", "v4_reason"),
        "V5_visual_only_action_branch": ("v5_action", "v5_reason"),
        "V6_base_logits_only_fixed": ("v6_action", "v6_reason"),
        "V6b_base_logits_current_ntmcal_threshold": ("v6b_action", "v6b_reason"),
    }
    rows = []
    for variant, (ak, rk) in variants.items():
        rows.append(summarize_variant(checkpoint_name, variant, tensors[ak], tensors[rk], tensors["base_action"], ya, yr))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--checkpoint_name", action="append", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_test_samples", type=int, default=None)
    ap.add_argument("--epoch_override", type=int, default=None)
    args = ap.parse_args()
    if len(args.checkpoint) != len(args.checkpoint_name):
        raise SystemExit("--checkpoint and --checkpoint_name counts must match")
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    load_notes: list[dict[str, Any]] = []
    for ckpt, name in zip(args.checkpoint, args.checkpoint_name):
        # Reload here once for load diagnostics; the actual evaluation reloads inside run_checkpoint.
        tmp_model = build_model(cfg, device)
        tmp_ckpt = torch.load(Path(ckpt), map_location=device)
        load_result = tmp_model.load_state_dict(tmp_ckpt["model"], strict=False)
        load_notes.append({
            "checkpoint": name,
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "epoch": int(tmp_ckpt.get("epoch", -1)),
        })
        del tmp_model
        all_rows.extend(run_checkpoint(args, Path(ckpt), name, cfg, device))
    json_path = out_dir / "action_regression_d1_ablation.json"
    csv_path = out_dir / "action_regression_d1_ablation.csv"
    json_path.write_text(json.dumps({"rows": all_rows, "load_notes": load_notes}, indent=2, ensure_ascii=False), encoding="utf-8")
    keys = [
        "checkpoint", "variant", "standard_joint", "Act_mF1", "Act_oF1", "Act_mAP", "Act_AUC_mean",
        "Exp_mF1", "Exp_oF1", "Exp_mAP", "deploy_vs_base_action_delta_abs_mean",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k) for k in keys})
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "rows": len(all_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
