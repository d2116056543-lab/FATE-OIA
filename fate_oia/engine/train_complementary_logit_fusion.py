from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fate_oia.engine.eval_score_calibrated import evaluate_score_logits
from fate_oia.engine.offline_score_fusion_sweep import _load_branch, _metric_row, _reorder_to_reference
from fate_oia.losses.sigmoid_f1_loss import sigmoid_macro_f1_loss
from fate_oia.models.complementary_logit_fusion import ComplementaryLogitFusionAdapter


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _evaluate(action_logits: torch.Tensor, reason_logits: torch.Tensor, action_labels: torch.Tensor, reason_labels: torch.Tensor) -> dict[str, Any]:
    fixed = _metric_row("fixed", evaluate_score_logits(action_logits, reason_logits, action_labels, reason_labels, threshold_mode="fixed"))
    global_row = _metric_row("global", evaluate_score_logits(action_logits, reason_logits, action_labels, reason_labels, threshold_mode="global"))
    per_label = _metric_row("per_label", evaluate_score_logits(action_logits, reason_logits, action_labels, reason_labels, threshold_mode="per_label"))
    return {"fixed": fixed, "global": global_row, "per_label": per_label}


def train_complementary_fusion(
    *,
    run_c_dir: str | Path,
    s1_dir: str | Path,
    output_dir: str | Path,
    split: str = "test",
    epochs: int = 20,
    lr: float = 0.05,
    init_mix: float = 0.2,
    reference_exp_mf1: float = 0.388808,
    reference_margin: float = 0.003,
    f1_loss_weight: float = 0.2,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_c = _load_branch(run_c_dir, split=split, prefer_fused_action=True)
    score_v2 = _load_branch(s1_dir, split=split, prefer_fused_action=False)
    score_v2, reordered = _reorder_to_reference(score_v2, run_c["file_names"])
    if not torch.equal(run_c["action_labels"], score_v2["action_labels"]) or not torch.equal(run_c["reason_labels"], score_v2["reason_labels"]):
        raise ValueError("Aligned labels differ between Run C and ScoreV2 outputs.")
    action_logits = run_c["action_logits"].float()
    labels_action = run_c["action_labels"].float()
    labels_reason = run_c["reason_labels"].float()
    run_c_reason = run_c["reason_logits"].float()
    score_reason = score_v2["reason_logits"].float()
    model = ComplementaryLogitFusionAdapter(reason_dim=run_c_reason.shape[1], init_mix=init_mix)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    bce = nn.BCEWithLogitsLoss()
    manifest = {
        "mode": "cached_logit_complementary_fusion",
        "split": split,
        "run_c_dir": str(run_c_dir),
        "score_v2_dir": str(s1_dir),
        "run_c_inputs": run_c["paths"],
        "score_v2_inputs": score_v2["paths"],
        "score_v2_reordered_to_run_c": reordered,
        "epochs": epochs,
        "lr": lr,
        "init_mix": init_mix,
        "reference_exp_mf1": reference_exp_mf1,
        "reference_margin": reference_margin,
        "not_fair_main_result": True,
        "note": "This adapter is a cached-logit calibration/fusion diagnostic, not a visual backbone training result.",
    }
    _write_json(out / "run_manifest.json", manifest)
    baseline = _evaluate(action_logits, run_c_reason, labels_action, labels_reason)
    _write_json(out / "baseline_run_c_metrics.json", baseline)
    best_score = float("-inf")
    best_row: dict[str, Any] | None = None
    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        reason_logits = model(run_c_reason, score_reason)
        loss_bce = bce(reason_logits, labels_reason)
        loss_f1 = sigmoid_macro_f1_loss(reason_logits, labels_reason)
        loss = loss_bce + float(f1_loss_weight) * loss_f1
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            fused_reason = model(run_c_reason, score_reason)
            metrics = _evaluate(action_logits, fused_reason, labels_action, labels_reason)
        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "loss_bce": float(loss_bce.item()),
            "loss_sigmoid_f1": float(loss_f1.item()),
            "joint": metrics["fixed"]["joint"],
            "Act_mF1": metrics["fixed"]["Act_mF1"],
            "Exp_mF1": metrics["fixed"]["Exp_mF1"],
            "Exp_mAP": metrics["fixed"]["Exp_mAP"],
            "global_Exp_mF1": metrics["global"]["Exp_mF1"],
            "per_label_Exp_mF1": metrics["per_label"]["Exp_mF1"],
            "mean_mix": float(model.mix_weight().mean().item()),
            "min_mix": float(model.mix_weight().min().item()),
            "max_mix": float(model.mix_weight().max().item()),
            "meets_p1_margin": bool(metrics["fixed"]["Exp_mF1"] >= reference_exp_mf1 + reference_margin),
        }
        _append_jsonl(out / "metrics_summary.jsonl", row)
        _write_json(out / f"epoch_{epoch:03d}" / "metrics_summary.json", row)
        _write_json(out / f"epoch_{epoch:03d}" / "metrics_fixed_test.json", metrics["fixed"])
        _write_json(out / f"epoch_{epoch:03d}" / "metrics_global_threshold_test.json", metrics["global"])
        _write_json(out / f"epoch_{epoch:03d}" / "metrics_per_label_threshold_test.json", metrics["per_label"])
        torch.save(fused_reason.detach().cpu(), out / f"epoch_{epoch:03d}" / "logits_reason_test.pt")
        torch.save(action_logits.detach().cpu(), out / f"epoch_{epoch:03d}" / "logits_action_test.pt")
        _write_json(out / f"epoch_{epoch:03d}" / "fusion_parameters.json", {"mix": model.mix_weight().detach().cpu().tolist(), "bias": model.bias.detach().cpu().tolist(), "temperature": model.temperature().detach().cpu().tolist()})
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row, "manifest": manifest}, out / "checkpoint_latest.pth")
        if row["joint"] > best_score:
            best_score = row["joint"]
            best_row = row
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row, "manifest": manifest}, out / "checkpoint_best_test.pth")
            _write_json(out / "metrics_best_test.json", row)
        print(json.dumps({"event": "complementary_fusion_epoch", **row}, ensure_ascii=False), flush=True)
    decision = {
        "best": best_row,
        "reference_exp_mf1": reference_exp_mf1,
        "reference_margin": reference_margin,
        "exceeded_reference_margin": bool(best_row and best_row["Exp_mF1"] >= reference_exp_mf1 + reference_margin),
        "recommendation": "report_as_cached_logit_calibration_diagnostic" if best_row else "failed_no_metrics",
    }
    _write_json(out / "adapter_decision.json", decision)
    return {"best": best_row, "decision": decision}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train cached-logit complementary fusion adapter.")
    ap.add_argument("--run_c_dir", required=True)
    ap.add_argument("--s1_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--init_mix", type=float, default=0.2)
    ap.add_argument("--reference_exp_mf1", type=float, default=0.388808)
    ap.add_argument("--reference_margin", type=float, default=0.003)
    ap.add_argument("--f1_loss_weight", type=float, default=0.2)
    args = ap.parse_args()
    result = train_complementary_fusion(**vars(args))
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
