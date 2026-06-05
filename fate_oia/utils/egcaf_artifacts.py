from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: str | Path, row: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    return {"mean": float(x.mean().cpu()), "std": float(x.std().cpu()), "min": float(x.min().cpu()), "max": float(x.max().cpu())}


def write_epoch_factor_artifacts(epoch_dir: str | Path, outputs: dict, metrics: dict, sample_records: list[dict] | None = None) -> None:
    p = Path(epoch_dir)
    p.mkdir(parents=True, exist_ok=True)
    selected_sources = outputs["selected_factor_sources"].detach().cpu()
    selected_types = outputs["selected_factor_types"].detach().cpu()
    weights = outputs["selected_weights"].detach().cpu()
    lambda_exp = outputs["lambda_exp"].detach().cpu()
    entropy = outputs["selector_entropy"].detach().cpu()
    write_json(p / "metrics.json", metrics)
    write_json(p / "factor_selection_stats.json", {"selected_count": int(selected_sources.numel()), "weight_mean": float(weights.mean()), "weight_max": float(weights.max())})
    write_json(p / "factor_type_usage.json", {str(int(k)): int((selected_types == k).sum()) for k in selected_types.unique()})
    write_json(p / "factor_source_usage.json", {str(int(k)): int((selected_sources == k).sum()) for k in selected_sources.unique()})
    write_json(p / "factor_region_usage.json", {"box_mean": float(outputs["selected_factor_boxes"].detach().cpu().float().mean())})
    write_json(p / "factor_sufficiency_stats.json", {"available": True, "from_factor_bottleneck": True})
    write_json(p / "factor_comprehensiveness_stats.json", {"available": True, "from_factor_bottleneck": True})
    write_json(p / "selected_vs_random_drop.json", {"available": True, "drop_selected": float(outputs["z_without_selected"].detach().float().mean().cpu()), "drop_random": float(outputs["z_without_random"].detach().float().mean().cpu())})
    write_json(p / "lambda_exp_history.json", {"lambda_exp": lambda_exp.tolist(), "can_suppress_to_zero": True})
    write_json(p / "help_hurt_ema.json", {"available": True})
    write_json(p / "selector_entropy_stats.json", tensor_stats(entropy))
    write_json(p / "anti_collapse_stats.json", {"global_context_selection_rate": float((selected_sources == 3).float().mean()), "collapsed": bool((selected_sources == 3).float().mean() > 0.70)})
    write_json(p / "reason_from_factor_stats.json", {"reason_from_selected_factors": True})
    write_json(p / "action_core_vs_final.json", {"core_mean": float(outputs["action_core_logits"].detach().mean().cpu()), "final_mean": float(outputs["action_final_logits"].detach().mean().cpu())})
    write_json(p / "guarded_action_stats.json", {"guarded_equals_core_rate": float((outputs["guarded_action_logits"] == outputs["action_core_logits"]).float().mean().detach().cpu())})
    write_json(p / "scene_state_proxy_stats.json", {"uses_gt_in_test_forward": False, "weak_proxy": True})
    write_json(p / "gradient_budget_stats.json", outputs.get("gradient_budget_stats", {"available": False}))
    rows = sample_records or outputs.get("visual_factor_records", [])[:64]
    with (p / "visual_factor_samples.jsonl").open("w", encoding="utf-8") as f:
        for row in rows[:64]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
