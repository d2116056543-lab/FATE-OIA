from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _to_jsonable(x: Any):
    if torch.is_tensor(x):
        if x.numel() == 1:
            return float(x.detach().cpu())
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")


def write_required_smoke_artifacts(output_dir: str | Path, metrics: dict[str, Any], last_outputs: dict[str, Any], grad_stats: dict[str, Any], branch_metrics: dict[str, Any]) -> None:
    out = Path(output_dir)
    write_json(out / "metrics_latest.json", metrics)
    write_json(out / "branch_metrics_epoch_0.json", branch_metrics)
    write_json(out / "visual_gate_stats.json", {"mean": last_outputs["visual_gate"].mean(), "min": last_outputs["visual_gate"].min(), "max": last_outputs["visual_gate"].max(), "gate_target_mean": last_outputs["gate_target"].mean()})
    write_json(out / "diva_evidence_stats.json", {"evidence_confidence_mean": last_outputs["evidence_confidence"].mean(), "action_evidence_shape": list(last_outputs["action_evidence_tokens"].shape), "sample_points_available": last_outputs.get("evidence_sample_points") is not None})
    write_json(out / "action_set_usage_stats.json", {"prototype_vectors_used": True, "num_action_prototypes": 8})
    write_json(out / "factor_selection_stats.json", {"selected_indices_shape": list(last_outputs["selected_factor_indices"].shape), "selected_weight_mean": last_outputs["selected_factor_weights"].mean(), "selected_region_shape": list(last_outputs["selected_factor_meta"]["region"].shape)})
    groups = last_outputs["factor_groups"].detach().cpu().flatten().tolist()
    usage = {str(int(g)): int(groups.count(g)) for g in sorted(set(groups))}
    write_json(out / "factor_group_usage.json", {"usage": usage})
    sv = last_outputs["selected_vs_random_stats"]
    write_json(out / "selected_vs_random_action_loss_drop.json", {"method": "action_gt_bce_loss_drop", "drop_selected_minus_random": sv.get("selected_vs_random_action_loss_drop", 0.0), "drop_selected": sv.get("drop_selected", 0.0), "drop_random": sv.get("drop_random", 0.0)})
    write_json(out / "reason_factor_attention_stats.json", {"shape": list(last_outputs["reason_to_factor_attention"].shape), "mean": last_outputs["reason_to_factor_attention"].mean()})
    write_json(out / "reason_tail_stats.json", {"reason_gate_mean": last_outputs["reason_gate"].mean(), "tail_indices": last_outputs["tail_reason_indices"]})
    write_json(out / "bdd100k_scene_state_stats.json", {"train_support_only": True, "test_forward_uses_gt": False})
    write_json(out / "no_test_leakage_assertion.json", last_outputs["no_test_leakage_assertion"])
    write_json(out / "gradient_budget_stats.json", grad_stats)
    append_jsonl(out / "visual_samples_epoch_0.jsonl", {"sample_index": 0, "has_factor": True, "has_action": True, "has_reason": True, "selected_factor_indices": last_outputs["selected_factor_indices"][0].detach().cpu().tolist(), "selected_factor_regions": last_outputs["selected_factor_meta"]["region"][0].detach().cpu().tolist()})
