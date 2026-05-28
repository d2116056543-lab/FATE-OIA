from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import torch


def _json_default(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return str(obj)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    labels = labels.float()
    positives = int(labels.sum().item())
    if positives <= 0:
        return None
    order = torch.argsort(scores.float(), descending=True)
    sorted_labels = labels[order]
    tp = torch.cumsum(sorted_labels, dim=0)
    ranks = torch.arange(1, sorted_labels.numel() + 1, dtype=torch.float32)
    precision = tp / ranks
    return float((precision * sorted_labels).sum().item() / max(1, positives))


def per_label_reason_audit(reason_logits: torch.Tensor, labels_reason: torch.Tensor, *, threshold: float = 0.5) -> dict[str, Any]:
    probs = torch.sigmoid(reason_logits.float())
    labels = labels_reason.float()
    pred = (probs >= threshold).float()
    rows: list[dict[str, Any]] = []
    for idx in range(labels.shape[1]):
        y = labels[:, idx]
        p = pred[:, idx]
        tp = int(((p == 1) & (y == 1)).sum().item())
        fp = int(((p == 1) & (y == 0)).sum().item())
        fn = int(((p == 0) & (y == 1)).sum().item())
        support = int(y.sum().item())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        rows.append({
            "label_index": idx,
            "support": support,
            "pred_positive": int(p.sum().item()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap": _average_precision(probs[:, idx], y),
        })
    macro_f1 = sum(row["f1"] for row in rows) / max(1, len(rows))
    valid_ap = [row["ap"] for row in rows if row["ap"] is not None]
    return {"threshold": threshold, "macro_f1": macro_f1, "macro_ap": sum(valid_ap) / max(1, len(valid_ap)), "per_label": rows}


def tail_group_metrics(audit: dict[str, Any], tail_reason_indices: Iterable[int]) -> dict[str, Any]:
    indices = [int(x) for x in tail_reason_indices]
    rows = audit["per_label"]
    selected = [rows[idx] for idx in indices]
    valid_ap = [row["ap"] for row in selected if row.get("ap") is not None]
    return {
        "tail_reason_indices": indices,
        "tail_positive_support": int(sum(int(row["support"]) for row in selected)),
        "tail_pred_positive": int(sum(int(row["pred_positive"]) for row in selected)),
        "tail_macro_f1": sum(float(row["f1"]) for row in selected) / max(1, len(selected)),
        "tail_macro_ap": sum(float(x) for x in valid_ap) / max(1, len(valid_ap)),
        "tail_labels": selected,
    }


def _load_or_use(value: torch.Tensor | None, path: Path) -> torch.Tensor:
    if value is not None:
        return value.detach().cpu()
    return torch.load(path, map_location="cpu")


def write_score_v2_epoch_diagnostics(
    epoch_dir: str | Path,
    *,
    run_dir: str | Path | None = None,
    split: str = "test",
    action_logits: torch.Tensor | None = None,
    reason_logits: torch.Tensor | None = None,
    labels_action: torch.Tensor | None = None,
    labels_reason: torch.Tensor | None = None,
    file_names: list[str] | None = None,
    tail_reason_indices: Iterable[int] = (12, 9, 5, 14, 6, 11, 10, 13),
    n_last_blocks: int | None = None,
) -> dict[str, Any]:
    epoch_path = Path(epoch_dir)
    root = Path(run_dir) if run_dir is not None else epoch_path.parent
    action = _load_or_use(action_logits, epoch_path / f"logits_action_{split}.pt").float()
    reason = _load_or_use(reason_logits, epoch_path / f"logits_reason_{split}.pt").float()
    action_y = _load_or_use(labels_action, epoch_path / f"labels_action_{split}.pt").float()
    reason_y = _load_or_use(labels_reason, epoch_path / f"labels_reason_{split}.pt").float()
    if file_names is None:
        file_names_path = epoch_path / f"file_names_{split}.json"
        file_names = json.loads(file_names_path.read_text(encoding="utf-8")) if file_names_path.exists() else [str(i) for i in range(action.shape[0])]

    if split == "test":
        torch.save(action, epoch_path / "logits_action.pt")
        torch.save(reason, epoch_path / "logits_reason.pt")
        torch.save(action, epoch_path / "logits_action_reason.pt")
        torch.save(action_y, epoch_path / "labels_action.pt")
        torch.save(reason_y, epoch_path / "labels_reason.pt")
        _write_json(epoch_path / "file_names.json", [str(x) for x in file_names])

    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        shutil.copyfile(manifest_path, epoch_path / "run_manifest.json")

    audit = per_label_reason_audit(reason, reason_y)
    tail = tail_group_metrics(audit, tail_reason_indices)
    label_query = {
        "source": "cached_logits_backfill",
        "label_query_tensor_available": False,
        "action_reason_available": False,
        "logits_action_reason_is_alias_of_action_logits": True,
        "action_logit_mean": float(action.mean().item()) if action.numel() else 0.0,
        "action_logit_std": float(action.std(unbiased=False).item()) if action.numel() else 0.0,
        "reason_logit_mean": float(reason.mean().item()) if reason.numel() else 0.0,
        "reason_logit_std": float(reason.std(unbiased=False).item()) if reason.numel() else 0.0,
    }
    feature_stats = {
        "source": "cached_logits_backfill",
        "feature_tokens_available": False,
        "n_last_blocks": n_last_blocks,
        "sample_count": int(action.shape[0]),
        "action_dim": int(action.shape[1]),
        "reason_dim": int(reason.shape[1]),
    }
    _write_json(epoch_path / "per_label_reason_audit.json", audit)
    _write_json(epoch_path / "tail_group_metrics.json", tail)
    _write_json(epoch_path / "label_query_stats.json", label_query)
    _write_json(epoch_path / "multilayer_feature_stats.json", feature_stats)
    return {"per_label_reason_audit": audit, "tail_group_metrics": tail, "label_query_stats": label_query, "multilayer_feature_stats": feature_stats}
