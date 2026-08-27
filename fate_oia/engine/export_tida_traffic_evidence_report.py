from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from fate_oia.utils.tida_traffic_evidence_report import (
    build_traffic_evidence_summary,
    render_traffic_evidence_html,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_first_jsonl(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"no JSON rows found in {path}")


def extract_metric_views(
    metrics: dict[str, Any],
    corrector: dict[str, Any],
    final_action_metrics: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    deploy = metrics.get("online", {}).get("deploy", {})
    image = deploy.get("image")
    video = deploy.get("video_stable")
    if not isinstance(image, dict) or not isinstance(video, dict):
        raise ValueError("locked deploy image/video_stable metrics are required")
    final = {
        key: copy.deepcopy(value)
        for key, value in video.items()
        if not key.startswith("Act_")
    }
    corrected_action = corrector.get("test", {}).get("full_Act_mF1")
    if corrected_action is None:
        raise ValueError("corrector test.full_Act_mF1 is required")
    final["Act_mF1"] = float(corrected_action)
    per_action = corrector.get("test", {}).get("full_per_action_f1")
    if per_action is not None:
        final["Act_per_label_f1"] = per_action
    final["Act_ranking_metric_status"] = "not_defined_for_discrete_router"
    if final_action_metrics is not None:
        final.update(final_action_metrics)
    return copy.deepcopy(image), copy.deepcopy(video), final


def compute_action_metrics(
    prediction: torch.Tensor,
    probability: torch.Tensor | None,
    target: torch.Tensor,
) -> dict[str, Any]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share [N,A]")
    if probability is not None and probability.shape != target.shape:
        raise ValueError("probability must share [N,A] when provided")
    prediction = prediction.bool()
    target = target.bool()
    tp = (prediction & target).sum(0).float()
    fp = (prediction & ~target).sum(0).float()
    fn = (~prediction & target).sum(0).float()
    per_f1 = 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)
    total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
    overall_f1 = 2.0 * total_tp / (
        2.0 * total_tp + total_fp + total_fn
    ).clamp_min(1.0)
    result = {
        "Act_mF1": float(per_f1.mean()),
        "Act_oF1": float(overall_f1),
        "Act_exact_match": float((prediction == target).all(1).float().mean()),
        "Act_per_label_f1": per_f1.tolist(),
    }
    if probability is None:
        result["Act_ranking_metric_status"] = "not_defined_for_discrete_router"
        return result
    average_precision = []
    for label in range(target.shape[1]):
        order = probability[:, label].argsort(descending=True)
        truth = target[order, label].float()
        positives = truth.sum()
        if positives <= 0:
            continue
        precision = truth.cumsum(0) / torch.arange(
            1, truth.numel() + 1, dtype=truth.dtype, device=truth.device
        )
        average_precision.append((precision * truth).sum() / positives)
    mean_ap = torch.stack(average_precision).mean() if average_precision else tp.new_nan(())
    result["Act_mAP"] = float(mean_ap)
    result["Act_ranking_metric_status"] = "continuous_score_available"
    return result


def _copy_case_images(case_dir: Path | None, output_dir: Path) -> list[str]:
    if case_dir is None or not case_dir.exists():
        return []
    destination = output_dir / "cases"
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in sorted(case_dir.glob("case_*.png")):
        target = destination / source.name
        shutil.copy2(source, target)
        copied.append(f"cases/{source.name}")
    return copied


def export_report(
    *,
    metrics_path: Path,
    effectiveness_path: Path,
    corrector_path: Path,
    output_dir: Path,
    case_dir: Path | None = None,
    corrector_tensors_path: Path | None = None,
) -> dict[str, Any]:
    metrics = (
        _read_first_jsonl(metrics_path)
        if metrics_path.suffix == ".jsonl"
        else _read_json(metrics_path)
    )
    effectiveness = _read_json(effectiveness_path)
    corrector = _read_json(corrector_path)
    final_action_metrics = None
    if corrector_tensors_path is not None:
        tensors = torch.load(corrector_tensors_path, map_location="cpu")
        final_action_metrics = compute_action_metrics(
            tensors["full_prediction"],
            None,
            tensors["action_target"],
        )
    image, video, final = extract_metric_views(
        metrics, corrector, final_action_metrics=final_action_metrics
    )
    summary = build_traffic_evidence_summary(
        image_metrics=image,
        video_metrics=video,
        final_metrics=final,
        effectiveness=effectiveness,
        corrector=corrector,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    case_images = _copy_case_images(case_dir, output_dir)
    with (output_dir / "traffic_evidence_scorecard.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    (output_dir / "report.html").write_text(
        render_traffic_evidence_html(summary, case_images=case_images),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--effectiveness", type=Path, required=True)
    parser.add_argument("--corrector", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--case_dir", type=Path)
    parser.add_argument("--corrector_tensors", type=Path)
    args = parser.parse_args()
    summary = export_report(
        metrics_path=args.metrics,
        effectiveness_path=args.effectiveness,
        corrector_path=args.corrector,
        output_dir=args.output_dir,
        case_dir=args.case_dir,
        corrector_tensors_path=args.corrector_tensors,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "claim": summary["evidence_verdict"]["claim"],
        "task_gain": summary["task_gain"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
