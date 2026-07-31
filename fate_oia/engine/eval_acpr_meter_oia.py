from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.meter_dataset import METERDataset
from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.transforms_meter import meter_image_transform
from fate_oia.utils.meter_artifacts import load_checkpoint, write_json
from fate_oia.utils.meter_config import load_meter_config
from fate_oia.utils.meter_posthoc_calibration import (
    METERCalibrationResult,
    apply_meter_deploy,
)


CHEAP_SAME_FORWARD_MODES: dict[str, tuple[str, ...]] = {
    "factor_off": ("factor_off",),
    "state_uniform": ("state_uniform",),
    "reason_correction_off": ("reason_correction_off",),
}

# B0--B5 alter training or data-assignment semantics.  They must never be
# reported as a same-forward diagnostic just because they share a checkpoint.
INDEPENDENT_HECA_ABLATIONS: dict[str, dict[str, str]] = {
    "B0": {"name": "CalAlign foundation", "execution": "independent_run"},
    "B1": {"name": "typed measurement only", "execution": "independent_run"},
    "B2": {"name": "state-conditioned action credit", "execution": "independent_run"},
    "B3": {"name": "measurement-credit bridge", "execution": "independent_run"},
    "B4": {"name": "robust reason global", "execution": "independent_run"},
    "B5": {"name": "evidence-label consistency", "execution": "independent_run"},
}

# Kept as a compatibility alias for callers that used the old symbol.  The
# contents intentionally contain only cheap decode-time interventions.
SEQUENTIAL_MODES = CHEAP_SAME_FORWARD_MODES


def _cat(rows: list[Tensor]) -> Tensor:
    return torch.cat(rows) if rows else torch.empty(0)


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _entropy(probability: Tensor, dim: int = -1) -> Tensor:
    probability = probability.float().clamp_min(1e-8)
    return -(probability * probability.log()).sum(dim)


def _metric_pair(action: Tensor, reason: Tensor, labels_a: Tensor, labels_r: Tensor) -> dict[str, Any]:
    action_metrics = multilabel_metrics_from_logits(action, labels_a, prefix="Act_")
    reason_metrics = multilabel_metrics_from_logits(reason, labels_r, prefix="Exp_")
    return {
        **action_metrics,
        **reason_metrics,
        "joint": 0.5
        * (
            float(action_metrics.get("Act_mF1", 0.0))
            + float(reason_metrics.get("Exp_mF1", 0.0))
        ),
    }


def _new_collector() -> dict[str, Any]:
    return {
        "action_visual": [],
        "action_final": [],
        "reason_calalign": [],
        "reason_global": [],
        "reason_final": [],
        "mechanism": {},
        "eval_mode_time": 0.0,
    }


def heca_ablation_manifest() -> dict[str, Any]:
    """Describe cheap decode probes without mislabelling B0--B5 as them."""
    return {
        "cheap_same_forward": list(CHEAP_SAME_FORWARD_MODES),
        "clean_branches": [
            "action_visual",
            "action_final",
            "reason_calalign",
            "reason_global",
            "reason_final",
        ],
        "independent_runs": INDEPENDENT_HECA_ABLATIONS,
    }


def _append_output(
    collector: dict[str, Any],
    output: dict[str, Any],
    *,
    elapsed: float,
    collect_mechanism: bool,
) -> None:
    collector["eval_mode_time"] += elapsed
    collector["action_visual"].append(output["action_logits_visual"].detach().cpu())
    collector["action_final"].append(output["action_logits_final"].detach().cpu())
    collector["reason_calalign"].append(output["reason_logits_calalign"].detach().cpu())
    collector["reason_global"].append(output["reason_logits_global"].detach().cpu())
    collector["reason_final"].append(output["reason_logits_final"].detach().cpu())
    if not collect_mechanism:
        return
    keys = (
        "factor_anchor_map",
        "factor_null_mass",
        "factor_state_prob",
        "factor_state_entropy",
        "factor_observability",
        "factor_reliability",
        "factor_layer_weights",
        "action_evidence_delta",
        "action_factor_weights",
        "action_factor_contributions",
        "action_correction_rms_ratio",
        "reason_evidence_delta",
        "reason_groundable_mask",
    )
    mechanism = collector["mechanism"]
    for key in keys:
        value = output.get(key)
        if isinstance(value, Tensor):
            if value.ndim == 1:
                value = value.unsqueeze(0)
            mechanism.setdefault(key, []).append(value.detach().cpu())


def _finalize_collector(collector: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_visual": _cat(collector["action_visual"]),
        "action_final": _cat(collector["action_final"]),
        "reason_calalign": _cat(collector["reason_calalign"]),
        "reason_global": _cat(collector["reason_global"]),
        "reason_final": _cat(collector["reason_final"]),
        "mechanism": {
            key: _cat(value) for key, value in collector["mechanism"].items()
        },
        "eval_mode_time": float(collector["eval_mode_time"]),
    }


@torch.no_grad()
def collect_outputs(
    model: METEROIAModel,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    progress: float,
    max_batches: int | None = None,
    sequential_modes: bool = True,
) -> dict[str, Any]:
    """Encode once per batch, then decode all cheap HECA interventions."""
    model.eval()
    collectors = {"clean": _new_collector()}
    if sequential_modes:
        collectors.update({name: _new_collector() for name in CHEAP_SAME_FORWARD_MODES})
    labels_action: list[Tensor] = []
    labels_reason: list[Tensor] = []
    file_names: list[str] = []
    encoded_batches = 0
    foundation = getattr(model, "foundation", None)
    backbone_calls_before = getattr(foundation, "ordinary_dino_calls", None)

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        # This is the only backbone invocation for this batch.  Every mode
        # below receives the identical encoded field and only changes decode.
        field = model.encode_images(images)
        encoded_batches += 1
        labels_action.append(batch["action"].detach().cpu())
        labels_reason.append(batch["reason"].detach().cpu())
        file_names.extend(str(name) for name in batch["file_name"])
        for name, flags in (("clean", ()), *CHEAP_SAME_FORWARD_MODES.items()):
            if name not in collectors:
                continue
            start = time.perf_counter()
            output = model.decode_from_field(
                field, progress=progress, diagnostic_modes=flags
            )
            _append_output(
                collectors[name],
                output,
                elapsed=time.perf_counter() - start,
                collect_mechanism=name == "clean",
            )
            del output
        del field, images

    backbone_calls_after = getattr(foundation, "ordinary_dino_calls", None)
    if isinstance(backbone_calls_before, int) and isinstance(backbone_calls_after, int):
        actual_backbone_calls = backbone_calls_after - backbone_calls_before
        if actual_backbone_calls != encoded_batches:
            raise RuntimeError(
                "DINO call mismatch: cheap diagnostics must use exactly one "
                f"backbone encode per batch, got {actual_backbone_calls} for "
                f"{encoded_batches} batches"
            )

    main = _finalize_collector(collectors["clean"])
    main.update(
        {
            "labels_action": _cat(labels_action),
            "labels_reason": _cat(labels_reason),
            "file_names": file_names,
            "dino_call_count": (
                encoded_batches
                if not isinstance(backbone_calls_before, int)
                else backbone_calls_after - backbone_calls_before
            ),
        }
    )
    modes: dict[str, dict[str, Any]] = {}
    for name in CHEAP_SAME_FORWARD_MODES:
        if name not in collectors:
            continue
        mode = _finalize_collector(collectors[name])
        modes[name] = {
            "action_final": mode["action_final"],
            "reason_final": mode["reason_final"],
            "metrics": _metric_pair(
                mode["action_final"],
                mode["reason_final"],
                main["labels_action"],
                main["labels_reason"],
            ),
            "eval_mode_time": mode["eval_mode_time"],
            "dino_call_count": 0,
            "execution": "same_forward_decode",
        }
    return {**main, "modes": modes}


def branch_metrics(collected: dict[str, Any]) -> dict[str, Any]:
    labels_action = collected["labels_action"]
    labels_reason = collected["labels_reason"]
    result = {
        "action_visual": multilabel_metrics_from_logits(
            collected["action_visual"], labels_action, prefix="Act_"
        ),
        "action_final": multilabel_metrics_from_logits(
            collected["action_final"], labels_action, prefix="Act_"
        ),
        "reason_global": multilabel_metrics_from_logits(
            collected["reason_global"], labels_reason, prefix="Exp_"
        ),
        "reason_calalign": multilabel_metrics_from_logits(
            collected["reason_calalign"], labels_reason, prefix="Exp_"
        ),
        "reason_final": multilabel_metrics_from_logits(
            collected["reason_final"], labels_reason, prefix="Exp_"
        ),
    }
    result.update(
        {name: payload["metrics"] for name, payload in collected["modes"].items()}
    )
    return result


def _split_calibration(
    calibration: METERCalibrationResult, action_dim: int, reason_dim: int
) -> tuple[METERCalibrationResult, METERCalibrationResult]:
    def part(start: int, end: int) -> METERCalibrationResult:
        return METERCalibrationResult(
            theta=calibration.theta[start:end],
            temperature=(
                None
                if calibration.temperature is None
                else calibration.temperature[start:end]
            ),
            strategy=calibration.strategy,
            model_state_hash_before=calibration.model_state_hash_before,
            model_state_hash_after=calibration.model_state_hash_after,
            fit_split=calibration.fit_split,
            representation_updated=False,
            accepted=calibration.accepted,
            fallback_reason=calibration.fallback_reason,
        )

    return part(0, action_dim), part(action_dim, action_dim + reason_dim)


def metrics_summary(
    collected: dict[str, Any],
    calibration: METERCalibrationResult | None = None,
) -> dict[str, Any]:
    raw = _metric_pair(
        collected["action_final"],
        collected["reason_final"],
        collected["labels_action"],
        collected["labels_reason"],
    )
    raw["raw_joint"] = raw.pop("joint")
    deploy = dict(raw)
    if calibration is not None:
        action_cal, reason_cal = _split_calibration(
            calibration,
            collected["action_final"].shape[1],
            collected["reason_final"].shape[1],
        )
        deploy_action = apply_meter_deploy(collected["action_final"], action_cal)
        deploy_reason = apply_meter_deploy(collected["reason_final"], reason_cal)
        deploy = _metric_pair(
            deploy_action,
            deploy_reason,
            collected["labels_action"],
            collected["labels_reason"],
        )
        deploy["deploy_joint"] = deploy.pop("joint")
    else:
        deploy["deploy_joint"] = deploy.pop("raw_joint")
    return {"metrics_raw": raw, "metrics_deploy": deploy}


def mechanism_stats_from_collected(collected: dict[str, Any]) -> dict[str, Any]:
    value = collected.get("mechanism", {})
    if not value:
        return {"available": False}
    anchor = value["factor_anchor_map"]
    null = value["factor_null_mass"]
    full_anchor = torch.cat([anchor, null.unsqueeze(-1)], -1)
    state = value["factor_state_prob"]
    visual = collected["action_visual"].float()
    delta = value["action_evidence_delta"].float()
    contribution = value["action_factor_contributions"].float()
    reason_delta = value["reason_evidence_delta"].float()
    modes = collected.get("modes", {})

    def mode_delta(name: str, branch: str) -> list[float]:
        if name not in modes:
            return []
        clean = collected[branch].float()
        changed = modes[name][branch].float()
        return (clean - changed).abs().mean(0).tolist()

    return {
        "available": True,
        "anchor_entropy_per_factor": _entropy(full_anchor).mean(0).tolist(),
        "anchor_null_mass_per_factor": null.mean(0).tolist(),
        "observability_per_factor": value["factor_observability"].mean(0).tolist(),
        "reliability_per_factor": value["factor_reliability"].mean(0).tolist(),
        "state_entropy_per_factor": _entropy(state).mean(0).tolist(),
        "layer_weights": value["factor_layer_weights"][0].tolist(),
        "action_correction_rms_ratio_per_action": (
            delta.square().mean(0).sqrt()
            / visual.square().mean(0).sqrt().clamp_min(1e-6)
        ).tolist(),
        "reason_correction_rms_per_label": reason_delta.square().mean(0).sqrt().tolist(),
        "factor_contribution_rms_by_action_factor": contribution.square()
        .mean(0)
        .sqrt()
        .tolist(),
        "factor_weight_mean_by_action_factor": value["action_factor_weights"]
        .mean(0)
        .tolist(),
        "factor_off_delta_per_action": mode_delta("factor_off", "action_final"),
        "state_uniform_delta_per_action": mode_delta(
            "state_uniform", "action_final"
        ),
        # Legacy readers consume this key.  It intentionally aliases the
        # full state-uniform recomputation rather than a state-free shortcut.
        "state_off_delta_per_action": mode_delta(
            "state_uniform", "action_final"
        ),
        "schema_corruption_delta_per_action": mode_delta(
            "schema_corruption", "action_final"
        ),
        "cross_sample_swap_delta_per_action": mode_delta(
            "cross_sample_swap", "action_final"
        ),
        "state_corruption_delta_per_action": mode_delta(
            "state_corruption", "action_final"
        ),
        "reason_correction_off_delta_per_label": mode_delta(
            "reason_correction_off", "reason_final"
        ),
        "eval_mode_time": {
            name: payload["eval_mode_time"] for name, payload in modes.items()
        },
        "dino_call_count": {
            "main": collected["dino_call_count"],
            **{name: payload["dino_call_count"] for name, payload in modes.items()},
        },
        "ablation_manifest": heca_ablation_manifest(),
    }


def evaluate_checkpoint(
    model: METEROIAModel,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    progress: float,
    calibration: METERCalibrationResult | None = None,
) -> dict[str, Any]:
    collected = collect_outputs(model, loader, device, progress=progress)
    return {
        "collected": collected,
        "branches": branch_metrics(collected),
        "summary": metrics_summary(collected, calibration),
        "mechanism": mechanism_stats_from_collected(collected),
        "ablation_manifest": heca_ablation_manifest(),
    }


def _calibration_from_checkpoint(payload: dict[str, Any]) -> METERCalibrationResult | None:
    value = payload.get("calibration") or {}
    if value.get("theta") is None:
        return None
    return METERCalibrationResult(
        theta=torch.as_tensor(value["theta"]),
        temperature=(
            None
            if value.get("temperature") is None
            else torch.as_tensor(value["temperature"])
        ),
        strategy=str(value.get("strategy", "per_label")),
        model_state_hash_before="checkpoint",
        model_state_hash_after="checkpoint",
        fit_split="train_calib",
        representation_updated=False,
        accepted=bool(value.get("accepted", True)),
        fallback_reason=str(value.get("fallback_reason", "")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--use_mock_dino", action="store_true")
    args = parser.parse_args()
    config = load_meter_config(args.config)
    device = torch.device(args.device)
    model = METEROIAModel(
        dim=int(config["model"]["dim"]),
        action_dim=int(config["model"]["action_dim"]),
        reason_dim=int(config["model"]["reason_dim"]),
        selected_layers=tuple(config["backbone"]["selected_layers"]),
        pretrained_weights=config["backbone"]["pretrained_weights"],
        use_mock_dino=args.use_mock_dino,
        factor_rank=int(config["model"].get("factor_rank", 16)),
    ).to(device)
    payload = load_checkpoint(args.checkpoint, model=model)
    dataset = METERDataset(
        data_root=config["data"]["data_root"],
        raw_root=config["data"]["raw_root"],
        split="test",
        transform=meter_image_transform(),
    )
    indices = list(range(len(dataset)))
    if args.max_test_samples:
        indices = indices[: args.max_test_samples]
    workers = int(config["data"].get("num_workers", 4))
    kwargs: dict[str, Any] = {
        "batch_size": int(config["training"]["batch_size"]),
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": bool(config["data"].get("pin_memory", True)),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(config["data"].get("prefetch_factor", 2))
    loader = DataLoader(Subset(dataset, indices), **kwargs)
    result = evaluate_checkpoint(
        model,
        loader,
        device,
        progress=1.0,
        calibration=_calibration_from_checkpoint(payload),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics_summary.json", result["summary"])
    write_json(output / "branch_metrics.json", result["branches"])
    write_json(output / "mechanism_stats.json", result["mechanism"])
    write_json(output / "heca_ablation_manifest.json", result["ablation_manifest"])


if __name__ == "__main__":
    main()
