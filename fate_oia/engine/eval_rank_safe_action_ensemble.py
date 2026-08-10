from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.aie_splits import stable_split_ids
from fate_oia.engine.train_aie_oia import (
    build_model as build_aie_model,
    canonical_model_state_dict,
    make_dataset,
)
from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.utils.aie_calibration import apply_posthoc_threshold, fit_posthoc_thresholds


@dataclass(frozen=True)
class EnsembleFit:
    family: str
    weights: list[float]
    thresholds: list[float]
    global_weight: float
    global_cv_mf1: float
    per_action_cv_mf1: float
    selected_cv_mf1: float


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _blend(left: Tensor, right: Tensor, weights: Tensor | float) -> Tensor:
    weight = torch.as_tensor(weights, dtype=torch.float32).reshape(1, -1)
    if weight.numel() == 1:
        weight = weight.expand(1, left.shape[1])
    return weight * left.float() + (1.0 - weight) * right.float()


def _macro_f1(logits: Tensor, target: Tensor) -> float:
    return float(multilabel_metrics_from_logits(logits, target.float(), prefix="Act_")["Act_mF1"])


def _stable_folds(names: list[str], folds: int) -> Tensor:
    values = [int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:16], 16) % folds for name in names]
    assignment = torch.tensor(values, dtype=torch.long)
    if any(not bool((assignment == fold).any()) for fold in range(folds)):
        raise RuntimeError("Calibration fold construction produced an empty fold")
    return assignment


def _fit_threshold(logits: Tensor, target: Tensor) -> Tensor:
    return fit_posthoc_thresholds(
        logits, target, [list(range(logits.shape[1]))], shrinkage_support=50.0, grid_step=0.01
    )["threshold_prob"]


def _cross_validated_global(
    pact: Tensor,
    aie: Tensor,
    target: Tensor,
    fold_ids: Tensor,
    candidates: Tensor,
) -> tuple[float, float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for candidate in candidates.tolist():
        fold_scores = []
        blended = _blend(pact, aie, candidate)
        for fold in sorted(fold_ids.unique().tolist()):
            train = fold_ids != fold
            valid = ~train
            threshold = _fit_threshold(blended[train], target[train])
            fold_scores.append(_macro_f1(apply_posthoc_threshold(blended[valid], threshold), target[valid]))
        rows.append({"weight": float(candidate), "cv_mf1": float(sum(fold_scores) / len(fold_scores))})
    best = max(rows, key=lambda row: (row["cv_mf1"], -abs(row["weight"] - 0.5)))
    return best["weight"], best["cv_mf1"], rows


def _cross_validated_per_action(
    pact: Tensor,
    aie: Tensor,
    target: Tensor,
    fold_ids: Tensor,
    global_weight: float,
    radius: float = 0.20,
    min_label_gain: float = 0.002,
) -> tuple[Tensor, float, list[dict[str, Any]]]:
    candidates = torch.arange(max(0.0, global_weight - radius), min(1.0, global_weight + radius) + 1e-6, 0.025)
    selected = torch.full((target.shape[1],), global_weight)
    diagnostics: list[dict[str, Any]] = []
    for label in range(target.shape[1]):
        scores: dict[float, float] = {}
        for candidate in candidates.tolist():
            logits = _blend(pact[:, label : label + 1], aie[:, label : label + 1], candidate)
            fold_scores = []
            for fold in sorted(fold_ids.unique().tolist()):
                train = fold_ids != fold
                valid = ~train
                threshold = _fit_threshold(logits[train], target[train, label : label + 1])
                fold_scores.append(_macro_f1(apply_posthoc_threshold(logits[valid], threshold), target[valid, label : label + 1]))
            scores[float(candidate)] = float(sum(fold_scores) / len(fold_scores))
        global_key = min(scores, key=lambda value: abs(value - global_weight))
        best_key = max(scores, key=lambda value: (scores[value], -abs(value - global_weight)))
        accepted = scores[best_key] >= scores[global_key] + min_label_gain
        selected[label] = best_key if accepted else global_weight
        diagnostics.append({
            "label": label,
            "global_weight": global_weight,
            "candidate_weight": best_key,
            "selected_weight": float(selected[label]),
            "global_cv_f1": scores[global_key],
            "candidate_cv_f1": scores[best_key],
            "accepted": accepted,
        })

    fold_scores = []
    blended = _blend(pact, aie, selected)
    for fold in sorted(fold_ids.unique().tolist()):
        train = fold_ids != fold
        valid = ~train
        threshold = _fit_threshold(blended[train], target[train])
        fold_scores.append(_macro_f1(apply_posthoc_threshold(blended[valid], threshold), target[valid]))
    return selected, float(sum(fold_scores) / len(fold_scores)), diagnostics


def fit_ensemble(
    pact_logits: Tensor,
    aie_logits: Tensor,
    target: Tensor,
    names: list[str],
    folds: int = 5,
) -> tuple[EnsembleFit, dict[str, Any]]:
    """Fit every deploy parameter from train_calib only."""
    if pact_logits.shape != aie_logits.shape or pact_logits.shape != target.shape:
        raise ValueError("Calibration logits/targets must have identical shapes")
    if len(names) != target.shape[0] or len(set(names)) != len(names):
        raise ValueError("Calibration file names must be unique and aligned")
    fold_ids = _stable_folds(names, folds)
    candidates = torch.arange(0.0, 1.0001, 0.025)
    global_weight, global_cv, global_rows = _cross_validated_global(
        pact_logits, aie_logits, target, fold_ids, candidates
    )
    per_weights, per_cv, per_rows = _cross_validated_per_action(
        pact_logits, aie_logits, target, fold_ids, global_weight
    )
    # Per-action freedom is accepted only when its out-of-fold macro-F1 gain is material.
    if per_cv >= global_cv + 0.001:
        family, weights, selected_cv = "per_action_shrunk", per_weights, per_cv
    else:
        family = "global"
        weights = torch.full((target.shape[1],), global_weight)
        selected_cv = global_cv
    blended = _blend(pact_logits, aie_logits, weights)
    thresholds = _fit_threshold(blended, target)
    fit = EnsembleFit(
        family=family,
        weights=weights.tolist(),
        thresholds=thresholds.tolist(),
        global_weight=global_weight,
        global_cv_mf1=global_cv,
        per_action_cv_mf1=per_cv,
        selected_cv_mf1=selected_cv,
    )
    return fit, {"global_candidates": global_rows, "per_action_candidates": per_rows, "fold_ids": fold_ids.tolist()}


@torch.no_grad()
def _collect_calibration_branch(
    model,
    loader: DataLoader,
    device: torch.device,
    branch: str,
) -> tuple[Tensor, Tensor, list[str]]:
    model.eval()
    rows, targets, names = [], [], []
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(images, action_scale=1.0, reason_scale=0.6)["action_logits_final"]
        rows.append(logits.float().cpu())
        targets.append(batch["action"].float().cpu())
        names.extend(batch["file_name"])
        if step == 1 or step % 50 == 0:
            print(json.dumps({
                "event": "calibration_forward", "branch": branch, "batch": step, "samples": len(names)
            }), flush=True)
    return torch.cat(rows), torch.cat(targets), names


def _action_metrics(logits: Tensor, target: Tensor) -> dict[str, Any]:
    return multilabel_metrics_from_logits(logits.float(), target.float(), prefix="Act_")


def _paired_bootstrap(
    candidate: Tensor,
    baseline: Tensor,
    target: Tensor,
    samples: int = 1000,
    seed: int = 20260810,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    differences = []
    for _ in range(samples):
        indices = torch.randint(0, target.shape[0], (target.shape[0],), generator=generator)
        differences.append(_macro_f1(candidate[indices], target[indices]) - _macro_f1(baseline[indices], target[indices]))
    values = torch.tensor(differences)
    return {
        "mean_delta": float(values.mean()),
        "ci95_low": float(values.quantile(0.025)),
        "ci95_high": float(values.quantile(0.975)),
        "probability_positive": float((values > 0).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pact-config", required=True)
    parser.add_argument("--aie-config", required=True)
    parser.add_argument("--pact-checkpoint", required=True)
    parser.add_argument("--aie-checkpoint", required=True)
    parser.add_argument("--pact-test-dir", required=True)
    parser.add_argument("--aie-test-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--include-train-audit", action="store_true")
    parser.add_argument("--test-release-count", type=int, default=1)
    parser.add_argument("--calib-cache")
    parser.add_argument("--locked-global-weight", type=float)
    parser.add_argument("--selection-provenance", default="calibration_cv")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pact_cfg, aie_cfg = _load_config(args.pact_config), _load_config(args.aie_config)
    split_contract = (
        int(pact_cfg["data"]["split_seed"]),
        float(pact_cfg["data"]["train_calib_fraction"]),
        int(pact_cfg["data"]["train_audit_count"]),
    )
    aie_contract = (
        int(aie_cfg["data"]["split_seed"]),
        float(aie_cfg["data"]["train_calib_fraction"]),
        int(aie_cfg["data"]["train_audit_count"]),
    )
    if split_contract != aie_contract:
        raise RuntimeError(f"Calibration split contract mismatch: {split_contract} != {aie_contract}")

    device = torch.device(args.device)
    # The strongest PACT CONTROL artifact intentionally uses the unmodified AIE
    # control architecture; only the experimental PACT arm uses PACTOIAModel.
    pact_checkpoint = torch.load(args.pact_checkpoint, map_location="cpu")
    aie_checkpoint = torch.load(args.aie_checkpoint, map_location="cpu")
    if args.calib_cache:
        cache = torch.load(args.calib_cache, map_location="cpu")
        pact_calib = cache["pact_logits"].float()
        aie_calib = cache["aie_logits"].float()
        calib_target = cache["labels"].float()
        calib_names = list(cache["file_names"])
        fit_source = "train_calib_cache"
    else:
        train = make_dataset(aie_cfg, "train")
        all_names = [sample.file_name for sample in train.samples]
        split = stable_split_ids(all_names, *split_contract)
        name_to_index = {sample.file_name: index for index, sample in enumerate(train.samples)}
        fit_names = list(split["train_calib"])
        fit_source = "train_calib"
        if args.include_train_audit:
            fit_names.extend(split["train_audit"])
            fit_source = "train_calib_plus_train_audit"
        if len(set(fit_names)) != len(fit_names):
            raise RuntimeError("Fit splits overlap")
        calib_indices = [name_to_index[name] for name in fit_names]
        loader_kwargs = dict(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=__import__("fate_oia.engine.train_aie_oia", fromlist=["collate"]).collate,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 4
        calib_loader = DataLoader(Subset(train, calib_indices), **loader_kwargs)
        pact_model = build_aie_model(pact_cfg, device)
        pact_model.load_state_dict(canonical_model_state_dict(pact_checkpoint["model"]), strict=True)
        pact_calib, calib_target, calib_names = _collect_calibration_branch(
            pact_model, calib_loader, device, "pact_control"
        )
        del pact_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        aie_model = build_aie_model(aie_cfg, device)
        aie_model.load_state_dict(canonical_model_state_dict(aie_checkpoint["model"]), strict=True)
        aie_calib, aie_target, aie_names = _collect_calibration_branch(
            aie_model, calib_loader, device, "aie_epoch4"
        )
        del aie_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if aie_names != calib_names or not torch.equal(aie_target, calib_target):
            raise RuntimeError("Sequential calibration forwards are not sample-aligned")

    # Fit is complete before any test artifact is loaded.
    if args.locked_global_weight is None:
        fit, cv_diagnostics = fit_ensemble(pact_calib, aie_calib, calib_target, calib_names)
    else:
        locked = float(args.locked_global_weight)
        if not 0.0 <= locked <= 1.0:
            raise ValueError("--locked-global-weight must be in [0, 1]")
        fold_ids = _stable_folds(calib_names, 5)
        _, locked_cv, locked_rows = _cross_validated_global(
            pact_calib, aie_calib, calib_target, fold_ids, torch.tensor([locked])
        )
        locked_weights = torch.full((calib_target.shape[1],), locked)
        locked_thresholds = _fit_threshold(_blend(pact_calib, aie_calib, locked_weights), calib_target)
        fit = EnsembleFit(
            family="locked_global",
            weights=locked_weights.tolist(),
            thresholds=locked_thresholds.tolist(),
            global_weight=locked,
            global_cv_mf1=locked_cv,
            per_action_cv_mf1=locked_cv,
            selected_cv_mf1=locked_cv,
        )
        cv_diagnostics = {"global_candidates": locked_rows, "per_action_candidates": [], "fold_ids": fold_ids.tolist()}
    torch.save({
        "pact_logits": pact_calib,
        "aie_logits": aie_calib,
        "labels": calib_target,
        "file_names": calib_names,
    }, output / "train_calib_logits.pt")
    _write_json(output / "fit_protocol.json", {**asdict(fit), **cv_diagnostics})

    pact_test_dir, aie_test_dir = Path(args.pact_test_dir), Path(args.aie_test_dir)
    pact_test = torch.load(pact_test_dir / "action_logits_final_test.pt", map_location="cpu").float()
    aie_test = torch.load(aie_test_dir / "action_logits_final_test.pt", map_location="cpu").float()
    action_target = torch.load(pact_test_dir / "labels_action_test.pt", map_location="cpu").float()
    pact_names = json.loads((pact_test_dir / "file_names_test.json").read_text(encoding="utf-8"))
    aie_names = json.loads((aie_test_dir / "file_names_test.json").read_text(encoding="utf-8"))
    if pact_names != aie_names or pact_test.shape != aie_test.shape or pact_test.shape != action_target.shape:
        raise RuntimeError("Frozen test artifacts are not sample-aligned")

    weights = torch.tensor(fit.weights)
    thresholds = torch.tensor(fit.thresholds)
    ensemble_raw = _blend(pact_test, aie_test, weights)
    ensemble_deploy = apply_posthoc_threshold(ensemble_raw, thresholds)
    pact_reason = torch.load(pact_test_dir / "reason_logits_final_test.pt", map_location="cpu").float()
    reason_target = torch.load(pact_test_dir / "labels_reason_test.pt", map_location="cpu").float()
    pact_thresholds = torch.as_tensor(pact_checkpoint["metrics"]["calibration_thresholds"]["threshold_prob"])
    reason_deploy = apply_posthoc_threshold(pact_reason, pact_thresholds[4:])
    reason_metrics = multilabel_metrics_from_logits(reason_deploy, reason_target, prefix="Exp_")
    pact_deploy = apply_posthoc_threshold(pact_test, pact_thresholds[:4])
    metrics = {
        "selection_source": fit_source + "_only",
        "selection_provenance": args.selection_provenance,
        "test_release_count": args.test_release_count,
        "fit": asdict(fit),
        "pact_action": _action_metrics(pact_deploy, action_target),
        "aie_action_raw_fixed": _action_metrics(aie_test, action_target),
        "ensemble_action_raw": _action_metrics(ensemble_raw, action_target),
        "ensemble_action_deploy": _action_metrics(ensemble_deploy, action_target),
        "pact_reason_deploy": reason_metrics,
        "joint": 0.5 * _action_metrics(ensemble_deploy, action_target)["Act_mF1"] + 0.5 * reason_metrics["Exp_mF1"],
        "paired_bootstrap_vs_pact": _paired_bootstrap(ensemble_deploy, pact_deploy, action_target),
        "targets_met": {
            "action_mf1_gt_0p73": _action_metrics(ensemble_deploy, action_target)["Act_mF1"] > 0.73,
            "reason_mf1_gt_0p39": reason_metrics["Exp_mF1"] > 0.39,
        },
    }
    _write_json(output / "test_once_metrics.json", metrics)
    torch.save({
        "action_logits_raw": ensemble_raw,
        "action_logits_deploy": ensemble_deploy,
        "reason_logits_deploy": reason_deploy,
        "action_labels": action_target,
        "reason_labels": reason_target,
        "file_names": pact_names,
    }, output / "frozen_test_outputs.pt")
    torch.save({
        "format": "rank_safe_action_ensemble_v1",
        "fit": asdict(fit),
        "pact_reason_thresholds": pact_thresholds[4:].tolist(),
        "pact_checkpoint": str(Path(args.pact_checkpoint).resolve()),
        "aie_checkpoint": str(Path(args.aie_checkpoint).resolve()),
        "pact_checkpoint_sha256": _sha256(args.pact_checkpoint),
        "aie_checkpoint_sha256": _sha256(args.aie_checkpoint),
        "split_contract": split_contract,
        "selection_provenance": args.selection_provenance,
        "calibration_names_sha256": hashlib.sha256("\n".join(calib_names).encode("utf-8")).hexdigest(),
    }, output / "rank_safe_deploy_bundle.pt")
    print(json.dumps({
        "event": "rank_safe_result",
        "family": fit.family,
        "weights": fit.weights,
        "Act_mF1": metrics["ensemble_action_deploy"]["Act_mF1"],
        "Act_oF1": metrics["ensemble_action_deploy"]["Act_oF1"],
        "Act_mAP": metrics["ensemble_action_deploy"]["Act_mAP"],
        "Exp_mF1": reason_metrics["Exp_mF1"],
        "Exp_oF1": reason_metrics["Exp_oF1"],
        "Exp_mAP": reason_metrics["Exp_mAP"],
    }), flush=True)


if __name__ == "__main__":
    main()
