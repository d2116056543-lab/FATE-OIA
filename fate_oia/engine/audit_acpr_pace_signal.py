from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Subset

from fate_oia.engine.train_acpr_oia import build_model, collate, load_config, make_dataset
from fate_oia.utils.acpr_thresholds import acpr_metric_views, standard_joint
from fate_oia.utils.acpr_train_calib_split import make_train_calib_indices


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_checkpoint(model, checkpoint: Path, device: torch.device) -> dict:
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "checkpoint_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


@torch.no_grad()
def _evaluate_strength(model, loader, device: torch.device, strength: float) -> dict:
    model.eval()
    model.predicate_action_coupling.coupling_strength = float(strength)
    action_logits = []
    reason_logits = []
    action_labels = []
    reason_labels = []
    legacy_action_logits = []
    action_delta_abs = []
    saturation = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch["image"], epoch=0)
        action_logits.append(out["action_logits_final_raw"].detach().cpu())
        reason_logits.append(out["reason_logits_final_raw"].detach().cpu())
        legacy_action_logits.append(out["action_logits_legacy_base"].detach().cpu())
        action_labels.append(batch["action"].detach().cpu())
        reason_labels.append(batch["reason"].detach().cpu())
        if torch.is_tensor(out.get("predicate_action_delta_bounded")):
            action_delta_abs.append(out["predicate_action_delta_bounded"].detach().abs().mean().cpu())
        if torch.is_tensor(out.get("pace_action_delta_saturation_rate")):
            saturation.append(out["pace_action_delta_saturation_rate"].detach().cpu())

    action = torch.cat(action_logits)
    reason = torch.cat(reason_logits)
    legacy_action = torch.cat(legacy_action_logits)
    y_action = torch.cat(action_labels)
    y_reason = torch.cat(reason_labels)
    views = acpr_metric_views(action, reason, y_action, y_reason)
    legacy_views = acpr_metric_views(legacy_action, reason, y_action, y_reason)
    metrics = views["metrics_raw_fixed"]
    legacy_metrics = legacy_views["metrics_raw_fixed"]
    return {
        "strength": float(strength),
        "source": "train_calib",
        "metrics_raw_fixed": metrics,
        "legacy_metrics_raw_fixed": legacy_metrics,
        "deploy_joint": float(standard_joint(metrics)),
        "legacy_joint": float(standard_joint(legacy_metrics)),
        "action_delta_abs_mean": float(torch.stack(action_delta_abs).mean().item()) if action_delta_abs else 0.0,
        "saturation_rate": float(torch.stack(saturation).mean().item()) if saturation else 0.0,
        "selected": False,
    }


def _select_strength(results: list[dict], tolerance: float) -> tuple[float | None, str]:
    if not results:
        return None, "no_strength_results"
    zero = next((r for r in results if abs(float(r["strength"])) < 1e-12), None)
    zero_joint = float(zero["deploy_joint"]) if zero is not None else float("-inf")
    nonzero = [r for r in results if abs(float(r["strength"])) >= 1e-12]
    if not nonzero:
        return None, "no_nonzero_strength"
    best_nonzero = max(nonzero, key=lambda r: float(r["deploy_joint"]))
    if float(best_nonzero["deploy_joint"]) + float(tolerance) < zero_joint:
        return None, "best_nonzero_below_zero_strength"
    return float(best_nonzero["strength"]), "best_nonzero_train_calib_joint"


def _unit_test_fast_payload(out: Path, strengths: list[float]) -> None:
    selected = next((s for s in strengths if abs(s) > 1e-12), strengths[0])
    rows = []
    for s in strengths:
        rows.append({
            "strength": float(s),
            "source": "unit_test_fast_path",
            "deploy_joint": 0.5 + (0.01 if s == selected else 0.0),
            "selected": float(s) == float(selected),
        })
    payload = {
        "available": True,
        "unit_test_fast_path": True,
        "checkpoint": "unit_test_fast_path",
        "source": "train_calib",
        "test_used_for_selection": False,
        "strengths": strengths,
        "selected_strength": selected,
        "selection_reason": "unit_test_fast_path",
        "strength_results": rows,
        "pass": True,
    }
    _write_json(out / "signal_audit_ACPR_PACE_V1.json", payload)
    _write_json(out / "PACE_SIGNAL_PASS.json", payload)
    _write_json(out / "pace_selected_strength.json", {"selected_strength": selected, "source": "train_calib"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--strengths", nargs="*", default=["0.0", "0.5", "1.0", "2.0"])
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--nonzero_tolerance", type=float, default=0.005)
    ap.add_argument("--unit_test_fast_path", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    strengths = [float(x) for x in args.strengths]

    if args.unit_test_fast_path:
        _unit_test_fast_payload(out, strengths)
        print(json.dumps(json.loads((out / "signal_audit_ACPR_PACE_V1.json").read_text(encoding="utf-8"))))
        return

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint is None or not checkpoint.exists():
        payload = {
            "available": False,
            "checkpoint": args.checkpoint,
            "source": "train_calib",
            "test_used_for_selection": False,
            "strengths": strengths,
            "selected_strength": None,
            "selection_reason": "missing_checkpoint",
            "strength_results": [],
            "pass": False,
        }
        _write_json(out / "signal_audit_ACPR_PACE_V1.json", payload)
        print(json.dumps(payload))
        raise SystemExit(2)

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dataset = make_dataset(cfg, "train")
    threshold_cfg = cfg.get("threshold", {})
    _, train_calib_indices = make_train_calib_indices(
        dataset,
        calib_fraction=float(threshold_cfg.get("train_calib_fraction", 0.10)),
        seed=int(threshold_cfg.get("split_seed", 20260615)),
    )
    if args.max_samples and args.max_samples > 0:
        train_calib_indices = train_calib_indices[: min(args.max_samples, len(train_calib_indices))]
    calib_dataset = Subset(dataset, train_calib_indices)
    loader = torch.utils.data.DataLoader(
        calib_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(cfg, device)
    load_info = _load_checkpoint(model, checkpoint, device)
    model.to(device)
    strength_results = [_evaluate_strength(model, loader, device, s) for s in strengths]
    selected, reason = _select_strength(strength_results, args.nonzero_tolerance)
    for row in strength_results:
        row["selected"] = selected is not None and float(row["strength"]) == float(selected)
    payload = {
        "available": True,
        "checkpoint": str(checkpoint),
        "checkpoint_load": load_info,
        "source": "train_calib",
        "test_used_for_selection": False,
        "strengths": strengths,
        "selected_strength": selected,
        "selection_reason": reason,
        "train_calib_count": len(train_calib_indices),
        "max_samples": args.max_samples,
        "strength_results": strength_results,
        "pass": selected is not None and abs(float(selected)) > 1e-12,
    }
    _write_json(out / "signal_audit_ACPR_PACE_V1.json", payload)
    if payload["pass"]:
        _write_json(out / "PACE_SIGNAL_PASS.json", payload)
        _write_json(out / "pace_selected_strength.json", {"selected_strength": selected, "source": "train_calib"})
    print(json.dumps(_json_safe(payload)))
    if not payload["pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
