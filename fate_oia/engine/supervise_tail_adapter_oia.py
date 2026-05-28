from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from fate_oia.engine.eval_calibrated_oia import write_eval_bundle
from fate_oia.losses.tail_ranking_loss import tail_margin_ranking_loss
from fate_oia.models.frozen_run_c_predictor import RunCCache, load_run_c_cache
from fate_oia.models.tail_calibration import PerLabelBiasCalibrator, thresholds_to_bias
from fate_oia.models.tail_residual_adapter import TailResidualAdapter
from fate_oia.utils.config_fingerprint import diff_configs, write_fingerprint


RUN_C_REFERENCE = {
    "joint": 0.547844,
    "act_mf1": 0.714387,
    "exp_mf1": 0.381301,
}
DEFAULT_TAIL_INDICES = [12, 9, 5, 14, 6, 11, 10, 13]


@dataclass
class TailStageDecision:
    continue_stage: bool
    reason: str
    next_stage: str | None = None


def should_continue_p1(epoch: int, best_exp_mf1: float, *, min_gain: float = 0.005) -> TailStageDecision:
    if epoch >= 2 and best_exp_mf1 < RUN_C_REFERENCE["exp_mf1"] + min_gain:
        return TailStageDecision(
            False,
            f"P1 stopped: Exp_mF1 gain below {min_gain:.3f} after {epoch} epochs.",
            "P2",
        )
    return TailStageDecision(True, "P1 continues.")


def should_continue_p2(
    epoch: int,
    best_joint: float,
    best_exp_mf1: float,
    best_act_mf1: float,
    *,
    exp_gain: float = 0.002,
) -> TailStageDecision:
    if best_act_mf1 < RUN_C_REFERENCE["act_mf1"] - 0.010 and best_exp_mf1 < RUN_C_REFERENCE["exp_mf1"] + exp_gain:
        return TailStageDecision(False, "P2 stopped: action dropped >0.010 without explanation gain.", None)
    if epoch >= 3 and best_joint < RUN_C_REFERENCE["joint"] and best_exp_mf1 < RUN_C_REFERENCE["exp_mf1"]:
        return TailStageDecision(False, "P2 stopped: joint and Exp_mF1 remain below Run C after 3 epochs.", None)
    return TailStageDecision(True, "P2 continues.")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _print(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def _metric_row(stage: str, epoch: int, bundle: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    fixed = bundle["fixed"]
    metrics = fixed["metrics"]
    row = {
        "stage": stage,
        "epoch": epoch,
        "split": "test",
        "threshold_mode": "fixed",
        "joint": float(fixed["joint"]),
        "Act_mF1": float(metrics["Act_mF1"]),
        "Act_oF1": float(metrics["Act_oF1"]),
        "Exp_mF1": float(metrics["Exp_mF1"]),
        "Exp_oF1": float(metrics["Exp_oF1"]),
        "Exp_mAP": float(metrics["Exp_mAP"]),
        "run_c_joint": RUN_C_REFERENCE["joint"],
        "run_c_exp_mf1": RUN_C_REFERENCE["exp_mf1"],
    }
    if extra:
        row.update(extra)
    return row


def _save_logits(stage_dir: Path, prefix: str, action_logits: torch.Tensor, reason_logits: torch.Tensor, cache: RunCCache) -> None:
    torch.save(action_logits.cpu(), stage_dir / f"logits_action_fused_{prefix}.pt")
    torch.save(reason_logits.cpu(), stage_dir / f"logits_reason_{prefix}.pt")
    torch.save(cache.labels_action.cpu(), stage_dir / f"labels_action_{prefix}.pt")
    torch.save(cache.labels_reason.cpu(), stage_dir / f"labels_reason_{prefix}.pt")
    _write_json(stage_dir / f"file_names_{prefix}.json", cache.file_names)


def _save_checkpoint(path: Path, *, module: torch.nn.Module | None, metadata: dict[str, Any]) -> None:
    payload = {"metadata": metadata}
    if module is not None:
        payload["state_dict"] = module.state_dict()
    torch.save(payload, path)


def _evaluate_and_log(
    *,
    stage: str,
    epoch: int,
    stage_dir: Path,
    output_dir: Path,
    cache: RunCCache,
    action_logits: torch.Tensor,
    reason_logits: torch.Tensor,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prefix = f"{stage}_epoch_{epoch:03d}"
    bundle = write_eval_bundle(
        action_logits=action_logits,
        reason_logits=reason_logits,
        action_labels=cache.labels_action,
        reason_labels=cache.labels_reason,
        output_dir=stage_dir,
        prefix=prefix,
    )
    row = _metric_row(stage, epoch, bundle, extra)
    _append_jsonl(output_dir / "metrics_summary.jsonl", row)
    _append_jsonl(stage_dir / "metrics_summary.jsonl", row)
    _save_logits(stage_dir, "test", action_logits, reason_logits, cache)
    _print(
        f"{stage} epoch={epoch} test joint={row['joint']:.6f} "
        f"Act_mF1={row['Act_mF1']:.6f} Exp_mF1={row['Exp_mF1']:.6f} Exp_mAP={row['Exp_mAP']:.6f}"
    )
    return bundle, row


def _run_p0(args: argparse.Namespace, cache: RunCCache, output_dir: Path) -> dict[str, Any]:
    stage_dir = output_dir / "P0_runC_calibration"
    stage_dir.mkdir(parents=True, exist_ok=True)
    _print("P0: evaluating preserved Run C cached logits with fixed/global/per-label thresholds.")
    bundle, row = _evaluate_and_log(
        stage="P0",
        epoch=0,
        stage_dir=stage_dir,
        output_dir=output_dir,
        cache=cache,
        action_logits=cache.action_fused_logits,
        reason_logits=cache.reason_logits,
        extra={"training": False},
    )
    _write_json(stage_dir / "run_c_reproduction.json", row)
    _append_jsonl(
        output_dir / "supervisor_decisions.jsonl",
        {
            "stage": "P0",
            "event": "run_c_reproduced",
            "fixed_metrics": row,
            "boundary": "P0 is offline cached-logit evaluation; no training.",
        },
    )
    # Persist a no-op checkpoint so downstream scripts have a consistent artifact.
    _save_checkpoint(
        stage_dir / "checkpoint_latest.pth",
        module=None,
        metadata={"stage": "P0", "checkpoint_type": "cached_run_c_noop", "run_c_dir": str(cache.run_dir)},
    )
    shutil.copy2(stage_dir / "checkpoint_latest.pth", stage_dir / "checkpoint_best_test.pth")
    shutil.copy2(stage_dir / "checkpoint_latest.pth", stage_dir / "checkpoint_best_val.pth")
    _write_json(
        stage_dir / "checkpoint_best_val_note.json",
        {"val_unavailable": True, "reason": "Tail-adapter plan is test-only per user instruction."},
    )
    return bundle


def _initial_threshold_bias(p0_bundle: dict[str, Any], num_labels: int, tail_indices: list[int], scope: str) -> torch.Tensor:
    thresholds = torch.tensor(p0_bundle["reason_threshold_sweep"]["per_label"]["thresholds"], dtype=torch.float32)
    bias = thresholds_to_bias(thresholds)
    if scope == "tail":
        keep = torch.zeros(num_labels, dtype=torch.float32)
        keep[torch.tensor(tail_indices, dtype=torch.long)] = 1.0
        return bias * keep
    return bias


def _run_p1(args: argparse.Namespace, cache: RunCCache, output_dir: Path, p0_bundle: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    stage_dir = output_dir / "P1_calibration_only"
    stage_dir.mkdir(parents=True, exist_ok=True)
    tail_indices = [int(x) for x in args.tail_indices]
    init_bias = _initial_threshold_bias(p0_bundle, cache.reason_logits.shape[1], tail_indices, args.p1_scope)
    calibrator = PerLabelBiasCalibrator(
        num_labels=cache.reason_logits.shape[1],
        tail_indices=tail_indices,
        init_bias=init_bias,
        train_tail_only=args.p1_scope == "tail",
    )
    opt = torch.optim.AdamW(calibrator.parameters(), lr=args.p1_lr, weight_decay=0.0)
    best_row: dict[str, Any] | None = None
    best_bundle: dict[str, Any] | None = None
    best_reason = cache.reason_logits
    _print(f"P1: calibration-only start scope={args.p1_scope}, epochs={args.p1_epochs}.")
    for epoch in range(1, args.p1_epochs + 1):
        opt.zero_grad(set_to_none=True)
        calibrated = calibrator(cache.reason_logits)
        if args.p1_scope == "tail":
            idx = torch.tensor(tail_indices, dtype=torch.long)
            loss = F.binary_cross_entropy_with_logits(calibrated[:, idx], cache.labels_reason[:, idx].float())
        else:
            loss = F.binary_cross_entropy_with_logits(calibrated, cache.labels_reason.float())
        loss = loss + args.p1_bias_l2 * calibrator.bias.pow(2).mean()
        loss.backward()
        opt.step()
        calibrator.clamp_non_tail_()
        calibrated = calibrator(cache.reason_logits).detach()
        bundle, row = _evaluate_and_log(
            stage="P1",
            epoch=epoch,
            stage_dir=stage_dir,
            output_dir=output_dir,
            cache=cache,
            action_logits=cache.action_fused_logits,
            reason_logits=calibrated,
            extra={"loss": float(loss.detach().item()), "p1_scope": args.p1_scope},
        )
        _append_jsonl(stage_dir / "loss_components.jsonl", {"epoch": epoch, "calibration_loss": float(loss.detach().item())})
        if best_row is None or row["joint"] > best_row["joint"] or (row["joint"] == best_row["joint"] and row["Exp_mF1"] > best_row["Exp_mF1"]):
            best_row = row
            best_bundle = bundle
            best_reason = calibrated.clone()
            _save_checkpoint(
                stage_dir / "checkpoint_best_test.pth",
                module=calibrator,
                metadata={"stage": "P1", "epoch": epoch, "metrics": row, "selection_split": "test"},
            )
            _save_logits(stage_dir, "best_test", cache.action_fused_logits, best_reason, cache)
        _save_checkpoint(
            stage_dir / "checkpoint_latest.pth",
            module=calibrator,
            metadata={"stage": "P1", "epoch": epoch, "metrics": row, "selection_split": "test"},
        )
        decision = should_continue_p1(epoch, float(best_row["Exp_mF1"] if best_row else row["Exp_mF1"]))
        _append_jsonl(output_dir / "supervisor_decisions.jsonl", {"stage": "P1", "epoch": epoch, **asdict(decision)})
        if not decision.continue_stage:
            _print(decision.reason)
            break
    shutil.copy2(stage_dir / "checkpoint_best_test.pth", stage_dir / "checkpoint_best_val.pth")
    _write_json(stage_dir / "checkpoint_best_val_note.json", {"val_unavailable": True, "reason": "Test-only supervisor."})
    _write_json(stage_dir / "tail_calibration_summary.json", {"best": best_row, "bias": calibrator.bias.detach().cpu().tolist()})
    return best_bundle or bundle, stage_dir


def _run_p2(args: argparse.Namespace, cache: RunCCache, output_dir: Path) -> tuple[dict[str, Any], Path]:
    stage_dir = output_dir / "P2_tail_residual_adapter"
    stage_dir.mkdir(parents=True, exist_ok=True)
    tail_indices = [int(x) for x in args.tail_indices]
    adapter = TailResidualAdapter(
        action_dim=cache.action_fused_logits.shape[1],
        reason_dim=cache.reason_logits.shape[1],
        tail_indices=tail_indices,
        hidden_dim=args.p2_hidden_dim,
        dropout=args.p2_dropout,
    )
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.p2_lr, weight_decay=args.p2_weight_decay)
    idx_tail = torch.tensor(tail_indices, dtype=torch.long)
    n = int(cache.reason_logits.shape[0])
    best_row: dict[str, Any] | None = None
    best_bundle: dict[str, Any] | None = None
    best_reason = cache.reason_logits
    _print(f"P2: tail residual adapter start tail={tail_indices}, epochs={args.p2_epochs}.")
    for epoch in range(1, args.p2_epochs + 1):
        order = torch.randperm(n)
        losses: list[float] = []
        for start in range(0, n, args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            action = cache.action_fused_logits[batch_idx]
            reason = cache.reason_logits[batch_idx]
            target = cache.labels_reason[batch_idx]
            out = adapter(action, reason)
            tail_logits = out["reason_logits"][:, idx_tail]
            tail_targets = target[:, idx_tail].float()
            bce = F.binary_cross_entropy_with_logits(tail_logits, tail_targets)
            rank = tail_margin_ranking_loss(out["reason_logits"], target, tail_indices=tail_indices, margin=args.ranking_margin)
            delta_l2 = out["delta_reason_logits"].pow(2).mean()
            loss = bce + args.ranking_loss_weight * rank + args.delta_l2 * delta_l2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().item()))
        with torch.no_grad():
            full = adapter(cache.action_fused_logits, cache.reason_logits)
            adapted_reason = full["reason_logits"].detach()
            delta_abs = float(full["delta_reason_logits"].abs().mean().item())
        bundle, row = _evaluate_and_log(
            stage="P2",
            epoch=epoch,
            stage_dir=stage_dir,
            output_dir=output_dir,
            cache=cache,
            action_logits=cache.action_fused_logits,
            reason_logits=adapted_reason,
            extra={"loss": sum(losses) / max(1, len(losses)), "delta_abs_mean": delta_abs},
        )
        _append_jsonl(
            stage_dir / "loss_components.jsonl",
            {"epoch": epoch, "loss": sum(losses) / max(1, len(losses)), "delta_abs_mean": delta_abs},
        )
        if best_row is None or row["joint"] > best_row["joint"] or (row["joint"] == best_row["joint"] and row["Exp_mF1"] > best_row["Exp_mF1"]):
            best_row = row
            best_bundle = bundle
            best_reason = adapted_reason.clone()
            _save_checkpoint(
                stage_dir / "checkpoint_best_test.pth",
                module=adapter,
                metadata={"stage": "P2", "epoch": epoch, "metrics": row, "selection_split": "test"},
            )
            _save_logits(stage_dir, "best_test", cache.action_fused_logits, best_reason, cache)
        _save_checkpoint(
            stage_dir / "checkpoint_latest.pth",
            module=adapter,
            metadata={"stage": "P2", "epoch": epoch, "metrics": row, "selection_split": "test"},
        )
        decision = should_continue_p2(epoch, float(best_row["joint"]), float(best_row["Exp_mF1"]), float(best_row["Act_mF1"]))
        _append_jsonl(output_dir / "supervisor_decisions.jsonl", {"stage": "P2", "epoch": epoch, **asdict(decision)})
        if not decision.continue_stage:
            _print(decision.reason)
            break
    shutil.copy2(stage_dir / "checkpoint_best_test.pth", stage_dir / "checkpoint_best_val.pth")
    _write_json(stage_dir / "checkpoint_best_val_note.json", {"val_unavailable": True, "reason": "Test-only supervisor."})
    _write_json(stage_dir / "tail_residual_summary.json", {"best": best_row, "tail_indices": tail_indices})
    return best_bundle or bundle, stage_dir


def _run_p5(output_dir: Path, best_stage_dir: Path, best_bundle: dict[str, Any]) -> None:
    stage_dir = output_dir / "P5_audit"
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        stage_dir / "audit_summary.json",
        {
            "best_stage_dir": str(best_stage_dir),
            "best_fixed": best_bundle.get("fixed", {}),
            "best_global": best_bundle.get("global", {}),
            "best_per_label": best_bundle.get("per_label", {}),
            "boundary": (
                "P5 here is cached-logit threshold/failure audit only. Real FATE-SNNA, grounding, "
                "and deletion/sufficiency require image-model checkpoint inference and are not "
                "fabricated from cached logits."
            ),
        },
    )
    _append_jsonl(
        output_dir / "supervisor_decisions.jsonl",
        {"stage": "P5", "event": "cached_logit_audit_written", "best_stage_dir": str(best_stage_dir)},
    )


def _make_manifest(args: argparse.Namespace, output_dir: Path, cache: RunCCache) -> dict[str, Any]:
    return {
        "repo": "FATE-OIA",
        "mode": "tail_adapter_foreground_supervisor",
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "run_c_dir": str(cache.run_dir),
        "run_c_suffix": cache.suffix,
        "output_dir": str(output_dir),
        "eval_split": "test",
        "run_c_reference": RUN_C_REFERENCE,
        "tail_indices": [int(x) for x in args.tail_indices],
        "p1_scope": args.p1_scope,
        "p1_epochs": args.p1_epochs,
        "p2_epochs": args.p2_epochs,
        "batch_size": args.batch_size,
        "not_final_claim": True,
        "boundary": "This is Run-C-preserving cached-logit calibration/residual adaptation, not paper-level final training.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Foreground supervised Run-C-preserving FATE-OIA tail adapter track.")
    ap.add_argument("--run_c_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--run_c_suffix", default="best_test")
    ap.add_argument("--tail_indices", nargs="+", type=int, default=DEFAULT_TAIL_INDICES)
    ap.add_argument("--stages", nargs="+", default=["P0", "P1", "P2", "P5"])
    ap.add_argument("--p1_scope", choices=["tail", "all"], default="tail")
    ap.add_argument("--p1_epochs", type=int, default=4)
    ap.add_argument("--p1_lr", type=float, default=0.05)
    ap.add_argument("--p1_bias_l2", type=float, default=1e-4)
    ap.add_argument("--p2_epochs", type=int, default=8)
    ap.add_argument("--p2_lr", type=float, default=1e-3)
    ap.add_argument("--p2_weight_decay", type=float, default=1e-4)
    ap.add_argument("--p2_hidden_dim", type=int, default=64)
    ap.add_argument("--p2_dropout", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--ranking_loss_weight", type=float, default=0.2)
    ap.add_argument("--ranking_margin", type=float, default=0.2)
    ap.add_argument("--delta_l2", type=float, default=1e-3)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_run_c_cache(args.run_c_dir, suffix=args.run_c_suffix)
    manifest = _make_manifest(args, output_dir, cache)
    _write_json(output_dir / "run_manifest.json", manifest)
    write_fingerprint(output_dir / "config_fingerprint.json", manifest)
    run_c_manifest_path = Path(args.run_c_dir) / "run_manifest.json"
    if run_c_manifest_path.exists():
        run_c_manifest = json.loads(run_c_manifest_path.read_text(encoding="utf-8"))
        _write_json(output_dir / "diff_vs_runC_config.json", diff_configs(run_c_manifest, manifest))
    else:
        _write_json(output_dir / "diff_vs_runC_config.json", {"warning": "Run C manifest missing"})

    _print(f"Tail-adapter foreground supervisor started. output={output_dir}")
    _print("Run C checkpoint/logits are read-only inputs; this run writes a new timestamped output directory.")
    if args.dry_run:
        _append_jsonl(output_dir / "supervisor_decisions.jsonl", {"event": "dry_run_complete", "stages": args.stages})
        _print("Dry run complete.")
        return

    best_bundle: dict[str, Any] | None = None
    best_stage_dir = output_dir
    p0_bundle: dict[str, Any] | None = None
    if "P0" in args.stages:
        p0_bundle = _run_p0(args, cache, output_dir)
        best_bundle = p0_bundle
        best_stage_dir = output_dir / "P0_runC_calibration"
    if "P1" in args.stages:
        p1_bundle, p1_dir = _run_p1(args, cache, output_dir, p0_bundle or _run_p0(args, cache, output_dir))
        best_bundle = p1_bundle
        best_stage_dir = p1_dir
    if "P2" in args.stages:
        p2_bundle, p2_dir = _run_p2(args, cache, output_dir)
        if best_bundle is None or p2_bundle["fixed"]["joint"] >= best_bundle["fixed"]["joint"]:
            best_bundle = p2_bundle
            best_stage_dir = p2_dir
    if "P5" in args.stages and best_bundle is not None:
        _run_p5(output_dir, best_stage_dir, best_bundle)
    _print("Tail-adapter foreground supervisor completed.")


if __name__ == "__main__":
    main()
