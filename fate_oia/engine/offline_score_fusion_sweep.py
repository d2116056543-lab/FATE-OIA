from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from fate_oia.engine.eval_score_calibrated import evaluate_score_logits


RUN_C_REFERENCE = {"joint": 0.547844, "Act_mF1": 0.714387, "Exp_mF1": 0.381301, "Exp_mAP": 0.367822}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_names(path: Path) -> list[str]:
    return [str(x) for x in json.loads(path.read_text(encoding="utf-8"))]


def _best_epoch_dir(run_dir: Path, split: str) -> Path:
    metrics = _read_jsonl(run_dir / "metrics_summary.jsonl")
    epoch = None
    if metrics:
        def key(row: dict[str, Any]) -> tuple[float, float, int]:
            return (
                float(row.get(f"{split}_joint", row.get("test_joint", row.get("joint", float("-inf"))))),
                float(row.get(f"{split}_Exp_mF1", row.get("test_Exp_mF1", row.get("Exp_mF1", float("-inf"))))),
                int(row.get("epoch", -1)),
            )

        epoch = int(max(metrics, key=key).get("epoch", -1))
    if epoch is None or epoch < 0:
        epoch_dirs = sorted([p for p in run_dir.glob("epoch_*") if p.is_dir()])
        if epoch_dirs:
            return epoch_dirs[-1]
        return run_dir
    candidate = run_dir / f"epoch_{epoch:03d}"
    return candidate if candidate.exists() else run_dir


def _find_file(base: Path, candidates: list[str]) -> Path:
    for name in candidates:
        path = base / name
        if path.exists():
            return path
    if base.name.startswith("epoch_"):
        root = base.parent
        for name in candidates:
            path = root / name
            if path.exists():
                return path
    raise FileNotFoundError(f"None of the candidate files exist under {base}: {candidates}")


def _load_branch(run_dir: str | Path, *, split: str, prefer_fused_action: bool) -> dict[str, Any]:
    root = Path(run_dir)
    base = _best_epoch_dir(root, split)
    if prefer_fused_action:
        action_candidates = [
            f"logits_action_fused_{split}.pt",
            f"logits_action_{split}.pt",
            f"logits_action_fused_best_{split}.pt",
            f"logits_action_best_{split}.pt",
        ]
    else:
        action_candidates = [
            f"logits_action_{split}.pt",
            f"logits_action_fused_{split}.pt",
            f"logits_action_best_{split}.pt",
            f"logits_action_fused_best_{split}.pt",
        ]
    paths = {
        "action_logits": _find_file(base, action_candidates),
        "reason_logits": _find_file(base, [f"logits_reason_{split}.pt", f"logits_reason_best_{split}.pt"]),
        "action_labels": _find_file(base, [f"labels_action_{split}.pt", f"labels_action_best_{split}.pt"]),
        "reason_labels": _find_file(base, [f"labels_reason_{split}.pt", f"labels_reason_best_{split}.pt"]),
        "file_names": _find_file(base, [f"file_names_{split}.json", f"file_names_best_{split}.json"]),
    }
    return {
        "root": str(root),
        "epoch_dir": str(base),
        "paths": {k: str(v) for k, v in paths.items()},
        "action_logits": torch.load(paths["action_logits"], map_location="cpu").float(),
        "reason_logits": torch.load(paths["reason_logits"], map_location="cpu").float(),
        "action_labels": torch.load(paths["action_labels"], map_location="cpu").float(),
        "reason_labels": torch.load(paths["reason_labels"], map_location="cpu").float(),
        "file_names": _load_names(paths["file_names"]),
    }


def _reorder_to_reference(branch: dict[str, Any], reference_names: list[str]) -> tuple[dict[str, Any], bool]:
    names = list(branch["file_names"])
    if names == reference_names:
        return branch, False
    if len(names) != len(set(names)):
        raise ValueError("Cannot align branch with duplicate file names.")
    if set(names) != set(reference_names):
        missing = sorted(set(reference_names) - set(names))[:10]
        extra = sorted(set(names) - set(reference_names))[:10]
        raise ValueError(f"Cannot align branches; missing={missing}, extra={extra}")
    index_by_name = {name: idx for idx, name in enumerate(names)}
    order = torch.tensor([index_by_name[name] for name in reference_names], dtype=torch.long)
    aligned = dict(branch)
    for key in ("action_logits", "reason_logits", "action_labels", "reason_labels"):
        aligned[key] = branch[key][order]
    aligned["file_names"] = list(reference_names)
    return aligned, True


def _metric_row(name: str, result: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = result["metrics"]
    row = {
        "name": name,
        "joint": float(result["joint"]),
        "Act_mF1": float(metrics["Act_mF1"]),
        "Act_oF1": float(metrics["Act_oF1"]),
        "Exp_mF1": float(metrics["Exp_mF1"]),
        "Exp_oF1": float(metrics["Exp_oF1"]),
        "Exp_mAP": float(metrics["Exp_mAP"]),
    }
    if extra:
        row.update(extra)
    return row


def _evaluate_modes(
    *,
    name: str,
    action_logits: torch.Tensor,
    reason_logits: torch.Tensor,
    action_labels: torch.Tensor,
    reason_labels: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    return {
        mode: _metric_row(
            f"{name}_{mode}",
            evaluate_score_logits(action_logits, reason_logits, action_labels, reason_labels, threshold_mode=mode),
        )
        for mode in ("fixed", "global", "per_label")
    }


def run_phase_b_fusion_sweep(
    *,
    run_c_dir: str | Path,
    s1_dir: str | Path,
    output_dir: str | Path,
    split: str = "test",
    action_dim: int = 4,
    reason_dim: int = 21,
    step: float = 0.05,
) -> dict[str, Any]:
    """Run the plan-required Phase B offline sweep over Run C and ScoreV2 logits.

    The main sweep keeps Run C action logits fixed and mixes only reason logits.
    A secondary all-logit sweep is also recorded for diagnosis, but the decision
    flag is based on whether either mixed branch improves raw ranking/F1 beyond
    Run C by a meaningful margin.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_c = _load_branch(run_c_dir, split=split, prefer_fused_action=True)
    s1 = _load_branch(s1_dir, split=split, prefer_fused_action=False)
    s1, reordered = _reorder_to_reference(s1, run_c["file_names"])
    if run_c["action_logits"].shape[1] != action_dim or run_c["reason_logits"].shape[1] != reason_dim:
        raise ValueError(f"Run C shape mismatch: action={run_c['action_logits'].shape}, reason={run_c['reason_logits'].shape}")
    if not torch.equal(run_c["action_labels"], s1["action_labels"]) or not torch.equal(run_c["reason_labels"], s1["reason_labels"]):
        raise ValueError("Aligned labels differ between Run C and S1 outputs.")

    action_labels = run_c["action_labels"]
    reason_labels = run_c["reason_labels"]
    thresholds = {
        "run_c": _evaluate_modes(name="run_c", action_logits=run_c["action_logits"], reason_logits=run_c["reason_logits"], action_labels=action_labels, reason_labels=reason_labels),
        "score_v2": _evaluate_modes(name="score_v2", action_logits=s1["action_logits"], reason_logits=s1["reason_logits"], action_labels=action_labels, reason_labels=reason_labels),
    }

    rows: list[dict[str, Any]] = []
    steps = int(round(1.0 / step))
    for idx in range(steps + 1):
        lambda_s1 = round(idx * step, 6)
        mixed_reason = (1.0 - lambda_s1) * run_c["reason_logits"] + lambda_s1 * s1["reason_logits"]
        reason_only = evaluate_score_logits(run_c["action_logits"], mixed_reason, action_labels, reason_labels, threshold_mode="fixed")
        rows.append(
            _metric_row(
                "reason_mix_fixed",
                reason_only,
                {"lambda_s1": lambda_s1, "mix_target": "reason_only", "threshold_mode": "fixed"},
            )
        )
        mixed_action = (1.0 - lambda_s1) * run_c["action_logits"] + lambda_s1 * s1["action_logits"]
        all_logits = evaluate_score_logits(mixed_action, mixed_reason, action_labels, reason_labels, threshold_mode="fixed")
        rows.append(
            _metric_row(
                "action_reason_mix_fixed",
                all_logits,
                {"lambda_s1": lambda_s1, "mix_target": "action_reason", "threshold_mode": "fixed"},
            )
        )

    run_c_fixed = thresholds["run_c"]["fixed"]
    s1_fixed = thresholds["score_v2"]["fixed"]
    best_fusion = max(rows, key=lambda row: (row["joint"], row["Exp_mF1"], row["Exp_mAP"]))
    best_reason_mix = max([row for row in rows if row["mix_target"] == "reason_only"], key=lambda row: (row["joint"], row["Exp_mF1"], row["Exp_mAP"]))
    best_threshold_row = max(
        [row for branch in thresholds.values() for row in branch.values()],
        key=lambda row: (row["Exp_mF1"], row["joint"], row["Exp_mAP"]),
    )
    exp_gain = best_fusion["Exp_mF1"] - run_c_fixed["Exp_mF1"]
    ap_gain = best_fusion["Exp_mAP"] - run_c_fixed["Exp_mAP"]
    logit_fusion_promising = bool(exp_gain > 0.005 or ap_gain > 0.005)
    best_fixed = max([thresholds["run_c"]["fixed"], thresholds["score_v2"]["fixed"]], key=lambda row: (row["Exp_mF1"], row["joint"]))
    calibration_only = bool((best_threshold_row["Exp_mF1"] - best_fixed["Exp_mF1"]) > 0.005 and not logit_fusion_promising)
    decision = {
        "split": split,
        "run_c_reference": RUN_C_REFERENCE,
        "logit_fusion_promising": logit_fusion_promising,
        "calibration_only": calibration_only,
        "fusion_exp_mf1_gain_vs_run_c_fixed": exp_gain,
        "fusion_exp_map_gain_vs_run_c_fixed": ap_gain,
        "recommendation": (
            "train_or_report_logit_fusion_adapter"
            if logit_fusion_promising
            else ("report_calibrated_threshold_diagnostic" if calibration_only else "skip_scorev2_followup_training")
        ),
    }
    payload = {
        "run_c_fixed": run_c_fixed,
        "score_v2_fixed": s1_fixed,
        "best_fusion": best_fusion,
        "best_reason_mix": best_reason_mix,
        "rows": rows,
        "threshold_modes": thresholds,
        "alignment": {"s1_reordered_to_run_c": reordered, "sample_count": len(run_c["file_names"])},
        "inputs": {"run_c": {"epoch_dir": run_c["epoch_dir"], "paths": run_c["paths"]}, "score_v2": {"epoch_dir": s1["epoch_dir"], "paths": s1["paths"]}},
        "decision": decision,
    }
    _write_json(output / f"fusion_sweep_{split}.json", payload)
    _write_json(output / f"threshold_sweep_{split}.json", thresholds)
    _write_json(output / "phase_b_decision.json", decision)
    _write_json(output / "score_fusion_sweep.json", payload)
    return payload


def run_score_fusion_sweep(
    *,
    action_logits_a: torch.Tensor,
    reason_logits_a: torch.Tensor,
    action_logits_b: torch.Tensor,
    reason_logits_b: torch.Tensor,
    action_labels: torch.Tensor,
    reason_labels: torch.Tensor,
    output_dir: str | Path,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha in [x / 10.0 for x in range(11)]:
        action = alpha * action_logits_a + (1.0 - alpha) * action_logits_b
        reason = alpha * reason_logits_a + (1.0 - alpha) * reason_logits_b
        result = evaluate_score_logits(action, reason, action_labels, reason_labels, threshold_mode="fixed")
        rows.append({"alpha_a": alpha, "joint": result["joint"], **result["metrics"]})
    best = max(rows, key=lambda row: (row["joint"], row["Exp_mF1"]))
    payload = {"rows": rows, "best": best}
    (output / "score_fusion_sweep.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline ScoreV2/Run C fusion diagnostics.")
    ap.add_argument("--run_c_dir")
    ap.add_argument("--s1_dir")
    ap.add_argument("--split", default="test")
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--action_logits_a")
    ap.add_argument("--reason_logits_a")
    ap.add_argument("--action_logits_b")
    ap.add_argument("--reason_logits_b")
    ap.add_argument("--action_labels")
    ap.add_argument("--reason_labels")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    if args.run_c_dir and args.s1_dir:
        result = run_phase_b_fusion_sweep(
            run_c_dir=args.run_c_dir,
            s1_dir=args.s1_dir,
            output_dir=args.output_dir,
            split=args.split,
            action_dim=args.action_dim,
            reason_dim=args.reason_dim,
        )
        print(json.dumps(result["decision"], ensure_ascii=False, indent=2))
        return
    required = [args.action_logits_a, args.reason_logits_a, args.action_logits_b, args.reason_logits_b, args.action_labels, args.reason_labels]
    if any(x is None for x in required):
        raise SystemExit("Either pass --run_c_dir/--s1_dir or all legacy logits/label paths.")
    result = run_score_fusion_sweep(
        action_logits_a=torch.load(args.action_logits_a, map_location="cpu"),
        reason_logits_a=torch.load(args.reason_logits_a, map_location="cpu"),
        action_logits_b=torch.load(args.action_logits_b, map_location="cpu"),
        reason_logits_b=torch.load(args.reason_logits_b, map_location="cpu"),
        action_labels=torch.load(args.action_labels, map_location="cpu"),
        reason_labels=torch.load(args.reason_labels, map_location="cpu"),
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
