from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from fate_oia.engine.audit_cafe_evidence_cache import audit_split, load_grounding_cache_jsonl
from fate_oia.engine.calibrate_cafe_oia import apply_calibration, fit_classwise_bias_temperature
from fate_oia.engine.train_cafe_oia import parse_args
from fate_oia.models.cafe_oia_model import CAFEOIAModel
from fate_oia.utils.cafe_artifacts import write_json
from fate_oia.utils.cafe_review_gates import FORBIDDEN_PLACEHOLDERS, scan_forbidden_tokens
from fate_oia.utils.plateau_rollback import PlateauRestore


def _git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def static_scan(repo: Path, output_dir: Path) -> dict[str, Any]:
    paths = [
        repo / "fate_oia" / "engine" / "train_cafe_oia.py",
        repo / "fate_oia" / "models" / "cafe_oia_model.py",
        repo / "fate_oia" / "models" / "causal_evidence_pooler.py",
        repo / "fate_oia" / "engine" / "calibrate_cafe_oia.py",
    ]
    hits = scan_forbidden_tokens(paths, FORBIDDEN_PLACEHOLDERS)
    write_json(output_dir / "forbidden_placeholder_scan.json", {"hits": hits})
    return {"passed": not hits, "hits": hits}


def config_audit(config: str, output_dir: Path) -> dict[str, Any]:
    args = parse_args(["--config", config, "--output_dir", str(output_dir / "config_dry_run")])
    resolved = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, type(None)))}
    write_json(output_dir / "config_dry_run.json", resolved)
    return {
        "passed": args.config_version == "cafe_oia_v2_evidence_fixed" and float(args.loss_direct_effect) == 0.03,
        "config_version": args.config_version,
        "loss_direct_effect": args.loss_direct_effect,
        "best_selection_split": args.best_selection_split,
    }


def evidence_audit(args, output_dir: Path) -> dict[str, Any]:
    train_args = parse_args(["--config", args.config, "--output_dir", str(output_dir / "evidence_dry_run")])
    cache = load_grounding_cache_jsonl(train_args.grounding_cache_jsonl)
    rows = {
        split: audit_split(split, train_args.data_root, train_args.raw_root, cache, max_samples=args.max_evidence_samples)
        for split in ("train", "val", "test")
    }
    for split, stats in rows.items():
        write_json(output_dir / f"evidence_audit_{split}.json", stats)
    merged = {"splits": rows}
    write_json(output_dir / "evidence_audit_real_split.json", merged)
    train = rows["train"]
    test = rows["test"]
    passed = train["key_hit_rate"] >= 0.70 and test["key_hit_rate"] >= 0.70 and train["object_box_rows"] > 0 and test["object_box_rows"] > 0
    return {"passed": bool(passed), "train": train, "test": test}


def counterfactual_smoke(output_dir: Path, device: str) -> dict[str, Any]:
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model = CAFEOIAModel(dim=32, action_dim=4, reason_dim=21, max_evidence_units_per_image=8).to(dev)
    tokens = torch.randn(2, 1 + 45 * 80, 32, device=dev)
    labels = torch.zeros(2, 21, device=dev)
    labels[:, 5] = 1.0
    batch = {"file_name": ["a.jpg", "b.jpg"]}
    cache = {
        "a.jpg": {"objects": [{"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 200, "y2": 200}}]},
        "b.jpg": {"objects": [{"category": "car", "box2d": {"x1": 20, "y1": 20, "x2": 220, "y2": 220}}]},
    }
    rules = {5: {"car"}}
    out = model(tokens, batch=batch, grounding_cache=cache, return_cf=True, cf_targets=labels, image_height=360, image_width=640, patch_size=8, reason_rules=rules)
    cf = out["cf"]
    drop = torch.sigmoid(cf["reason_logits_factual"]) - torch.sigmoid(cf["reason_logits_target_deleted"])
    valid = cf["cf_real_evidence_mask"]
    direct = float(drop[valid].mean().detach().cpu()) if bool(valid.any()) else 0.0
    result = {
        "cf_valid_count": int(cf["cf_valid_mask"].sum().detach().cpu()),
        "cf_real_evidence_count": int(cf["cf_real_evidence_mask"].sum().detach().cpu()),
        "cf_is_proxy": bool(cf["cf_is_proxy"]),
        "direct_effect_mean": direct,
        "target_deleted_drop_mean": direct,
    }
    write_json(output_dir / "counterfactual_smoke.json", result)
    return {"passed": result["cf_valid_count"] > 0 and result["cf_real_evidence_count"] > 0 and not result["cf_is_proxy"] and math.isfinite(direct), **result}


def calibration_smoke(output_dir: Path) -> dict[str, Any]:
    logits = torch.tensor([[-2.0, 0.1], [0.2, -1.0], [1.0, -0.2], [-0.5, 2.0]])
    labels = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    params = fit_classwise_bias_temperature(logits, labels)
    cal = apply_calibration(logits, params)
    result = {"params": params, "changed": bool((cal - logits).abs().sum().item() > 0)}
    write_json(output_dir / "calibration_smoke.json", result)
    return {"passed": result["changed"] and len(params["bias"]) == 2 and len(params["temperature"]) == 2, **result}


def plateau_smoke(output_dir: Path) -> dict[str, Any]:
    model = torch.nn.Linear(1, 1, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    restore = PlateauRestore(patience=0, factor=0.5, min_lr=0.01, max_restores=2)
    with torch.no_grad():
        model.weight.fill_(2.0)
    restore.step(1.0, 0, model, opt, None, output_dir)
    with torch.no_grad():
        model.weight.fill_(0.0)
    event = restore.step(0.0, 1, model, opt, None, output_dir)
    weight = float(model.weight.detach().cpu().item())
    result = {"restored": event["restored"], "weight": weight, "lr": opt.param_groups[0]["lr"]}
    write_json(output_dir / "plateau_smoke.json", result)
    return {"passed": event["restored"] and abs(weight - 2.0) < 1e-6 and opt.param_groups[0]["lr"] < 0.1, **result}


def artifact_smoke(config: str, output_dir: Path, device: str) -> dict[str, Any]:
    smoke_dir = output_dir / "artifact_smoke"
    cmd = [
        sys.executable,
        "-m",
        "fate_oia.engine.train_cafe_oia",
        "--config",
        config,
        "--output_dir",
        str(smoke_dir),
        "--epochs",
        "1",
        "--batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--num_workers",
        "0",
        "--max_train_samples",
        "4",
        "--max_val_samples",
        "4",
        "--max_test_samples",
        "4",
        "--device",
        device,
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output_dir / "artifact_smoke_output.log").write_text(proc.stdout, encoding="utf-8")
    required = [
        smoke_dir / "run_manifest.json",
        smoke_dir / "checkpoint_latest.pth",
        smoke_dir / "checkpoint_best_test.pth",
        smoke_dir / "epoch_000" / "metrics_summary.json",
        smoke_dir / "epoch_000" / "evidence_stats.jsonl",
        smoke_dir / "epoch_000" / "counterfactual_stats.jsonl",
        smoke_dir / "epoch_000" / "calibration_params_test_diagnostic.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    return {"passed": proc.returncode == 0 and not missing, "returncode": proc.returncode, "missing": missing, "log": str(output_dir / "artifact_smoke_output.log")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_cafe_oia_v2.yaml")
    ap.add_argument("--output_dir", default=".background_runs/cafe_oia_v2_preflight")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_evidence_samples", type=int, default=512)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path.cwd()
    checks: dict[str, Any] = {}
    failures: list[str] = []
    for name, fn in [
        ("static_scan", lambda: static_scan(repo, out)),
        ("config", lambda: config_audit(args.config, out)),
        ("evidence", lambda: evidence_audit(args, out)),
        ("counterfactual", lambda: counterfactual_smoke(out, args.device)),
        ("calibration", lambda: calibration_smoke(out)),
        ("plateau_restore", lambda: plateau_smoke(out)),
        ("artifact_schema", lambda: artifact_smoke(args.config, out, args.device)),
    ]:
        try:
            checks[name] = fn()
        except Exception as exc:
            checks[name] = {"passed": False, "error": repr(exc)}
        if not checks[name].get("passed"):
            failures.append(name)
    report = {
        "passed": not failures,
        "git_head": _git_text(["rev-parse", "HEAD"]),
        "dirty_status": _git_text(["status", "--short"]),
        "checks": checks,
        "failures": failures,
        "required_next_action": "fix failed checks before training" if failures else "training permitted by executable V2 review gate",
    }
    write_json(out / "review_report.json", report)
    if not failures:
        (out / "REVIEW_PASS_CAFE_V2.txt").write_text(
            "\n".join(
                [
                    f"commit={report['git_head']}",
                    "py_compile=required before formal launch",
                    "targeted_tests=required before formal launch",
                    f"evidence_audit={out / 'evidence_audit_real_split.json'}",
                    f"counterfactual_smoke={out / 'counterfactual_smoke.json'}",
                    f"calibration_smoke={out / 'calibration_smoke.json'}",
                    f"no_placeholder_scan={out / 'forbidden_placeholder_scan.json'}",
                ]
            ),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
