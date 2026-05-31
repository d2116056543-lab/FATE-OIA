from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from fate_oia.engine.train_trace_oia import parse_args
from fate_oia.losses.action_primary_trace_losses import compute_action_primary_trace_loss
from fate_oia.models.trace_oia_model import TraceOIAModel
from fate_oia.utils.action_candidate_selector import select_action_candidate
from fate_oia.utils.action_primary_conflict_gate import ActionPrimaryConflictGate
from fate_oia.utils.config_io import load_yaml_config
from fate_oia.utils.trace_optimizer_groups import build_action_primary_trace_optimizer


FORBIDDEN_ACTIVE_STRINGS = [
    "Start-Process",
    "Start-Job",
    "Win32_Process",
    "Invoke-WmiMethod",
    "nohup",
    "detached",
    "hidden",
    "checkpoint_best_val",
    "metrics_val",
    "best_val",
]


def _check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _scan_active_files(failures: list[str]) -> None:
    for path in [
        Path("fate_oia/engine/supervise_trace_oia_foreground.py"),
        Path("scripts/FATE_OIA_trace_oia_v1_foreground.ps1"),
    ]:
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_ACTIVE_STRINGS:
            if token in text:
                failures.append(f"Forbidden active-path token {token!r} in {path}")


def _tiny_smoke(out_dir: Path, failures: list[str]) -> None:
    cmd = [
        sys.executable,
        "-m",
        "fate_oia.engine.train_trace_oia",
        "--config",
        "configs/fate_oia_train_360x640_trace_action_primary_v2.yaml",
        "--output_dir",
        str(out_dir),
        "--epochs",
        "1",
        "--batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--device",
        "cuda",
        "--max_train_samples",
        "2",
        "--max_test_samples",
        "2",
        "--no-feature_cache_enabled",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out_dir / "smoke_stdout.log").write_text(proc.stdout, encoding="utf-8")
    _check(proc.returncode == 0, f"tiny smoke failed with code {proc.returncode}", failures)
    metrics = out_dir / "epoch_000" / "metrics_summary.json"
    _check(metrics.exists(), "tiny smoke missing metrics_summary.json", failures)
    if metrics.exists():
        data = json.loads(metrics.read_text(encoding="utf-8"))
        _check(data.get("selected_action_mode") is not None, "tiny smoke missing selected_action_mode", failures)
        _check(data.get("split") == "test", "tiny smoke did not evaluate test split", failures)


def main() -> None:
    failures: list[str] = []
    cfg_path = Path("configs/fate_oia_train_360x640_trace_action_primary_v2.yaml")
    cfg = load_yaml_config(str(cfg_path))
    _check(cfg.get("config_version") == "trace_oia_action_primary_v2_direct_image", "wrong config_version", failures)
    _check(cfg.get("feature_cache", {}).get("enabled") is False, "feature cache must be disabled", failures)
    _check(cfg.get("evaluation", {}).get("splits") == ["test"], "evaluation must be test only", failures)
    _check(cfg.get("model", {}).get("action_final_mode") == "action_safe_selector", "action_safe_selector missing", failures)

    parsed = parse_args(["--config", str(cfg_path), "--output_dir", ".background_runs/audit_parse", "--no-feature_cache_enabled"])
    _check(parsed.feature_cache_enabled is False, "parse did not keep feature cache disabled", failures)
    _check(parsed.best_selection_split == "test", "parse did not keep test-only best split", failures)
    _check(parsed.config_data.get("evaluation", {}).get("splits") == ["test"], "parse did not keep test-only evaluation", failures)

    model = TraceOIAModel(dim=32, action_final_mode="action_safe_selector")
    tokens = torch.randn(2, 12, 32)
    out = model(tokens)
    _check("action_candidates" in out, "model missing action_candidates", failures)
    _check("action_bias" in dict(model.named_parameters()), "model missing action_bias parameter", failures)

    class _OptArgs:
        lr_action_head = 1e-4
        lr_reason_head = 1e-4
        lr_transport = 1e-4
        lr_label_corr = 1e-4
        lr_reason_alpha = 1e-4
        lr_action_bias = 1e-4
        weight_decay = 0.01

    opt, summary = build_action_primary_trace_optimizer(model, _OptArgs())
    _check(opt is not None, "optimizer not constructed", failures)
    _check("reason_alpha" in summary["param_to_group"], "optimizer summary missing reason_alpha", failures)
    _check("action_bias" in summary["param_to_group"], "optimizer summary missing action_bias", failures)

    labels = torch.cat([torch.randint(0, 2, (2, 4)).float(), torch.randint(0, 2, (2, 21)).float()], dim=1)
    out["model_for_gate"] = model
    out["args_for_gate"] = _OptArgs()
    _OptArgs.action_dim = 4
    loss, loss_info = compute_action_primary_trace_loss(_OptArgs(), out, labels, epoch=0)
    _check(torch.isfinite(loss), "loss is not finite", failures)
    _check("action_primary_total" in loss_info, "loss_info missing action_primary_total", failures)

    gate = ActionPrimaryConflictGate(conflict_threshold=-0.1, downscale_reason_min=0.25, downscale_evidence_min=0.25)
    p = torch.nn.Parameter(torch.tensor([1.0]))
    info = gate.compute((p * p).sum(), -(p * p).sum(), (p * 0.0).sum(), [p], epoch=0, latest_act_mf1=None)
    _check(info.get("applied_reason_scale", 1.0) < 1.0, "conflict gate did not downscale reason on negative gradient cosine", failures)

    selected = select_action_candidate(
        {
            "base": {"metrics": {"Act_mF1": 0.7, "Act_mAP": 0.8, "Exp_mF1": 0.3, "Exp_mAP": 0.4}, "test_action_primary_score": 0.58, "standard_joint": 0.50},
            "safe": {"metrics": {"Act_mF1": 0.71, "Act_mAP": 0.79, "Exp_mF1": 0.29, "Exp_mAP": 0.4}, "test_action_primary_score": 0.59, "standard_joint": 0.49},
        }
    )
    _check(selected["selected_action_mode"] == "safe", "candidate selector did not prioritize action mF1", failures)
    _scan_active_files(failures)

    out_dir = Path(".background_runs/trace_action_primary_v2_preflight/tiny_smoke")
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _tiny_smoke(out_dir, failures)

    pass_dir = Path(".background_runs/trace_action_primary_v2_preflight")
    pass_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "PASS" if not failures else "FAIL", "failures": failures}
    (pass_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures:
        print(json.dumps(report, indent=2))
        raise SystemExit(1)
    (pass_dir / "REVIEW_PASS_TRACE_ACTION_PRIMARY.txt").write_text(
        "TRACE ActionPrimary V2 implementation audit PASS\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
