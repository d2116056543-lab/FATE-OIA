from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from fate_oia.losses.psr_train_losses import psr_train_loss
from fate_oia.models.psr_train_oia_model import PSRTrainOIAFeatureModel


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def check_config(config_path: Path) -> list[str]:
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    failures: list[str] = []
    if not bool(cfg.get("test_only_evaluation", False)):
        failures.append("config must set test_only_evaluation=true")
    if cfg.get("best_selection_split") != "test":
        failures.append("config must select best checkpoint by test")
    if bool(cfg.get("feature_cache_enabled", True)):
        failures.append("config must disable feature_cache_enabled")
    if bool(cfg.get("old_logits_training", True)):
        failures.append("config must set old_logits_training=false")
    if int(cfg.get("batch_size", 0)) * int(cfg.get("gradient_accumulation_steps", 0)) != 32:
        failures.append("effective batch must be 32 for default config")
    if not Path(str(cfg.get("pretrained_weights", ""))).exists():
        failures.append(f"pretrained_weights missing: {cfg.get('pretrained_weights')}")
    return failures


def check_source_static() -> list[str]:
    failures: list[str] = []
    train_src = _read("fate_oia/engine/train_psr_train_oia.py")
    supervisor_src = _read("fate_oia/engine/supervise_psr_train_oia_foreground.py")
    forbidden_train = ["resume_checkpoint", "registry_config", "logits_action_", "logits_reason_"]
    for token in forbidden_train:
        if token in train_src:
            failures.append(f"train_psr_train_oia.py contains forbidden old-artifact token: {token}")
    if "torch.load(" in train_src and "resume_psr_train_checkpoint" not in train_src:
        failures.append("torch.load is only allowed for same-output-dir resume_psr_train_checkpoint")
    for token in ["same_output_dir_psr_train_checkpoint_only", "old RunC/CARE checkpoints are not allowed"]:
        if token not in train_src:
            failures.append(f"safe same-run resume guard missing: {token}")
    for token in ["Start-Process", "Start-Job", "nohup", "-WindowStyle Hidden"]:
        if token.lower() in supervisor_src.lower():
            failures.append(f"supervisor contains forbidden background token: {token}")
    required = ["PSRTrainOIAFeatureModel", "psr_train_loss", "make_loader(args, \"test\"", "checkpoint_best_test.pth"]
    for token in required:
        if token not in train_src:
            failures.append(f"train source missing required token: {token}")
    return failures


def check_model_gradients() -> list[str]:
    failures: list[str] = []
    torch.manual_seed(7)
    model = PSRTrainOIAFeatureModel(dim=32, action_dim=4, reason_dim=21, num_heads=4, dropout=0.0)
    tokens = torch.randn(2, 33, 32)
    action = torch.randint(0, 2, (2, 4)).float()
    reason = torch.randint(0, 2, (2, 21)).float()
    args = argparse.Namespace(
        asl_gamma_pos=0.0,
        asl_gamma_neg=4.0,
        asl_clip=0.05,
        pareto_margin_action=0.005,
        pareto_margin_reason=0.005,
        loss_final_action=1.0,
        loss_final_reason=1.0,
        loss_a_action=0.4,
        loss_e_reason=0.4,
        loss_a_reason=0.05,
        loss_e_action=0.01,
        loss_calibration_reason=0.05,
        loss_pareto=0.2,
        loss_gradient_budget=0.001,
    )
    out = model(tokens, epoch=7)
    for key, shape in {
        "a_action_logits": (2, 4),
        "e_reason_logits": (2, 21),
        "final_action_logits": (2, 4),
        "final_reason_logits": (2, 21),
        "reason_router_gate": (2, 21),
        "action_router_gate": (2, 4),
    }.items():
        if tuple(out[key].shape) != shape:
            failures.append(f"{key} shape mismatch: {tuple(out[key].shape)} vs {shape}")
    loss, parts = psr_train_loss(out, action, reason, args)
    loss.backward()
    grad_checks = {
        "action_router": model.action_router[-1].weight.grad,
        "reason_router": model.reason_router[-1].weight.grad,
        "calibration_bias": model.reason_calibration_bias.grad,
    }
    for name, grad in grad_checks.items():
        if grad is None or float(grad.detach().abs().sum()) <= 0.0:
            failures.append(f"{name} has no gradient")
    if parts["pareto_action_loss"] < 0 or parts["pareto_reason_loss"] < 0:
        failures.append("pareto losses must be non-negative")
    warm = model(tokens, epoch=0)
    if not torch.allclose(warm["final_action_logits"], warm["a_action_logits"]):
        failures.append("warmup final_action must equal A_action")
    if not torch.allclose(warm["final_reason_logits"], warm["e_reason_logits"]):
        failures.append("warmup final_reason must equal E_reason")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_psr_train_oia_v1.yaml")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    failures.extend(check_config(ROOT / args.config))
    failures.extend(check_source_static())
    failures.extend(check_model_gradients())
    compile_files = [
        "fate_oia/models/psr_train_oia_model.py",
        "fate_oia/losses/psr_train_losses.py",
        "fate_oia/engine/train_psr_train_oia.py",
        "fate_oia/engine/audit_psr_train_oia_implementation.py",
        "fate_oia/engine/supervise_psr_train_oia_foreground.py",
    ]
    try:
        _run([sys.executable, "-m", "py_compile", *compile_files])
        _run([sys.executable, "-m", "pytest", "tests/test_psr_train_oia_model.py", "tests/test_psr_train_oia_audit.py", "-q"])
    except subprocess.CalledProcessError as exc:
        failures.append(f"compile_or_pytest_failed exit={exc.returncode}")
    report = {"status": "PASS" if not failures else "FAIL", "failures": failures}
    (out / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit(1)
    (out / "REVIEW_PASS_PSR_TRAIN_OIA_V1.txt").write_text("REVIEW_PASS_PSR_TRAIN_OIA_V1\n", encoding="utf-8")
    print("REVIEW_PASS_PSR_TRAIN_OIA_V1", flush=True)


if __name__ == "__main__":
    main()
