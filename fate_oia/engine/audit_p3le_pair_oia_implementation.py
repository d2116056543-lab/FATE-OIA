from __future__ import annotations

import argparse
import json
import py_compile
import subprocess
import sys
from pathlib import Path

import torch

from fate_oia.models.p3le_pair_oia_model import P3LEPairOIAFeatureModel
from fate_oia.utils.p3le_pair_artifacts import write_json
from fate_oia.utils.p3le_pair_review_gates import assert_foreground_script


ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = {}
    for value in data.values():
        if isinstance(value, dict):
            out.update(value)
    out.update({k: v for k, v in data.items() if not isinstance(v, dict)})
    return out


def check_config(config_path: Path) -> list[str]:
    cfg = load_config(config_path)
    failures = []
    if cfg.get("feature_cache", False) or cfg.get("feature_cache_enabled", False):
        failures.append("feature_cache must be false")
    if str(cfg.get("token_compression", "none")).lower() != "none":
        failures.append("token_compression must be none")
    if not bool(cfg.get("test_only_evaluation", True)):
        failures.append("test_only_evaluation must be true")
    if cfg.get("best_selection_split", "test") != "test":
        failures.append("best_selection_split must be test")
    if int(cfg.get("action_dim", 4)) != 4 or int(cfg.get("reason_dim", 21)) != 21:
        failures.append("action_dim/reason_dim must be 4/21")
    if int(cfg.get("batch_size", 4)) * int(cfg.get("gradient_accumulation_steps", 8)) != 32:
        failures.append("effective batch must be 32")
    pretrained = Path(str(cfg.get("pretrained_weights", "")))
    if not pretrained.exists():
        failures.append(f"pretrained_weights missing: {pretrained}")
    return failures


def read_rel(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def check_static() -> list[str]:
    failures = []
    required_files = [
        "fate_oia/models/p3le_pair_oia_model.py",
        "fate_oia/models/p3le_shared_encoder.py",
        "fate_oia/models/p3le_progressive_experts.py",
        "fate_oia/models/p3le_pair_head.py",
        "fate_oia/models/p3le_reason_reliability.py",
        "fate_oia/models/p3le_action_set_head.py",
        "fate_oia/models/p3le_router.py",
        "fate_oia/models/p3le_sparse_attention.py",
        "fate_oia/models/p3le_evidence_bag.py",
        "fate_oia/losses/p3le_pair_losses.py",
        "fate_oia/losses/cagrad_lite.py",
        "fate_oia/engine/train_p3le_pair_oia.py",
        "fate_oia/engine/supervise_p3le_pair_oia_foreground.py",
    ]
    for rel in required_files:
        if not (ROOT / rel).exists():
            failures.append(f"missing required file: {rel}")
    model_src = read_rel("fate_oia/models/p3le_pair_oia_model.py")
    expert_src = read_rel("fate_oia/models/p3le_progressive_experts.py")
    pair_src = read_rel("fate_oia/models/p3le_pair_head.py")
    loss_src = read_rel("fate_oia/losses/p3le_pair_losses.py")
    evidence_src = read_rel("fate_oia/models/p3le_evidence_bag.py")
    router_src = read_rel("fate_oia/models/p3le_router.py")
    train_src = read_rel("fate_oia/engine/train_p3le_pair_oia.py")
    for token in ["action_visual_logits", "reason_to_action_logits", "action_fused_logits", "reason_logits"]:
        if token not in read_rel("fate_oia/models/p3le_shared_encoder.py"):
            failures.append(f"base FATE strong action path token missing: {token}")
    for token in ["shared_1", "action_1", "reason_1", "shared_2", "action_2", "reason_2", "tail_2"]:
        if token not in expert_src:
            failures.append(f"PLE expert missing: {token}")
    if "pair_tensor" not in pair_src or "action_dim" not in pair_src or "reason_dim" not in pair_src:
        failures.append("pair tensor head implementation is incomplete")
    if "positive action x positive reason" not in pair_src:
        failures.append("pair seed code must document that it does not hard-label all positive action x reason pairs")
    if "reason_reliability" not in model_src or "q.detach()" not in loss_src:
        failures.append("q_r reliability must exist and weight reason supervision")
    if "never modifies action logits" not in evidence_src or "evidence_lambda_active" not in evidence_src:
        failures.append("evidence bag must be weak and selected<=random-gated")
    if "a_action +" not in router_src or "action_residual_cap" not in router_src:
        failures.append("Router_A must anchor on A_action with bounded residual")
    if "clip_shared_gradient_budget" not in train_src:
        failures.append("gradient-budget fallback must be wired in training")
    if "make_loader(args, \"test\"" not in train_src:
        failures.append("test-only loader must be wired")
    for forbidden in ["resume_checkpoint", "registry_config", "logits_action_", "logits_reason_", "feature_cache=True"]:
        if forbidden in train_src:
            failures.append(f"forbidden old-artifact/cache token in train script: {forbidden}")
    try:
        assert_foreground_script(ROOT / "scripts/FATE_OIA_p3le_pair_oia_v1_foreground.ps1")
        assert_foreground_script(ROOT / "fate_oia/engine/supervise_p3le_pair_oia_foreground.py")
    except AssertionError as exc:
        failures.append(str(exc))
    return failures


def check_model() -> list[str]:
    failures = []
    torch.manual_seed(7)
    model = P3LEPairOIAFeatureModel(dim=32, action_dim=4, reason_dim=21)
    tokens = torch.randn(2, 17, 32)
    action = torch.randint(0, 2, (2, 4)).float()
    reason = torch.randint(0, 2, (2, 21)).float()
    out = model(tokens, action, reason, epoch=12)
    if tuple(out["pair_tensor"].shape) != (2, 4, 21):
        failures.append(f"pair tensor wrong shape: {tuple(out['pair_tensor'].shape)}")
    if out["final_action_logits"].shape != action.shape:
        failures.append("final action shape mismatch")
    if out["final_reason_logits"].shape != reason.shape:
        failures.append("final reason shape mismatch")
    loss = out["final_action_logits"].mean() + out["final_reason_logits"].mean() + out["reason_reliability"].mean()
    loss.backward()
    router_grad = sum(float(p.grad.abs().sum()) for p in model.router.parameters() if p.grad is not None)
    if router_grad <= 0:
        failures.append("router gradients are zero")
    if float(out["evidence_lambda_active"]) not in (0.0, 1.0):
        failures.append("evidence active gate must be binary scalar")
    return failures


def run_command(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout


def check_compile_and_tests(output_dir: Path) -> list[str]:
    failures = []
    compile_files = [
        "fate_oia/models/p3le_pair_oia_model.py",
        "fate_oia/models/p3le_shared_encoder.py",
        "fate_oia/models/p3le_progressive_experts.py",
        "fate_oia/models/p3le_pair_head.py",
        "fate_oia/models/p3le_reason_reliability.py",
        "fate_oia/models/p3le_action_set_head.py",
        "fate_oia/models/p3le_router.py",
        "fate_oia/models/p3le_sparse_attention.py",
        "fate_oia/models/p3le_evidence_bag.py",
        "fate_oia/losses/p3le_pair_losses.py",
        "fate_oia/losses/cagrad_lite.py",
        "fate_oia/engine/train_p3le_pair_oia.py",
        "fate_oia/engine/audit_p3le_pair_oia_implementation.py",
        "fate_oia/engine/supervise_p3le_pair_oia_foreground.py",
        "fate_oia/engine/eval_p3le_pair_oia.py",
    ]
    for rel in compile_files:
        try:
            py_compile.compile(str(ROOT / rel), doraise=True)
        except Exception as exc:
            failures.append(f"py_compile failed for {rel}: {exc}")
    tests = [
        "tests/test_p3le_pair_shared_encoder.py",
        "tests/test_p3le_pair_experts.py",
        "tests/test_p3le_pair_head.py",
        "tests/test_p3le_reason_reliability.py",
        "tests/test_p3le_router.py",
        "tests/test_p3le_pair_losses.py",
        "tests/test_p3le_sparse_attention.py",
        "tests/test_p3le_pair_no_cache_testonly.py",
        "tests/test_p3le_pair_audit_gates.py",
        "tests/test_p3le_pair_supervisor_foreground.py",
    ]
    code, out = run_command([sys.executable, "-m", "pytest", *tests, "-q"], ROOT)
    (output_dir / "pytest_output.txt").write_text(out, encoding="utf-8")
    if code != 0:
        failures.append("targeted pytest failed")
    return failures


def run_smoke(config: Path, output_dir: Path, device: str) -> list[str]:
    failures = []
    smoke_dir = output_dir / "smoke"
    cmd = [
        sys.executable,
        "-m",
        "fate_oia.engine.train_p3le_pair_oia",
        "--config",
        str(config),
        "--output_dir",
        str(smoke_dir),
        "--device",
        device,
        "--epochs",
        "1",
        "--batch_size",
        "2",
        "--gradient_accumulation_steps",
        "2",
        "--num_workers",
        "0",
        "--max_train_samples",
        "16",
        "--max_test_samples",
        "16",
    ]
    code, out = run_command(cmd, ROOT)
    (output_dir / "smoke_output.txt").write_text(out, encoding="utf-8")
    if code != 0:
        failures.append("1-epoch smoke failed")
    required = [
        "run_manifest.json",
        "metrics_summary.jsonl",
        "epoch_000/branch_metrics.json",
        "epoch_000/pair_tensor_stats.json",
        "epoch_000/reason_reliability_stats.json",
        "epoch_000/evidence_bag_stats.json",
        "epoch_000/router_stats.json",
        "epoch_000/logits/action_final_test.pt",
        "epoch_000/logits/reason_final_test.pt",
    ]
    for rel in required:
        if not (smoke_dir / rel).exists():
            failures.append(f"smoke missing artifact: {rel}")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit P3LE-PAIR-OIA V1 implementation before training.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--canonical_read_confirmed", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--skip_smoke", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    if not args.canonical_read_confirmed:
        failures.append("canonical md files must be read before audit")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch in {"main", "fate-oia"}:
        failures.append(f"audit must run in isolated branch, got {branch}")
    failures.extend(check_config(ROOT / args.config))
    failures.extend(check_static())
    failures.extend(check_model())
    failures.extend(check_compile_and_tests(out_dir))
    if not args.skip_smoke:
        failures.extend(run_smoke(ROOT / args.config, out_dir, args.device))
    report = {"status": "fail" if failures else "pass", "failures": failures, "branch": branch}
    write_json(out_dir / "audit_report.json", report)
    write_json(out_dir / "module_checklist.json", {"checked": True, "failures": failures})
    write_json(out_dir / "artifact_schema_check.json", {"smoke_checked": not args.skip_smoke, "failures": failures})
    if failures:
        (out_dir / "REVIEW_FAIL_P3LE_PAIR_OIA_V1.txt").write_text("\n".join(failures), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    (out_dir / "REVIEW_PASS_P3LE_PAIR_OIA_V1.txt").write_text("REVIEW_PASS_P3LE_PAIR_OIA_V1", encoding="utf-8")
    print("REVIEW_PASS_P3LE_PAIR_OIA_V1")


if __name__ == "__main__":
    main()
