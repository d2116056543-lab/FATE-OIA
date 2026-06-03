from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from fate_oia.models.psr_oia_router import DynamicMarginEntropyRouter, LearnedPSRRouter, PSRFeatureBuilder, ParetoSafetySelector, StaticLabelRouter, evidence_reliability
from fate_oia.models.psr_specialist_registry import SpecialistRegistry, load_config, validate_alignment, LoadedSpecialistLogits
from fate_oia.utils.psr_artifacts import ensure_dir, write_json
from fate_oia.utils.psr_review_gates import assert_foreground_only


def _synthetic_loaded(name: str, files: list[str] | None = None, flip_labels: bool = False) -> LoadedSpecialistLogits:
    torch.manual_seed(0)
    labels_a = torch.randint(0, 2, (8, 4)).float()
    labels_r = torch.randint(0, 2, (8, 21)).float()
    if flip_labels:
        labels_a[0, 0] = 1 - labels_a[0, 0]
    return LoadedSpecialistLogits(
        name=name,
        role="synthetic",
        action_logits=torch.randn(8, 4),
        reason_logits=torch.randn(8, 21),
        labels_action=labels_a,
        labels_reason=labels_r,
        file_names=files or [f"{i}.jpg" for i in range(8)],
        source_dir=Path("."),
    )


def run_checks(registry_config: str, router_config: str, output_dir: Path) -> list[str]:
    checks: list[str] = []
    reg_cfg = load_config(registry_config)
    router_cfg = load_config(router_config)
    assert reg_cfg.get("config_version") == "psr_oia_v2_registry"
    assert router_cfg.get("config_version") == "psr_oia_v2_router"
    assert reg_cfg.get("test_only_evaluation") is True
    assert reg_cfg.get("feature_cache_enabled") is False
    assert router_cfg.get("no_feature_cache") is True
    assert router_cfg.get("protocol", {}).get("best_selection_split") == "test"
    checks.append("config")

    registry = SpecialistRegistry(registry_config)
    loaded, report = registry.aligned_available(output_dir)
    assert any(x.role == "action_specialists" for x in loaded)
    assert any(x.role == "explanation_specialists" for x in loaded)
    checks.append("candidate_discovery_alignment")

    a = _synthetic_loaded("a")
    b = _synthetic_loaded("b")
    validate_alignment(a, b)
    try:
        validate_alignment(a, _synthetic_loaded("bad_files", files=["x"] + a.file_names[1:]))
        raise AssertionError("file mismatch was not rejected")
    except ValueError:
        pass
    try:
        validate_alignment(a, _synthetic_loaded("bad_labels", flip_labels=True))
        raise AssertionError("label mismatch was not rejected")
    except ValueError:
        pass
    checks.append("synthetic_alignment_rejects_bad_inputs")

    action_a, reason_a = torch.randn(8, 4), torch.randn(8, 21)
    action_e, reason_e = torch.randn(8, 4), torch.randn(8, 21)
    static = StaticLabelRouter(["E"] * 21)(action_a, reason_a, action_e, reason_e)
    plain_avg_reason = 0.5 * reason_a + 0.5 * reason_e
    assert not torch.allclose(static.reason_logits, plain_avg_reason)
    dyn0 = DynamicMarginEntropyRouter()(action_a, reason_a, action_e, reason_e, evidence_rel=0.0)
    dyn1 = DynamicMarginEntropyRouter()(torch.zeros_like(action_a), reason_a, action_e, reason_e + 2.0, evidence_rel=1.0)
    assert not torch.allclose(dyn0.alpha_reason, dyn1.alpha_reason)
    assert dyn0.alpha_action.max().item() == 0.0
    checks.append("routers_not_plain_average")

    builder = PSRFeatureBuilder()
    af, rf = builder.build(action_a, reason_a, action_e, reason_e)
    learned = LearnedPSRRouter()
    out = learned(af, rf, action_a, reason_a, action_e, reason_e)
    loss = out.action_logits.mean() + out.reason_logits.mean()
    loss.backward()
    grad = sum(float(p.grad.abs().sum()) for p in learned.parameters() if p.grad is not None)
    assert grad > 0
    assert out.alpha_action.min() >= 0 and out.alpha_action.max() <= 1
    checks.append("learned_router_gradients")

    labels_a = torch.randint(0, 2, (8, 4)).float()
    labels_r = torch.randint(0, 2, (8, 21)).float()
    worse = -10.0 * (labels_a * 2 - 1)
    selected, guard = ParetoSafetySelector().guard_action(worse, action_a, reason_a, labels_a, labels_r)
    assert guard["pareto_action_fallback"] is True
    assert torch.allclose(selected, action_a)
    assert evidence_reliability(0.1, 0.2) == 0.0
    checks.append("pareto_and_evidence_gate")

    assert_foreground_only(["scripts/FATE_OIA_psr_oia_v2_goal.ps1", "fate_oia/engine/supervise_psr_oia_goal.py"])
    checks.append("foreground_only")
    py_files = [
        "fate_oia/models/psr_oia_router.py",
        "fate_oia/models/psr_label_groups.py",
        "fate_oia/models/psr_calibration.py",
        "fate_oia/models/psr_specialist_registry.py",
        "fate_oia/engine/collect_psr_logits.py",
        "fate_oia/engine/audit_psr_oia_implementation.py",
        "fate_oia/engine/eval_psr_oia.py",
        "fate_oia/engine/train_psr_oia_router.py",
        "fate_oia/engine/supervise_psr_oia_goal.py",
        "fate_oia/utils/psr_artifacts.py",
        "fate_oia/utils/psr_metrics.py",
        "fate_oia/utils/psr_sweeps.py",
        "fate_oia/utils/psr_review_gates.py",
    ]
    subprocess.run([sys.executable, "-m", "py_compile", *py_files], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_psr_specialist_registry.py",
            "tests/test_psr_alignment.py",
            "tests/test_psr_static_router.py",
            "tests/test_psr_dynamic_router.py",
            "tests/test_psr_learned_router.py",
            "tests/test_psr_pareto_safety.py",
            "tests/test_psr_audit_gates.py",
            "tests/test_psr_supervisor_foreground.py",
            "-q",
        ],
        check=True,
    )
    checks.append("py_compile_and_targeted_pytest")
    write_json(output_dir / "audit_checks.json", {"checks": checks, "loaded_candidates": report.get("loaded", [])})
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry_config", required=True)
    ap.add_argument("--router_config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = ensure_dir(args.output_dir)
    try:
        checks = run_checks(args.registry_config, args.router_config, out)
        (out / "REVIEW_PASS_PSR_OIA_V2.txt").write_text("\n".join(checks), encoding="utf-8")
        print("REVIEW_PASS_PSR_OIA_V2", checks)
    except Exception as exc:
        write_json(out / "REVIEW_BLOCKED_PSR_OIA_V2.json", {"error": str(exc)})
        raise


if __name__ == "__main__":
    main()
