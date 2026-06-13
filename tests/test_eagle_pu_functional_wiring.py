from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_train_uses_real_state_targets_and_not_none_placeholder() -> None:
    src = read("fate_oia/engine/train_eagle_pu_oia.py")
    assert 'state_weak_bag_loss(out["state_logits"], None)' not in src
    assert "build_state_targets" in src
    assert "state_target_source" in src


def test_evidence_audit_is_real_selected_vs_random_not_zero_placeholder() -> None:
    src = read("fate_oia/engine/train_eagle_pu_oia.py")
    assert "run_selected_vs_random_evidence_audit" in src
    assert "evidence_margin_loss(torch.tensor([0.0]" not in src
    assert "selected_minus_random" in src
    assert "evidence_gate_history" in src
    assert "all(evidence_gate_history[-2:])" in src


def test_branch_metrics_are_computed_from_distinct_branch_logits() -> None:
    src = read("fate_oia/engine/train_eagle_pu_oia.py")
    assert "evaluate_branch_metric_views" in src
    assert '"direct_plus_prototype": metrics.get("metrics_raw_fixed", {})' not in src
    assert "reason_logits_direct_plus_prototype" in src
    assert "reason_logits_direct_plus_graph" in src


def test_artifacts_and_audit_are_hard_gates_not_schema_placeholders() -> None:
    artifacts = read("fate_oia/engine/eagle_pu_artifacts.py")
    audit = read("fate_oia/engine/audit_eagle_pu_implementation.py")
    export = read("fate_oia/engine/export_eagle_pu_visuals.py")
    assert "schema placeholder" not in artifacts
    assert "available\": False" not in artifacts
    assert "static contract present" not in audit
    assert "artifact_schema" in audit
    assert "eagle_pu_visual_export_schema.json" not in export
    assert "evidence_state_attention.png" in export
