"""P21 RED contracts for real evaluator-to-P18 epoch publication."""

from __future__ import annotations

import ast
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1] if HERE.parent.name == "tests" else HERE.parents[2]
STAGING = ROOT / "remote_patch" / "P21"
EVALUATOR = STAGING / "eval_acpr_rael_oia.py" if STAGING.is_dir() else ROOT / "fate_oia" / "engine" / "eval_acpr_rael_oia.py"
SUPERVISOR = STAGING / "supervise_acpr_rael_oia_foreground.py" if STAGING.is_dir() else ROOT / "fate_oia" / "engine" / "supervise_acpr_rael_oia_foreground.py"
TRAINER = STAGING / "train_acpr_rael_oia.py" if STAGING.is_dir() else ROOT / "fate_oia" / "engine" / "train_acpr_rael_oia.py"


def test_evaluator_exports_real_diagnostic_rows_from_the_same_test_decode() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    assert "_epoch_diagnostic_row" in source
    assert '"diagnostic_rows"' in source
    assert "field = model.encode_images(images.to(device))" in source
    assert source.count("model.encode_images(images.to(device))") == 1


def test_supervisor_has_real_p18_epoch_adapter_and_full_route() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "build_p18_epoch_artifacts" in names
    assert "fit_posthoc_calibration" in source
    assert "evaluate_rael_test_only" in source
    assert "RAELArtifactWriter" in source
    assert "train_epoch_and_publish" in source
    assert "checkpoint_latest.pth" in source


def test_full_epoch_publication_requires_real_counterfactual_audit_and_not_defaults() -> None:
    supervisor = SUPERVISOR.read_text(encoding="utf-8")
    trainer = TRAINER.read_text(encoding="utf-8")
    assert "writer.write_run_file" in supervisor
    assert "train_epoch_and_publish(" in supervisor
    assert "run_epoch_counterfactual_audit" in trainer
    assert "formal 128-case audit" in trainer
    assert "len(sample_ids) != 128" in supervisor
    assert '"soft_positive_count": 0.0' not in supervisor
    assert '"cosine": {}' not in supervisor
    assert 'float("nan")' not in supervisor


def test_no_cycle_loader_cache_and_full_consumes_each_epoch_once() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert "itertools.cycle" not in source
    assert "StopIteration" in source
    assert "for epoch in range" in source
    assert "train_epoch_and_publish" in source


def test_full_order_has_train_audit_pu_then_train_calib_then_test() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    full = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_rael_mode"
    )
    epoch_loop = next(
        node for node in ast.walk(full)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "epoch"
    )
    transition = next(
        node for node in epoch_loop.body
        if isinstance(node, ast.FunctionDef) and node.name == "epoch_transition"
    )
    transition_source = ast.get_source_segment(source, transition) or ""
    assert transition_source.index("run_fixed_train_audit_and_update_pu") < transition_source.index("fit_train_calib_calibration(runtime)")
    train_call = next(
        node for node in ast.walk(epoch_loop)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "train_epoch_and_publish"
    )
    assert any(keyword.arg == "epoch_transition" and isinstance(keyword.value, ast.Name) and keyword.value.id == "epoch_transition" for keyword in train_call.keywords)
    assert "train_audit_loader" in source
    assert "set_pu_label_gate" in source


def test_epoch_publication_is_streamed_and_does_not_retain_all_step_results() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    assert "step_results: list[RAELStepResult]" not in source
    assert "step_results.append" not in source
    assert "on_step_result" in source
    assert "last_step_result" in source


def test_grounding_sources_have_audited_layout_fallback_not_recursive_guessing() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert "_audited_bdd100k_layout" in source
    assert "rglob" not in source


def test_full_checkpoint_set_covers_latest_joint_action_reason_and_global_action() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    for name in (
        "checkpoint_latest.pth",
        "checkpoint_best_test_deploy_joint.pth",
        "checkpoint_best_test_action_mf1.pth",
        "checkpoint_best_test_exp_mf1.pth",
        "checkpoint_best_test_exp_map.pth",
        "checkpoint_best_test_global_action.pth",
    ):
        assert name in source


def test_each_epoch_runs_formal_counterfactual_audit_before_artifact_builder() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train_epoch_and_publish"
    )
    body = ast.get_source_segment(source, method) or ""
    assert body.index("run_epoch_counterfactual_audit") < body.index("epoch_artifact_builder(")
    assert "required_cases: int = 128" in source


def test_epoch_saves_pre_evaluation_state_before_counterfactual_or_test_eval() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    method = ast.get_source_segment(
        source,
        next(
            node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "train_epoch_and_publish"
        ),
    ) or ""
    assert "pre_evaluation_checkpoint" in method
    assert method.index("pre_evaluation_checkpoint(") < method.index("run_epoch_counterfactual_audit")
    supervisor = SUPERVISOR.read_text(encoding="utf-8")
    assert "checkpoint_pre_test_eval_epoch_" in supervisor
    assert "checkpoint_latest_pre_test_eval.pth" in supervisor


def test_epoch_counterfactual_audit_compares_canonical_case_identity() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    assert "case_id = canonicalize_sample_id(handoff.file_names[local_index])" in trainer
    assert 'if result.get("case_id") != case_id:' in trainer
    assert trainer.index('if result.get("available") is not True:') < trainer.index(
        'if result.get("case_id") != case_id:'
    )
