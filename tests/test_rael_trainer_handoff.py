"""P21 RED contracts for P17's public field handoff and epoch publisher."""

from __future__ import annotations

import ast
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1] if HERE.parent.name == "tests" else HERE.parents[2]
STAGING = ROOT / "remote_patch" / "P21"
TRAINER = STAGING / "train_acpr_rael_oia.py" if STAGING.is_dir() else ROOT / "fate_oia" / "engine" / "train_acpr_rael_oia.py"


def _class_methods() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    trainer = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RAELTrainer"
    )
    return {node.name: node for node in trainer.body if isinstance(node, ast.FunctionDef)}


def test_p21_exposes_public_read_only_encoded_field_handoff() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    methods = _class_methods()
    assert "RAELEncodedFieldHandoff" in source
    assert "prepare_counterfactual_handoff" in methods
    assert "replay_counterfactual_from_encoded_field" in methods
    replay = ast.get_source_segment(source, methods["replay_counterfactual_from_encoded_field"])
    assert replay is not None
    assert "encode_images(" not in replay
    assert "run_feature_intervention" in replay


def test_p21_epoch_method_requires_real_p18_writer_and_artifact_builder() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    methods = _class_methods()
    assert "train_epoch_and_publish" in methods
    publisher = ast.get_source_segment(source, methods["train_epoch_and_publish"])
    assert publisher is not None
    assert "epoch_artifact_builder" in publisher
    assert "evaluate_rael_test_only" in source
    assert "RAELArtifactWriter" in source
    assert "writer.write_epoch" in publisher
