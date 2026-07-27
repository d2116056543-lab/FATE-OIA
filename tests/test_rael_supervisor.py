"""P21 RED contracts for the real RAEL foreground launch path."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types

import pytest


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1] if HERE.parent.name == "tests" else HERE.parents[2]
STAGING = ROOT / "remote_patch" / "P21"
SUPERVISOR = STAGING / "supervise_acpr_rael_oia_foreground.py" if STAGING.is_dir() else ROOT / "fate_oia" / "engine" / "supervise_acpr_rael_oia_foreground.py"
SCRIPT = STAGING / "FATE_OIA_acpr_rael_oia_v1_foreground.ps1" if STAGING.is_dir() else ROOT / "scripts" / "FATE_OIA_acpr_rael_oia_v1_foreground.ps1"


def _module():
    spec = importlib.util.spec_from_file_location("rael_p21_supervisor", SUPERVISOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_supervisor_exposes_real_yaml_launch_and_runtime_factory() -> None:
    module = _module()
    for name in (
        "load_rael_config",
        "build_rael_runtime",
        "build_runtime_runner",
        "run_rael_mode",
        "main",
    ):
        assert callable(getattr(module, name, None)), name

    source = SUPERVISOR.read_text(encoding="utf-8")
    assert "BDDOIAMultiTaskDataset" in source
    assert "RAELGroundingTransform" in source
    assert "DataLoader" in source
    assert "RAELOIAModel" in source
    assert "RAELTrainer" in source
    assert "RAELTaskAwareBDD100KIndex" in source


def test_runner_factory_is_not_a_synthetic_or_placeholder_path() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "build_runtime_runner" in names
    forbidden = ("Synthetic", "FakeRunner", "random batch", "torch.randn")
    assert not any(token in source for token in forbidden)
    assert "prepare_counterfactual_handoff" in source
    assert "replay_counterfactual_from_encoded_field" in source


def test_foreground_script_calls_supervisor_without_background_mechanisms() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    assert "fate_oia.engine.supervise_acpr_rael_oia_foreground" in script
    for forbidden in ("Start-Process", "Start-Job", "Register-ScheduledTask", "-WindowStyle Hidden", "nohup", "daemon"):
        assert forbidden not in script
    assert "& $Python" in script
    assert "RAEL_OIA_V1_FULL_TRAIN_READY.json" in script
    assert "--runtime_profile" in script


def test_mode_contract_rejects_full_without_current_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _module()
    gate = tmp_path / "FULL_TRAIN_READY.json"
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        module.require_mode_gate("full", gate, expected_git_head="a" * 40)


def test_mode_contract_rejects_gate_without_real_smoke_or_user_override(tmp_path: Path) -> None:
    module = _module()
    gate = tmp_path / "RAEL_OIA_V1_FULL_TRAIN_READY.json"
    base = {
        "pass": True,
        "git_head": "a" * 40,
        "unresolved": [],
        "smoke_result": {"passed": True},
        "pilot_override": {
            "pilot_protocol_override": True,
            "pilot_completed": False,
            "replacement": "minimal_real_smoke_only",
        },
    }
    gate.write_text(__import__("json").dumps(base), encoding="utf-8")
    module.require_mode_gate("full", gate, expected_git_head="a" * 40)
    base["smoke_result"]["passed"] = False
    gate.write_text(__import__("json").dumps(base), encoding="utf-8")
    with pytest.raises(RuntimeError):
        module.require_mode_gate("full", gate, expected_git_head="a" * 40)


def test_full_launch_routes_through_gate_check() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert "require_mode_gate(mode, full_gate" in source
    assert "--full_gate" in source


def test_runtime_profile_loader_consumes_p20_directory_artifacts(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "runtime_profile.json").write_text(
        __import__("json").dumps({"candidates": [{"name": "w4"}]}),
        encoding="utf-8",
    )
    (tmp_path / "selected_runtime_profile.json").write_text(
        __import__("json").dumps({"selected": {"name": "w4"}, "reason": "fastest stable"}),
        encoding="utf-8",
    )
    (tmp_path / "runtime_steps.jsonl").write_text(
        __import__("json").dumps({"elapsed": 1.0}) + "\n",
        encoding="utf-8",
    )
    profile, selected, steps = module._load_selected_runtime_profile(
        tmp_path,
        provenance={"schema_version": "x"},
    )
    assert profile["candidates"][0]["name"] == "w4"
    assert selected["selected"]["name"] == "w4"
    assert len(steps) == 1


def test_mechanism_row_uses_observed_dino_count() -> None:
    module = _module()
    result = types.SimpleNamespace(
        optimizer_step=1,
        components={},
        owner_parameter_delta={"x": 0.1},
        owner_gradient_norms_pre_clip={"x": 0.2},
        mechanism_observation={"dino_call_count": 2},
    )
    row = module._mechanism_row(result)
    assert row["dino_call_count"] == 2


def test_smoke_mechanism_row_surfaces_component_activation_diagnostics() -> None:
    module = _module()
    result = types.SimpleNamespace(
        optimizer_step=3,
        components={"total": __import__("torch").tensor(1.0)},
        owner_parameter_delta={"unary": 0.1},
        owner_gradient_norms_pre_clip={"unary": 0.2},
        mechanism_observation={
            "dino_call_count": 1,
            "rho_nonzero_rate": 0.4,
            "q_view_bootstrap_count": 7,
            "action_unary_rms_over_global": 0.01,
            "pu_active_label_count": 0.0,
        },
    )
    row = module._mechanism_row(result)
    assert row["mechanism"] == {
        "action_unary_rms_over_global": 0.01,
        "q_view_bootstrap_count": 7,
        "rho_nonzero_rate": 0.4,
        "pu_active_label_count": 0.0,
    }


def test_windows_loader_policy_caps_workers_and_keeps_aux_loaders_single_process() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert "train_worker_count = min(int(num_workers), 4)" in source
    assert "aux_loader_kwargs" in source


def test_supervisor_requires_explicit_grounding_sources_and_rejects_incomplete_full_contract() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert "grounding_sources must explicitly provide the audited train/val" in source
    assert "_assert_full_contract_available" in source
    assert "public read-only field replay" in source


def test_audited_host_label_layout_requires_explicit_direct_train_and_val_json(tmp_path: Path) -> None:
    module = _module()
    train = tmp_path / "labels" / "100k" / "train"
    val = tmp_path / "labels" / "100k" / "val"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    (train / "a.json").write_text("{}", encoding="utf-8")
    (val / "b.json").write_text("{}", encoding="utf-8")
    resolved = module._audited_bdd100k_layout(
        {"label_directories": {"train": str(train), "val": str(val)}}
    )
    assert resolved == {"train": train, "val": val}
    with pytest.raises((FileNotFoundError, ValueError)):
        module._audited_bdd100k_layout({"label_directories": {"train": str(train)}})


def test_grounding_index_reads_data_scoped_explicit_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    train, val = tmp_path / "train", tmp_path / "val"
    train.mkdir(); val.mkdir()
    (train / "a.json").write_text("{}", encoding="utf-8")
    (val / "b.json").write_text("{}", encoding="utf-8")
    package = types.ModuleType("fate_oia")
    datasets = types.ModuleType("fate_oia.datasets")
    index_module = types.ModuleType("fate_oia.datasets.bdd100k_task_aware_index")
    class Index:
        def __init__(self, *, label_directories, include_file_names=None):
            self.label_directories = label_directories
            self.include_file_names = include_file_names
    index_module.RAELTaskAwareBDD100KIndex = Index
    monkeypatch.setitem(sys.modules, "fate_oia", package)
    monkeypatch.setitem(sys.modules, "fate_oia.datasets", datasets)
    monkeypatch.setitem(sys.modules, index_module.__name__, index_module)
    index = module._grounding_index({"grounding_sources": {"label_directories": {"train": str(train), "val": str(val)}}})
    assert index.label_directories == {"train": train, "val": val}


def test_labelwise_pu_audit_requires_hidden_positive_recovery_over_visual_baseline() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    # Forty known positives and forty observed-zero rows per label expose a
    # deterministic 30% hidden-positive recovery task for every reason.
    targets = torch.cat((torch.ones(40, 21), torch.zeros(40, 21)), dim=0)
    rows = module.labelwise_pu_audit(
        known_targets=targets,
        visual_baseline_logits=torch.zeros_like(targets),
        recovered_scores=torch.zeros_like(targets),
        hidden_positive_fraction=0.30,
        minimum_positive_count=20,
        seed=7,
    )
    assert len(rows) == 21
    assert all(row["hidden_positive_count"] == 12 for row in rows)
    assert all(row["recovery_lcb95"] == 0.0 for row in rows)
    assert not any(row["eligible"] for row in rows)
