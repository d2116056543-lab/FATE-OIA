from __future__ import annotations

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FORMAL_FILES = (
    ROOT / "fate_oia/models/save_oia_model.py",
    ROOT / "fate_oia/utils/save_contracts.py",
)
FORBIDDEN_IMPORTS = {
    "fate_oia.models.meter_oia_model",
    "fate_oia.models.acpr_pair_memory",
    "fate_oia.models.acpr_threshold_head",
    "fate_oia.models.acpr_calibration",
    "fate_oia.optim.heca_optimization",
}
FORBIDDEN_CONFIG_KEYS = {
    "pair_memory",
    "graph",
    "pmi",
    "threshold_head",
    "resume_checkpoint",
    "feature_cache_path",
    "token_compression_path",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            values.add(node.module)
    return values


def test_save_formal_sources_do_not_import_forbidden_paths() -> None:
    imports = set().union(*(_imports(path) for path in FORMAL_FILES))
    assert not imports & FORBIDDEN_IMPORTS


def test_save_config_disables_legacy_paths_without_dynamic_state_paths() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/fate_oia_train_360x640_save_oia_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["runtime"]["feature_cache_enabled"] is False
    assert config["runtime"]["no_feature_cache"] is True
    assert config["runtime"]["token_compression"] == "none"
    assert config["model"]["trainable_threshold"] is False
    assert config["model"]["use_pair_memory"] is False
    assert config["model"]["use_graph"] is False
    assert config["model"]["use_pmi"] is False
    assert not any(
        key in FORBIDDEN_CONFIG_KEYS
        for section in config.values()
        if isinstance(section, dict)
        for key in section
    )
    assert "V3" not in repr(config)


def test_save_factor_schema_has_exact_order_and_unknown_boundary() -> None:
    schema = yaml.safe_load(
        (ROOT / "configs/save_factor_schema.yaml").read_text(encoding="utf-8")
    )
    factors = schema["factors"]
    assert [int(row["id"]) for row in factors] == list(range(21))
    assert schema["unknown_is_negative"] is False
    assert {str(row["groundability"]).lower() for row in factors} >= {
        "full",
        "partial",
        "latent",
    }
