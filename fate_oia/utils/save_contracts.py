from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml


SAVE_SOURCE_HEAD = "b8669c4951f58d3c6b6831e0ae7fb5b2c7827db6"
SAVE_TARGET_BRANCH = "acpr_save_oia_v1_direct_image"
SAVE_TARGET_WORKTREE_SUFFIX = "fate_oia_acpr_save_oia_v1_worktree"
SAVE_LOCAL_MIRROR_SUFFIX = "fate_oia_acpr_save_oia_v1_sync_worktree"

L08_WRITE_SET = frozenset(
    {
        "configs/fate_oia_train_360x640_save_oia_v1.yaml",
        "configs/save_factor_schema.yaml",
        "fate_oia/models/save_oia_model.py",
        "fate_oia/utils/save_contracts.py",
        "fate_oia/models/meter_calalign_foundation.py",
        "tests/test_save_source_head_contract.py",
        "tests/test_save_worktree_contract.py",
        "tests/test_save_forbidden_paths.py",
        "tests/test_save_full_calalign_equivalence.py",
        "tests/test_save_uses_calalign_fused_action.py",
        "tests/test_save_one_dino_call.py",
        "tests/test_save_same_forward_branches.py",
        "tests/test_save_test_forward_image_only.py",
    }
)

SAVE_L08_WRITE_SET = L08_WRITE_SET

_FORBIDDEN_CONFIG_KEYS = frozenset(
    {
        "pair_memory",
        "graph",
        "pmi",
        "threshold_head",
        "resume_checkpoint",
        "feature_cache_path",
        "token_compression_path",
    }
)


def _repo_root(root: str | Path) -> Path:
    path = Path(root).resolve()
    if not path.exists() or not path.is_dir():
        raise ValueError(f"SAVE worktree does not exist: {path}")
    return path


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"SAVE git contract failed for {root}: {exc}") from exc
    return result.stdout.strip()


def validate_save_source_head(root: str | Path, *, head: str | None = None) -> bool:
    """Require the approved CalAlign ancestor to remain in the current DAG."""
    worktree = _repo_root(root)
    candidate = head or _git(worktree, "rev-parse", "HEAD")
    if len(candidate) != 40 or any(char not in "0123456789abcdef" for char in candidate.lower()):
        raise ValueError(f"invalid git head for SAVE source contract: {candidate!r}")
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", SAVE_SOURCE_HEAD, candidate],
            cwd=worktree,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValueError(f"unable to inspect SAVE source ancestry: {exc}") from exc
    if result.returncode != 0:
        raise ValueError(
            f"SAVE source {SAVE_SOURCE_HEAD} is not an ancestor of {candidate}"
        )
    return True


def validate_save_worktree(root: str | Path) -> bool:
    """Validate the dedicated L08 worktree without requiring a clean tree."""
    worktree = _repo_root(root)
    if worktree.name not in {
        SAVE_TARGET_WORKTREE_SUFFIX,
        SAVE_LOCAL_MIRROR_SUFFIX,
    }:
        raise ValueError(
            "SAVE L08 must run in the planned worktree or its local sync mirror: "
            f"{SAVE_TARGET_WORKTREE_SUFFIX!r}, {SAVE_LOCAL_MIRROR_SUFFIX!r}"
        )
    top_level = Path(_git(worktree, "rev-parse", "--show-toplevel")).resolve()
    if top_level != worktree:
        raise ValueError(f"git worktree root mismatch: {top_level} != {worktree}")
    branch = _git(worktree, "branch", "--show-current")
    if branch != SAVE_TARGET_BRANCH:
        raise ValueError(
            f"SAVE L08 must use branch {SAVE_TARGET_BRANCH!r}, got {branch!r}"
        )
    validate_save_source_head(worktree)
    return True


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def validate_save_config(config: Mapping[str, Any] | str | Path) -> bool:
    """Validate the central SAVE defaults and reject legacy side paths."""
    if isinstance(config, (str, Path)):
        data = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    else:
        data = dict(config)
    if not isinstance(data, Mapping):
        raise ValueError("SAVE config must be a mapping")
    model = data.get("model")
    runtime = data.get("runtime")
    if not isinstance(model, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("SAVE config requires model and runtime mappings")
    required_false = {
        "trainable_threshold": model.get("trainable_threshold"),
        "use_pair_memory": model.get("use_pair_memory"),
        "use_graph": model.get("use_graph"),
        "use_pmi": model.get("use_pmi"),
        "feature_cache_enabled": runtime.get("feature_cache_enabled"),
        "no_feature_cache": runtime.get("no_feature_cache"),
    }
    if any(
        value is not False
        for key, value in required_false.items()
        if key != "no_feature_cache"
    ) or required_false["no_feature_cache"] is not True:
        raise ValueError(f"SAVE forbidden runtime/model defaults are enabled: {required_false}")
    if runtime.get("token_compression") not in {None, "none"}:
        raise ValueError("SAVE token compression must be disabled")
    if model.get("selected_layers") not in ([3, 7, 11], (3, 7, 11)):
        raise ValueError("SAVE requires DINO layers [3, 7, 11]")
    if float(model.get("predicate_action_bridge_scale", 0.05)) != 0.05:
        raise ValueError("SAVE predicate action bridge must be fixed at 5%")
    forbidden = _walk_keys(data).intersection(_FORBIDDEN_CONFIG_KEYS)
    if forbidden:
        raise ValueError(f"SAVE config contains forbidden path keys: {sorted(forbidden)}")
    return True


def validate_save_factor_schema(schema: Mapping[str, Any] | str | Path) -> bool:
    """Validate the ordered, unknown-aware 21-factor SAVE schema."""
    if isinstance(schema, (str, Path)):
        data = yaml.safe_load(Path(schema).read_text(encoding="utf-8")) or {}
    else:
        data = dict(schema)
    rows = data.get("factors")
    if not isinstance(rows, list) or [int(row.get("id", -1)) for row in rows] != list(range(21)):
        raise ValueError("SAVE factor schema must contain ordered factor IDs 0..20")
    if data.get("unknown_is_negative") is not False:
        raise ValueError("SAVE unknown_is_negative must be false")
    groundability = {str(row.get("groundability", "")).lower() for row in rows}
    if not {"full", "partial", "latent"}.issubset(groundability):
        raise ValueError("SAVE schema must distinguish full, partial, and latent factors")
    if tuple(data.get("action_names", ())) != ("forward", "stop", "left", "right"):
        raise ValueError("SAVE action names do not match the four-action contract")
    if any("compatible_actions" in row for row in rows):
        raise ValueError("SAVE schema cannot contain hard action-factor compatibility")
    return True


__all__ = [
    "L08_WRITE_SET",
    "SAVE_L08_WRITE_SET",
    "SAVE_SOURCE_HEAD",
    "SAVE_TARGET_BRANCH",
    "SAVE_TARGET_WORKTREE_SUFFIX",
    "SAVE_LOCAL_MIRROR_SUFFIX",
    "validate_save_config",
    "validate_save_factor_schema",
    "validate_save_source_head",
    "validate_save_worktree",
]
