"""P0 contract tests for the isolated RAEL-OIA worktree.

These tests deliberately load only declarative P0 files.  Later phases may
implement the listed source files, but P0 must already preserve their exact
contract so a partial implementation cannot silently change the plan.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
import yaml
from yaml.constructor import ConstructorError


EXPECTED_BASE = "373aa49feac17372574fd7fb056c1d79c7c848fe"
EXPECTED_BRANCH = "acpr_rael_oia_v1_direct_image"
EXPECTED_GITHUB_REMOTE_IDENTITY = ("github.com", "d2116056543-lab", "fate-oia")
EXPECTED_TARGET_WORKTREE = "E:/sbw/FATE_Drive/fate_oia_acpr_rael_oia_v1_worktree"
EXPECTED_SOURCE_WORKTREE = "E:/sbw/FATE_Drive/fate_oia_acpr_calalign_v1_2_worktree"
EXPECTED_SOURCE_BRANCH = "acpr_calalign_v1_2"
EXPECTED_SKILL_RAW_SHA256 = "d55386d177f648535a66586942d02edc415d40d05843a7f493dae7360be6776d"
EXPECTED_SKILL_LF_NORMALIZED_SHA256 = "d55386d177f648535a66586942d02edc415d40d05843a7f493dae7360be6776d"
EXPECTED_CANONICAL_RECORDS = [
    "E:/sbw/FATE_Drive/task_plan.md",
    "E:/sbw/FATE_Drive/findings.md",
    "E:/sbw/FATE_Drive/progress.md",
]
EXPECTED_REQUIRED_SOURCE_FILES = {
    "configs/fate_oia_train_360x640_acpr_rael_oia_v1.yaml",
    "configs/rael_reason_semantics.yaml",
    "configs/rael_slot_schema.yaml",
    "configs/rael_action_semantics.yaml",
    "fate_oia/datasets/bdd100k_task_aware_index.py",
    "fate_oia/datasets/rael_grounding_targets.py",
    "fate_oia/transforms_rael.py",
    "fate_oia/models/rael_dino_field.py",
    "fate_oia/models/rael_multilayer_field.py",
    "fate_oia/models/rael_slot_ledger.py",
    "fate_oia/models/rael_semantic_reason.py",
    "fate_oia/models/rael_category_foundation.py",
    "fate_oia/models/rael_action_reason_bridge.py",
    "fate_oia/models/rael_relation_contributions.py",
    "fate_oia/models/rael_reason_private.py",
    "fate_oia/models/rael_oia_model.py",
    "fate_oia/losses/rael_task_losses.py",
    "fate_oia/losses/rael_grounding_losses.py",
    "fate_oia/losses/rael_counterfactual_losses.py",
    "fate_oia/losses/rael_pu_losses.py",
    "fate_oia/optim/rael_gradient_admission.py",
    "fate_oia/utils/rael_schema.py",
    "fate_oia/utils/rael_artifacts.py",
    "fate_oia/utils/rael_posthoc_calibration.py",
    "fate_oia/utils/rael_runtime.py",
    "fate_oia/engine/train_acpr_rael_oia.py",
    "fate_oia/engine/eval_acpr_rael_oia.py",
    "fate_oia/engine/audit_acpr_rael_oia.py",
    "fate_oia/engine/profile_acpr_rael_oia.py",
    "fate_oia/engine/export_rael_cases.py",
    "fate_oia/engine/supervise_acpr_rael_oia_foreground.py",
    "scripts/FATE_OIA_acpr_rael_oia_v1_foreground.ps1",
    ".codex/skills/rael-oia-v1-implementation-audit/SKILL.md",
}
EXPECTED_TEST_FILES = {
    "tests/test_rael_worktree_contract.py",
    "tests/test_rael_reason_schema.py",
    "tests/test_rael_grounding_index.py",
    "tests/test_rael_dino_contract.py",
    "tests/test_rael_multilayer_reading.py",
    "tests/test_rael_slot_competition.py",
    "tests/test_rael_slot_attributes.py",
    "tests/test_rael_absence_evidence.py",
    "tests/test_rael_semantic_reason.py",
    "tests/test_rael_action_reason_firewall.py",
    "tests/test_rael_adaptive_entmax.py",
    "tests/test_rael_unary_contribution.py",
    "tests/test_rael_pairwise_relation.py",
    "tests/test_rael_reason_private.py",
    "tests/test_rael_pu.py",
    "tests/test_rael_gradient_admission.py",
    "tests/test_rael_counterfactual.py",
    "tests/test_rael_posthoc_calibration.py",
    "tests/test_rael_model_forward.py",
    "tests/test_rael_train_protocol.py",
    "tests/test_rael_eval_contract.py",
    "tests/test_rael_runtime.py",
    "tests/test_rael_artifacts.py",
    "tests/test_rael_supervisor.py",
    "tests/test_rael_audit.py",
}
FORBIDDEN_CONFIG_FLAGS = {
    "feature_cache_enabled": False,
    "visual_feature_cache": False,
    "token_compression": "none",
    "video_input": False,
    "external_vlm": False,
    "per_image_caption": False,
    "checkpoint_distillation": False,
    "run_c_checkpoint": False,
    "cached_logits": False,
    "scene_graph": False,
    "pmi_bias": False,
    "static_cooccurrence_bias": False,
    "action_set_final": False,
    "bdd100k_test_forward": False,
    "reason_logits_to_action": False,
    "reason_probabilities_to_action": False,
    "reason_labels_to_action": False,
    "reason_private_to_action": False,
    "pu_to_action": False,
    "text_to_action": False,
}
EXPECTED_OWNER_LRS = {
    "multilayer_field": 0.0002,
    "slot_ledger_core": 0.0002,
    "slot_attribute_heads": 0.0002,
    "action_category": 0.0002,
    "semantic_reason": 0.0002,
    "action_reason_bridge": 0.0002,
    "unary_contribution": 0.0002,
    "pairwise_relation": 0.0002,
    "reason_private": 0.0003,
    "pu_private": 0.0003,
}


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects silent duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key ({key!r})",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _lf_normalized_sha256(payload: bytes) -> str:
    normalized = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"P0 contract file is missing: {path.as_posix()}"
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    assert isinstance(payload, dict), f"P0 YAML must be a mapping: {path.as_posix()}"
    return payload


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _github_remote_identity(remote: str) -> tuple[str, str, str]:
    """Normalize HTTPS and SSH remotes while preserving the target repo check."""
    value = remote.strip()
    if value.startswith("git@"):
        user_host, separator, path = value.partition(":")
        assert separator and "@" in user_host, f"invalid SCP-style Git remote: {remote!r}"
        host = user_host.split("@", 1)[1]
    else:
        parsed = urlparse(value)
        assert parsed.scheme in {"http", "https", "ssh"}, f"unsupported Git remote scheme: {remote!r}"
        assert parsed.hostname, f"Git remote has no hostname: {remote!r}"
        assert not parsed.query and not parsed.fragment, f"Git remote must not contain query/fragment: {remote!r}"
        host = parsed.hostname
        path = parsed.path
    parts = [part for part in path.strip("/").split("/") if part]
    assert len(parts) == 2, f"Git remote must identify exactly owner/repository: {remote!r}"
    owner, repository = parts
    if repository.endswith(".git"):
        repository = repository[:-4]
    assert owner and repository, f"Git remote owner/repository is empty: {remote!r}"
    return host.lower(), owner.lower(), repository.lower()


def _assert_expected_github_remote(remote: str) -> None:
    identity = _github_remote_identity(remote)
    assert identity == EXPECTED_GITHUB_REMOTE_IDENTITY, (
        f"unexpected GitHub remote identity: {identity!r}; "
        f"expected {EXPECTED_GITHUB_REMOTE_IDENTITY!r}"
    )


def _read_canonical_records_before_contract_load() -> list[str]:
    # This preserves the P0 ordering within every contract test invocation.
    paths = [Path(item) for item in EXPECTED_CANONICAL_RECORDS]
    for path in paths:
        assert path.is_file(), f"canonical project record is missing: {path}"
        assert path.read_text(encoding="utf-8"), f"canonical project record is empty: {path}"
    return [path.as_posix() for path in paths]


@pytest.fixture()
def config() -> dict[str, Any]:
    _read_canonical_records_before_contract_load()
    return _load_yaml(_root() / "configs/fate_oia_train_360x640_acpr_rael_oia_v1.yaml")


def test_worktree_isolation_and_canonical_record_contract(config: dict[str, Any]) -> None:
    root = _root()
    governance = config["governance"]
    assert Path(_git(root, "rev-parse", "--show-toplevel")).resolve() == root.resolve()
    assert _git(root, "branch", "--show-current") == EXPECTED_BRANCH
    _assert_expected_github_remote(_git(root, "remote", "get-url", "github"))
    assert _git(root, "merge-base", "--is-ancestor", EXPECTED_BASE, "HEAD") == ""
    assert governance["expected_base_commit"] == EXPECTED_BASE
    assert governance["target_branch"] == EXPECTED_BRANCH
    assert governance["push_remote"] == "github"
    assert governance["target_worktree"] == EXPECTED_TARGET_WORKTREE
    assert governance["source_worktree"] == EXPECTED_SOURCE_WORKTREE
    assert governance["source_branch"] == EXPECTED_SOURCE_BRANCH
    assert governance["canonical_records"] == EXPECTED_CANONICAL_RECORDS
    assert governance["canonical_records_read_before"] == ["code", "test", "run", "commit", "push"]
    assert governance["code_only_git_history"] is True
    assert governance["artifacts_not_committed"] == [
        ".background_runs", "checkpoint", "logits", "dataset", "cache"
    ]
    assert "p0_start_head" not in governance
    assert "p0_master_read_evidence" not in governance


def test_required_file_and_global_firewall_contract(config: dict[str, Any]) -> None:
    contracts = config["contracts"]
    assert set(contracts["required_source_files"]) == EXPECTED_REQUIRED_SOURCE_FILES
    assert set(contracts["required_test_files"]) == EXPECTED_TEST_FILES
    assert contracts["formal_path_forbidden_imports"] == [
        "ACPROIAModel",
        "ACPRLabelTrunk",
        "ACPRScenePredicateHead",
        "ACPRPredicateReasoner",
        "ACPRPairMemory",
        "ACPRActionComboAux",
        "ACPRCalibrationHead",
        "ACPRThresholdHead",
    ]
    assert config["constraints"] == FORBIDDEN_CONFIG_FLAGS
    assert config["experiment"] == {
        "name": "acpr_rael_oia_v1",
        "direct_image": True,
        "test_only_evaluation": True,
        "best_selection_split": "test",
        "best_selection_metric": "deploy_fixed_joint",
        "internal_test_selected": True,
        "publication_eligible": False,
    }


def test_fixed_training_optimizer_and_runtime_contract(config: dict[str, Any]) -> None:
    assert config["data"] == {
        "data_root": "E:/sbw/BDD-OIA/data",
        "raw_root": "E:/sbw/BDD-OIA",
        "bdd100k_root": "E:/sbw/BDD100K",
        "image_height": 360,
        "image_width": 640,
        "action_dim": 4,
        "reason_dim": 21,
    }
    assert config["backbone"] == {
        "arch": "vit_small",
        "patch_size": 8,
        "pretrained_weights": "ckp/reference/dino_deitsmall8_pretrain.pth",
        "checkpoint_key": "teacher",
        "selected_layers": [3, 6, 9, 12],
        "freeze": True,
        "no_grad": True,
        "eval_mode": True,
        "full_patch_tokens": 3600,
        "original_tokens": 3601,
        "no_cache": True,
        "no_token_compression": True,
        "mirror_fraction": 0.25,
    }
    assert config["model"] == {
        "dim": 384,
        "attention_heads": 6,
        "slot_iterations": 2,
        "entity_slots": 12,
        "road_slots": 5,
        "latent_slots": 3,
        "background_slots": 1,
        "internal_slot_count": 21,
        "public_explainable_slot_count": 20,
        "relation_hidden": 64,
        "reason_private_rank": 64,
        "entmax_alpha_init": 1.10,
        "entmax_alpha_min": 1.05,
        "entmax_alpha_max": 1.50,
        "lambda_mask_bias": 0.50,
        "gamma_local": 0.02,
        "action_semantic_bridge_rezero_init": 0.0,
        "action_semantic_bridge_cap": 0.25,
    }
    training = config["training"]
    assert training == {
        "epochs": 14,
        "precision": "bf16",
        "optimizer": "AdamW",
        "lr_main": 0.0002,
        "lr_reason_private": 0.0003,
        "weight_decay": 0.05,
        "warmup_ratio": 0.05,
        "scheduler": "cosine",
        "counterfactual_every_optimizer_steps": 8,
        "log_every_optimizer_steps": 50,
        "seed": 20260725,
        "grad_clip_norm": 1.0,
        "no_metric_early_stop": True,
    }
    assert config["gradient_admission"] == {
        "ema": 0.95,
        "reason_budget": 0.25,
        "grounding_budget": 0.15,
        "counterfactual_budget": 0.05,
    }
    assert config["pu"] == {
        "initial_enabled": False,
        "hidden_positive_fraction": 0.30,
        "min_positive_count": 20,
        "max_lambda": 0.20,
    }
    assert config["calibration"] == {
        "train_calib_fraction": 0.10,
        "in_model_optimizer": False,
        "posthoc_each_epoch": True,
        "candidate_order": ["global", "group", "shrinkage_per_label", "temperature_optional"],
        "max_threshold_rms_to_raw_logit_rms": 0.35,
        "fallback_if_deploy_mf1_below_raw_by": 0.005,
    }
    assert config["runtime"] == {
        "target_reserved_gb": 42.0,
        "hard_max_reserved_gb": 45.0,
        "test_every_epoch": True,
        "save_every_epoch": True,
        "foreground_only": True,
        "background_process_forbidden": True,
        "profile_candidates": [
            {"batch_size": 8, "gradient_accumulation_steps": 4, "num_workers": 8},
            {"batch_size": 6, "gradient_accumulation_steps": 5, "num_workers": 8},
            {"batch_size": 4, "gradient_accumulation_steps": 8, "num_workers": 8},
        ],
    }
    assert config["loss_contract"] == {
        "action_final_asl": 1.0,
        "action_global_asl": 0.5,
        "action_two_way": 0.05,
        "action_soft_f1": 0.05,
        "reason_final_evidence_conditional": 1.0,
        "reason_global_asl": 0.5,
        "reason_rank": 0.05,
        "reason_two_way": 0.05,
        "grounding_entity": 1.0,
        "grounding_road": 1.0,
        "grounding_mask_view": 0.1,
        "grounding_slot_diversity": 0.02,
        "non_regression_margin": 0.002,
        "warmup": {
            "r5_fraction_of_total_updates": 0.05,
            "r10_fraction_of_total_updates": 0.10,
            "grounding_base": 0.05,
            "grounding_r5_increment": 0.10,
            "non_regression_base": 0.02,
            "non_regression_r5_increment": 0.03,
            "pairwise_aux_r10": 0.05,
            "counterfactual_r10": 0.05,
            "feature_view_r5": 0.02,
        },
    }
    assert config["pilot"] == {
        "train_main_samples": 4096,
        "train_audit_samples": 1024,
        "train_calib_samples": 512,
        "test_samples": 512,
        "epochs": 3,
        "seed": 20260725,
    }
    owners = {entry["name"]: entry for entry in config["optimizer_owners"]}
    assert set(owners) == set(EXPECTED_OWNER_LRS)
    for name, lr in EXPECTED_OWNER_LRS.items():
        assert owners[name]["lr"] == lr
        assert owners[name]["weight_decay"] == 0.05
        assert owners[name]["exclude_from_weight_decay"] == ["norm", "bias", "embedding"]


def test_slot_action_and_skill_contract(config: dict[str, Any]) -> None:
    root = _root()
    slot_schema = _load_yaml(root / "configs/rael_slot_schema.yaml")
    action_schema = _load_yaml(root / "configs/rael_action_semantics.yaml")
    assert slot_schema["counts"] == {
        "entity_control": 12,
        "road": 5,
        "latent": 3,
        "background": 1,
        "internal": 21,
        "public_explainable": 20,
    }
    assert slot_schema["mask_contract"] == {
        "internal_shape": "[B,21,45,80]",
        "public_shape": "[B,20,45,80]",
        "background_shape": "[B,1,45,80]",
        "normalization_axis": "slot",
        "background_allowed_in_contribution": False,
        "background_allowed_in_explanation": False,
        "background_allowed_in_counterfactual": False,
    }
    assert slot_schema["binding"] == {
        "iterations": 2,
        "lambda_mask_bias": 0.50,
        "patch_competition": True,
        "layer_reading": "query_conditioned",
    }
    assert slot_schema["global_context"] == {
        "separate_from_public_slots": True,
        "separate_from_internal_slot_competition": True,
        "contribution_source": "global_only",
    }
    assert [slot["name"] for slot in slot_schema["road_slots"]] == [
        "drivable_left",
        "drivable_center",
        "drivable_right",
        "boundary_left",
        "boundary_right",
    ]
    assert len(slot_schema["entity_control_slots"]) == 12
    assert len(slot_schema["latent_slots"]) == 3
    for latent_slot in slot_schema["latent_slots"]:
        assert latent_slot["human_name"] is None
        assert latent_slot["explicit_bdd100k_semantics"] is False
        assert latent_slot["named_evidence"] is False
    all_slots = [
        *slot_schema["entity_control_slots"],
        *slot_schema["road_slots"],
        *slot_schema["latent_slots"],
        slot_schema["background_sink"],
    ]
    assert [slot["id"] for slot in all_slots] == list(range(21))
    slot_names = [slot["name"] for slot in all_slots]
    assert len(slot_names) == len(set(slot_names)) == 21
    assert slot_schema["background_sink"] == {
        "id": 20,
        "name": "background_sink",
        "contribution_allowed": False,
        "explanation_allowed": False,
        "counterfactual_allowed": False,
    }
    assert action_schema["action_order"] == ["forward", "stop", "left", "right"]
    assert action_schema["bridge"] == {
        "source": "semantic_reason_tokens",
        "rezero_init": 0.0,
        "cap": 0.25,
        "reads_reason_logits": False,
        "reads_reason_labels": False,
        "reads_reason_private": False,
        "reads_pu_targets": False,
        "reads_bdd100k_geometry": False,
        "reads_text_encoder": False,
    }
    action_queries = {item["name"]: item["query_components"] for item in action_schema["actions"]}
    action_ids = [item["id"] for item in action_schema["actions"]]
    assert action_ids == [0, 1, 2, 3]
    assert all(item["semantic_reason_access"] is True for item in action_schema["actions"])
    assert action_queries["forward"] == ["forward"]
    assert action_queries["stop"] == ["stop"]
    assert action_queries["left"] == ["side_shared", "left"]
    assert action_queries["right"] == ["side_shared", "right"]
    assert action_schema["role_application"] == "soft_prior_only"
    assert action_schema["hard_action_masks"] is False
    skill = root / ".codex/skills/rael-oia-v1-implementation-audit/SKILL.md"
    assert skill.is_file(), "the user-supplied RAEL audit skill must be installed verbatim"
    assert _lf_normalized_sha256(skill.read_bytes()) == EXPECTED_SKILL_LF_NORMALIZED_SHA256
    assert config["audit_skill"] == {
        "path": ".codex/skills/rael-oia-v1-implementation-audit/SKILL.md",
        "sha256": EXPECTED_SKILL_RAW_SHA256,
        "normalized_lf_sha256": EXPECTED_SKILL_LF_NORMALIZED_SHA256,
    }


def test_all_formal_yaml_contracts_use_the_strict_loader() -> None:
    root = _root()
    formal_yaml = (
        "configs/fate_oia_train_360x640_acpr_rael_oia_v1.yaml",
        "configs/rael_action_semantics.yaml",
        "configs/rael_reason_semantics.yaml",
        "configs/rael_slot_schema.yaml",
    )
    for relative_path in formal_yaml:
        assert isinstance(_load_yaml(root / relative_path), dict)


@pytest.mark.parametrize(
    "remote",
    [
        "https://github.com/d2116056543-lab/FATE-OIA.git",
        "git@github.com:d2116056543-lab/FATE-OIA.git",
        "ssh://git@github.com/d2116056543-lab/FATE-OIA.git",
    ],
)
def test_github_remote_identity_accepts_equivalent_target_urls(remote: str) -> None:
    _assert_expected_github_remote(remote)


@pytest.mark.parametrize(
    "remote",
    [
        "https://github.com/d2116056543-lab/other-repo.git",
        "git@github.com:other-owner/FATE-OIA.git",
        "https://gitlab.com/d2116056543-lab/FATE-OIA.git",
    ],
)
def test_github_remote_identity_rejects_other_repositories(remote: str) -> None:
    with pytest.raises(AssertionError, match="unexpected GitHub remote identity"):
        _assert_expected_github_remote(remote)


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("slot: 0\nslot: 1\n", encoding="utf-8")
    with pytest.raises(ConstructorError, match="duplicate key"):
        _load_yaml(duplicate)
