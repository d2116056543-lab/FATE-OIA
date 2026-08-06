from pathlib import Path

from fate_oia.utils.aie_hashes import object_sha256
from fate_oia.datasets.aie_splits import stable_split_ids
from fate_oia.utils.aie_artifacts import REQUIRED_BOUND_RUN_FILES, REQUIRED_EPOCH_FILES


def test_hash_is_order_stable_and_change_sensitive():
    assert object_sha256({"a": 1, "b": 2}) == object_sha256({"b": 2, "a": 1})
    assert object_sha256({"a": 1}) != object_sha256({"a": 2})


def test_calibration_and_audit_splits_are_stable_and_disjoint():
    ids = [f"sample_{index}" for index in range(100)]
    first = stable_split_ids(ids, seed=7, calib_fraction=0.1, audit_count=20)
    second = stable_split_ids(ids, seed=7, calib_fraction=0.1, audit_count=20)
    assert first == second
    assert not (set(first["train_calib"]) & set(first["train_audit"]))


def test_artifact_contract_includes_exact_plan_outputs():
    assert set(REQUIRED_BOUND_RUN_FILES) == {"AIE_IMPLEMENTATION_REVIEW.json", "AIE_RUNTIME_PROFILE.json"}
    required = {
        "metrics_summary.json", "branch_metrics.json", "calibration.json", "predicate_metrics.json",
        "naming_metrics.json", "probe_metrics.json", "counterfactual_metrics.json", "owner_metrics.json", "runtime_metrics.json",
        "action_logits_primary_test.pt", "action_logits_final_test.pt", "reason_logits_primary_test.pt",
        "reason_logits_final_test.pt", "labels_action_test.pt", "labels_reason_test.pt", "file_names_test.json",
    }
    assert required.issubset(set(REQUIRED_EPOCH_FILES))


def test_goal_completed_is_full_run_only():
    text = Path("fate_oia/engine/train_aie_oia.py").read_text(encoding="utf-8")
    assert 'if args.run_kind == "full"' in text
    assert "GOAL_COMPLETED_AIE_OIA_V1.json" in text
