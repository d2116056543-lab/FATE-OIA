import torch

from fate_oia.engine.fit_tida_traffic_conditional_calibrator import (
    TrafficConditionalCalibrator,
    _align_traffic_rows,
    _apply_policy,
    _fold_assignment,
)


def test_align_traffic_rows_indexes_legacy_policy_tensors_by_target_count():
    policy = {
        "file_names": ["core-a", "core-b"],
        "source_batches": ["calib", "core", "core"],
        "pre_object_intent_action": torch.tensor([[10.0], [20.0], [30.0]]),
        "action_target": torch.tensor([[1.0], [0.0], [1.0]]),
        "reason_target": torch.tensor([[0.0], [1.0], [0.0]]),
        "_policy_cohort_sizes": {"train_calib": 1, "train_core": 2},
    }
    traffic = {
        "file_names": ["core-b", "calib-a", "core-a"],
        "source_batches": ["core", "calib", "core"],
        "video_action_logits_base": policy["pre_object_intent_action"][[2, 0, 1]],
        "action_target": policy["action_target"][[2, 0, 1]],
        "reason_target": policy["reason_target"][[2, 0, 1]],
    }

    _, _, _, aligned = _align_traffic_rows(policy, traffic)

    assert aligned["_policy"]["pre_object_intent_action"].shape[0] == 3
    assert aligned["_policy"]["pre_object_intent_action"].squeeze(1).tolist() == [30.0, 10.0, 20.0]


def test_zero_initialized_calibrator_preserves_locked_margin():
    model = TrafficConditionalCalibrator(actions=2, feature_dim=3, cap=0.04, bandwidth=0.25)
    margin = torch.tensor([[0.1, -0.2]])
    output, residual = model(margin, torch.ones(1, 2, 3), torch.ones(1, 2))
    torch.testing.assert_close(output, margin)
    torch.testing.assert_close(residual, torch.zeros_like(margin))


def test_policy_reconstruction_uses_selected_risk_route():
    rows = {
        "pre_object_intent_action": torch.zeros(2, 1),
        "object_intent_action_candidate": torch.tensor([[0.01], [0.02]]),
        "object_intent_action_directional_utility_gate": torch.zeros(2, 1),
        "object_intent_action_risk_utility_gate": torch.tensor([[0.9], [0.1]]),
    }
    policy = {
        "utility_source": [1], "gate": [1.0], "scale": [4.0], "cutoff": [0.5]
    }
    output = _apply_policy(rows, policy)
    torch.testing.assert_close(output, torch.tensor([[0.04], [0.0]]))


def test_source_stratified_folds_cover_each_source():
    sources = torch.tensor([0, 0, 0, 1, 1, 1])
    assignment = _fold_assignment(sources, folds=3)
    assert assignment.tolist() == [0, 1, 2, 0, 1, 2]


def test_fingerprint_alignment_recovers_shuffled_core_rows():
    policy = {
        "file_names": ["a", "b", "c"],
        "source_batches": ["s", "s", "s"],
        "pre_object_intent_action": torch.tensor([[1.0], [2.0], [3.0]]),
        "action_target": torch.tensor([[0.0], [1.0], [0.0]]),
        "reason_target": torch.tensor([[1.0], [0.0], [1.0]]),
        "_policy_cohort_sizes": {"train_calib": 1, "train_audit": 0, "train_core": 2},
    }
    traffic = {
        "file_names": ["0", "1", "2"],
        "source_batches": ["s", "s", "s"],
        "video_action_logits_base": torch.tensor([[3.0], [2.0], [1.0]]),
        "action_target": torch.tensor([[0.0], [1.0], [0.0]]),
        "reason_target": torch.tensor([[1.0], [0.0], [1.0]]),
    }

    policy_index, _, _, _ = _align_traffic_rows(policy, traffic)

    assert policy_index.tolist() == [2, 1, 0]
