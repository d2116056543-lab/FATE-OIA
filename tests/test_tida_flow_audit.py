import torch

from fate_oia.engine.audit_tida_oia_implementation import _flow_mechanism_pass
from fate_oia.engine.evaluate_tida_oia import gt_margin_advantage


def test_gt_margin_advantage_is_positive_when_real_history_improves_target_margin():
    real = torch.tensor([[2.0, -2.0]])
    changed = torch.tensor([[1.0, -1.0]])
    target = torch.tensor([[1.0, 0.0]])
    torch.testing.assert_close(gt_margin_advantage(real, changed, target), torch.ones(1, 2))


def test_gt_margin_advantage_exposes_harmful_history_by_label():
    real = torch.tensor([[0.0, 0.0]])
    changed = torch.tensor([[1.0, -1.0]])
    target = torch.tensor([[1.0, 0.0]])
    torch.testing.assert_close(gt_margin_advantage(real, changed, target), -torch.ones(1, 2))


def test_flow_mechanism_gate_uses_continuous_credit_and_image_no_harm():
    interventions = {
        "history_off": {"action_gt_margin_advantage_mean": 0.002, "reason_gt_margin_advantage_mean": 0.01},
        "repeated_last": {"action_gt_margin_advantage_mean": 0.0002, "reason_gt_margin_advantage_mean": 0.004},
        "time_shuffle": {"action_gt_margin_advantage_mean": 0.0, "reason_gt_margin_advantage_mean": -1e-5,
                         "velocity_cosine_with_reference": -0.5},
        "time_reverse": {"action_gt_margin_advantage_mean": 0.0, "reason_gt_margin_advantage_mean": -1e-5,
                         "velocity_cosine_with_reference": -1.0},
    }
    metrics = {
        "mechanism": {"available": True, "sample_count": 128, "intervention_metrics": interventions},
        "online": {"raw_fixed": {
            "image": {"Act_mF1": 0.73, "Exp_mF1": 0.40},
            "video": {"Act_mF1": 0.73, "Exp_mF1": 0.402},
        }},
    }
    assert _flow_mechanism_pass(metrics, torch.full((128, 4), 0.2), torch.full((128, 21), 0.2))
    metrics["mechanism"]["intervention_metrics"]["repeated_last"]["action_gt_margin_advantage_mean"] = -0.001
    assert not _flow_mechanism_pass(metrics, torch.full((128, 4), 0.2), torch.full((128, 21), 0.2))
