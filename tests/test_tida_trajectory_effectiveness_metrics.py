import inspect

import torch
import pytest

from fate_oia.engine.evaluate_tida_oia import (
    collect_tida_outputs,
    save_epoch_outputs,
    trajectory_traffic_effectiveness_metrics,
)


def test_collect_outputs_collects_every_declared_trajectory_diagnostic():
    source = inspect.getsource(collect_tida_outputs)
    assert source.count('"trajectory_order_contrast_rms"') >= 2
    assert source.count('"trajectory_support_gate"') >= 2


def test_trajectory_metrics_expose_dynamic_gain_transport_and_grounding_quality():
    target = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]] * 8)
    sign = 2.0 * target - 1.0
    semantic = -0.05 * sign
    delta = 0.10 * sign
    rows = {
        "semantic_action": semantic,
        "trajectory_action": delta,
        "semantic_trajectory_action": semantic + delta,
        "video_action": semantic + delta,
        "image_reason": torch.zeros(16, 21),
        "video_reason": torch.zeros(16, 21),
        "action_target": target,
        "reason_target": torch.zeros(16, 21),
        "traffic_trajectory_delta": delta,
        "traffic_trajectory_control_delta": torch.zeros_like(delta),
        "traffic_trajectory_support": torch.full((16, 4), 0.8),
        "trajectory_support_gate": torch.full((16, 4), 0.94),
        "trajectory_order_gate": torch.full((16, 4), 0.6),
        "trajectory_uncertainty_gate": torch.full((16, 4), 0.7),
        "trajectory_attention": torch.full((16, 4, 3), 1.0 / 3.0),
        "trajectory_speed": torch.ones(16, 4, 3, 4),
        "trajectory_cycle_confidence": torch.full((16, 4, 3, 5), 0.9),
        "trajectory_exclusive_displacement": torch.ones(16, 4, 3, 4, 2) * 0.1,
        "trajectory_order_contrast_rms": torch.full((16, 4), 0.2),
    }
    metrics = trajectory_traffic_effectiveness_metrics(rows)
    assert metrics["overall"]["trajectory_incremental_action_mf1"] > 0
    assert metrics["target_transport"]["action_signed_margin_mean"] > 0
    assert metrics["target_transport"]["ordered_control_advantage_mean"] > 0
    assert metrics["target_transport"]["correction_to_harm_ratio"] > 1
    assert metrics["grounding_quality"]["cycle_confidence_mean"] == pytest.approx(0.9)
    assert metrics["grounding_quality"]["support_gate_mean"] == pytest.approx(0.94)
    assert metrics["dynamic_conditioned"]["high_motion"]["count"] > 0


def test_trajectory_metrics_report_corrective_and_harmful_flips():
    target = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
    semantic = torch.tensor([[-0.2], [0.2], [-0.2], [0.2]])
    trajectory = torch.tensor([[0.2], [-0.2], [-0.3], [0.3]])
    rows = {
        "semantic_action": semantic,
        "semantic_trajectory_action": trajectory,
        "video_action": trajectory,
        "image_reason": torch.zeros(4, 21),
        "video_reason": torch.zeros(4, 21),
        "action_target": target,
        "reason_target": torch.zeros(4, 21),
        "traffic_trajectory_delta": trajectory - semantic,
        "traffic_trajectory_control_delta": torch.zeros_like(trajectory),
        "traffic_trajectory_support": torch.ones(4, 1),
        "trajectory_support_gate": torch.ones(4, 1),
        "trajectory_order_gate": torch.ones(4, 1),
        "trajectory_uncertainty_gate": torch.ones(4, 1),
        "trajectory_attention": torch.ones(4, 1, 1),
        "trajectory_speed": torch.ones(4, 1, 1, 1),
        "trajectory_cycle_confidence": torch.ones(4, 1, 1, 2),
        "trajectory_exclusive_displacement": torch.ones(4, 1, 1, 1, 2),
        "trajectory_order_contrast_rms": torch.full((4, 1), 0.2),
    }
    metrics = trajectory_traffic_effectiveness_metrics(rows)
    assert metrics["decision_flips"]["fn_to_tp"] == [1]
    assert metrics["decision_flips"]["fp_to_tn"] == [1]
    assert metrics["decision_flips"]["tn_to_fp"] == [0]
    assert metrics["decision_flips"]["tp_to_fn"] == [0]


def test_epoch_artifacts_persist_trajectory_metrics_and_tensors(tmp_path):
    trajectory_metrics = {"overall": {"trajectory_incremental_action_mf1": 0.01}}
    rows = {
        "file_names": ["clip.mp4"],
        "traffic_trajectory_delta": torch.ones(1, 4),
        "traffic_trajectory_control_delta": torch.zeros(1, 4),
        "traffic_trajectory_support": torch.ones(1, 4),
        "trajectory_support_gate": torch.ones(1, 4),
        "trajectory_order_gate": torch.ones(1, 4),
        "trajectory_uncertainty_gate": torch.ones(1, 4),
        "trajectory_attention": torch.ones(1, 4, 2),
        "trajectory_speed": torch.ones(1, 4, 2, 3),
        "trajectory_xy": torch.ones(1, 4, 2, 4, 2),
        "trajectory_order_contrast_rms": torch.ones(1, 4),
    }
    save_epoch_outputs(
        tmp_path, 0, rows,
        {"online": {"trajectory_traffic_effectiveness": trajectory_metrics}},
        {"image": torch.full((25,), 0.5), "video": torch.full((25,), 0.5)},
        {"available": True},
    )
    epoch_dir = tmp_path / "epoch_000"
    assert (epoch_dir / "trajectory_traffic_effectiveness.json").exists()
    assert (epoch_dir / "traffic_trajectory_delta_test.pt").exists()
    assert (epoch_dir / "trajectory_support_gate_test.pt").exists()
    assert (epoch_dir / "trajectory_attention_test.pt").exists()
    assert (epoch_dir / "trajectory_xy_test.pt").exists()
