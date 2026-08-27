import torch

from fate_oia.engine.fit_tida_traffic_boundary_cv import (
    _boundary_effectiveness,
    _collect,
    _per_source_effectiveness,
    _trainable_boundary_head,
)
from fate_oia.models.tida_traffic_boundary import TIDATrafficAdaptiveBoundary


def test_boundary_effectiveness_counts_recovery_and_damage():
    base = torch.tensor([[-0.1, 0.2], [0.2, -0.2]])
    adaptive = torch.tensor([[0.1, -0.2], [-0.1, -0.2]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    threshold = torch.tensor([0.5, 0.5])

    result = _boundary_effectiveness(base, adaptive, target, threshold)

    assert result["errors_recovered"] == 2
    assert result["correct_damaged"] == 1
    assert result["net_corrected"] == 1


def test_per_source_effectiveness_keeps_domains_separate():
    base = torch.tensor([[-0.2], [-0.2], [-0.2], [-0.2]])
    adaptive = torch.tensor([[0.2], [0.2], [-0.3], [-0.3]])
    target = torch.ones(4, 1)
    threshold = torch.tensor([0.5])

    result = _per_source_effectiveness(
        ["batch_a", "batch_a", "batch_b", "batch_b"],
        base, adaptive, target, threshold,
    )

    assert result["batch_a"]["net_corrected"] == 2
    assert result["batch_b"]["net_corrected"] == 0


def test_cv_copy_reenables_disabled_boundary_parameters():
    source = TIDATrafficAdaptiveBoundary(num_actions=4)
    for parameter in source.parameters():
        parameter.requires_grad = False

    head = _trainable_boundary_head(source, torch.device("cpu"), max_delta=0.05)

    assert all(parameter.requires_grad for parameter in head.parameters())
    assert head.cap == 0.05


class _CollectionModel(torch.nn.Module):
    def forward(self, target_image, context_images, timestamps, frame_valid_mask, **kwargs):
        batch = target_image.shape[0]
        zeros4 = torch.zeros(batch, 4)
        zeros21 = torch.zeros(batch, 21)
        return {
            "video_action_logits_base": zeros4,
            "traffic_trajectory_state_features": torch.zeros(batch, 4, 8),
            "traffic_trajectory_order_delta": zeros4,
            "traffic_trajectory_support": torch.ones(batch, 4),
            "trajectory_interaction_risk": torch.zeros(batch, 4, 2),
            "trajectory_state_strength": torch.ones(batch, 4),
            "video_reason_logits": zeros21,
            "image_action_logits": zeros4,
            "image_reason_logits": zeros21,
        }


def test_collect_preserves_dataset_file_names():
    batch = {
        "target_image": torch.zeros(2, 3, 2, 2),
        "context_images": torch.zeros(2, 1, 3, 2, 2),
        "timestamps": torch.zeros(2, 1),
        "frame_valid_mask": torch.ones(2, 2, dtype=torch.bool),
        "action": torch.zeros(2, 4),
        "reason": torch.zeros(2, 21),
        "file_name": ["a.jpg", "b.jpg"],
        "clip_meta": [{"source_batch": "batch_a"}, {"source_batch": "batch_b"}],
    }

    result = _collect(_CollectionModel(), [batch], torch.device("cpu"))

    assert result["file_names"] == ["a.jpg", "b.jpg"]
    assert result["source_batches"] == ["batch_a", "batch_b"]
