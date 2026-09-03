import pytest
import torch

from fate_oia.metrics import multilabel_metrics_from_logits


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device mismatch regression")
def test_multilabel_metrics_moves_vector_threshold_to_logits_device():
    logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    threshold = torch.tensor([0.5, 0.5], device="cuda")

    metrics = multilabel_metrics_from_logits(logits, targets, threshold=threshold)

    assert metrics["mF1"] == pytest.approx(1.0)
