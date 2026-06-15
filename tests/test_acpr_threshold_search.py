import torch

from fate_oia.utils.acpr_threshold_search import compute_fixed_metrics_at_thresholds, search_best_thresholds_for_f1


def _f1_at_threshold(prob, target, threshold):
    pred = (prob >= threshold).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    return (2 * tp / (2 * tp + fp + fn).clamp_min(1e-6)).item()


def test_search_best_thresholds_improves_over_fixed_point_five():
    probs = torch.tensor([[0.12], [0.18], [0.22], [0.80], [0.90]])
    targets = torch.tensor([[1.0], [1.0], [1.0], [0.0], [0.0]])
    logits = torch.logit(probs.clamp(1e-5, 1 - 1e-5))

    result = search_best_thresholds_for_f1(logits, targets, grid=torch.arange(0.01, 0.96, 0.01))

    fixed = _f1_at_threshold(probs, targets, 0.5)
    assert result["best_f1"][0] > fixed
    assert result["threshold_prob"].shape == (1,)
    assert torch.isfinite(result["threshold_logit"]).all()


def test_threshold_search_handles_no_positive_labels_without_nan():
    logits = torch.randn(10, 3)
    targets = torch.zeros(10, 3)

    result = search_best_thresholds_for_f1(logits, targets)
    metrics = compute_fixed_metrics_at_thresholds(logits, targets, result["threshold_prob"])

    assert torch.isfinite(result["threshold_prob"]).all()
    assert torch.isfinite(result["best_f1"]).all()
    assert "macro_f1" in metrics
