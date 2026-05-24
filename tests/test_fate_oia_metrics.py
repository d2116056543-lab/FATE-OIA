import torch

from fate_oia.metrics import multilabel_metrics_from_logits, sigmoid_probs
from fate_oia.threshold_tuning import tune_global_threshold, tune_per_label_thresholds


def test_sigmoid_not_softmax_metrics():
    logits = torch.tensor([[5.0, -5.0, 5.0], [-5.0, 5.0, -5.0]])
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    probs = sigmoid_probs(logits)
    assert probs[0, 0] > 0.99 and probs[0, 2] > 0.99
    m = multilabel_metrics_from_logits(logits, labels, 0.5)
    assert m["mF1"] > 0.99
    t, gm = tune_global_threshold(logits, labels)
    assert 0.05 <= t <= 0.95
    pt, pm = tune_per_label_thresholds(logits, labels)
    assert pt.shape[0] == labels.shape[1]
