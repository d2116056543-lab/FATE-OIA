from __future__ import annotations

from dataclasses import dataclass

from fate_oia.datasets.tail_aware_sampler import compute_tail_aware_sample_weights


@dataclass
class _Sample:
    reason: tuple[float, ...]


def test_tail_aware_weights_prioritize_samples_with_rare_positive_labels() -> None:
    samples = [
        _Sample((1.0, 0.0, 0.0)),
        _Sample((1.0, 0.0, 0.0)),
        _Sample((0.0, 1.0, 0.0)),
        _Sample((0.0, 0.0, 1.0)),
    ]
    weights = compute_tail_aware_sample_weights(samples, reason_dim=3, power=1.0)
    assert weights.shape[0] == 4
    assert weights[2] > 0
    assert weights[3] > 0
    assert weights[0] < weights[3] or weights[1] < weights[3]
