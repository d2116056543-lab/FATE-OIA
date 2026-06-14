import torch

from fate_oia.engine.train_acpr_oia import pair_artifact_payload
from fate_oia.utils.acpr_pair_mining import pair_summary


def test_pair_artifact_contains_active_hard_semihard_schema():
    pairs = {
        "pair_count": 2,
        "active_pair_count": 1,
        "hard_pair_count": 1,
        "semi_hard_pair_count": 0,
        "easy_pair_count": 1,
        "zero_loss_pair_count": 1,
        "tail_pair_count": 1,
        "tail_active_pair_count": 1,
        "pair_hinge_raw": torch.tensor([0.5, -0.3]),
        "pair_active_mask": torch.tensor([True, False]),
        "pair_count_per_reason": [0] * 21,
        "active_pair_count_per_reason": [0] * 21,
        "hard_pair_count_per_reason": [0] * 21,
        "semi_hard_pair_count_per_reason": [0] * 21,
        "easy_pair_count_per_reason": [0] * 21,
        "margin_mean_per_reason": [0.0] * 21,
        "active_margin_mean_per_reason": [0.0] * 21,
    }
    pairs["pair_count_per_reason"][5] = 2
    pairs["active_pair_count_per_reason"][5] = 1
    pairs["hard_pair_count_per_reason"][5] = 1
    payload = pair_artifact_payload(pairs)
    summary = pair_summary(pairs)
    assert summary["active_pair_count"] == 1
    assert summary["hard_pair_count"] == 1
    assert "per_reason" in payload
    assert payload["per_reason"][5]["active_pair_count"] == 1

