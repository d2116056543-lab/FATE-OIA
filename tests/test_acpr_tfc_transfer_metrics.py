import torch

from fate_oia.engine.tfc_transfer_metrics import (
    aggregate_target_evidence_transfer,
    build_target_evidence_transfer_rows,
)
from fate_oia.models.tfc_deletion_contrast import TFCDeletionContrast


def test_target_transfer_gap_uses_counterfactual_deletion_not_credit_only():
    labels = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    credit = torch.ones(2, 2, 3)
    selected_effect = torch.tensor([[0.8, 0.0, 0.0], [0.4, 0.2, 0.0]])
    random_effect = torch.tensor([[0.1, 0.0, 0.0], [0.1, 0.1, 0.0]])
    selected_effect_all = torch.zeros(2, 3, 3)
    selected_effect_all[:, 0, 0] = selected_effect[:, 0]
    selected_effect_all[:, 0, 1:] = 0.05
    selected_effect_all[:, 1, 1] = selected_effect[:, 1]
    selected_effect_all[:, 1, [0, 2]] = 0.03
    stats = {
        "selected_effect": selected_effect,
        "random_effect": random_effect,
        "selected_vs_random_gap": selected_effect - random_effect,
        "credit_sign": torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        "selected_pair_mask": torch.tensor([[True, False, False], [True, True, False]]),
        "selected_factor_id": torch.tensor([[0, -1, -1], [1, 1, -1]]),
        "selected_credit_value": torch.tensor([[0.9, 0.0, 0.0], [0.7, 0.4, 0.0]]),
        "selected_effect_all_targets": selected_effect_all,
    }

    rows = build_target_evidence_transfer_rows(
        epoch=2,
        target_type="action",
        deletion_stats=stats,
        credit=credit,
        labels=labels,
    )

    target0 = rows[0]
    assert target0["valid_pairs"] == 2
    assert abs(target0["target_evidence_transfer_gap"] - 0.5) < 1e-6
    assert target0["target_evidence_specificity"] > 0.5 - 0.06
    assert target0["credit_causality_agreement"] == 1.0
    assert abs(target0["selected_credit_abs_mean"] - 0.8) < 1e-6


def test_specificity_penalizes_all_target_damage():
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    selected_effect = torch.tensor([[0.4, 0.0, 0.0]])
    random_effect = torch.tensor([[0.1, 0.0, 0.0]])
    selected_effect_all = torch.zeros(1, 3, 3)
    selected_effect_all[0, 0] = torch.tensor([0.4, 0.7, 0.7])
    stats = {
        "selected_effect": selected_effect,
        "random_effect": random_effect,
        "selected_vs_random_gap": selected_effect - random_effect,
        "credit_sign": torch.tensor([[1.0, 0.0, 0.0]]),
        "selected_pair_mask": torch.tensor([[True, False, False]]),
        "selected_factor_id": torch.tensor([[0, -1, -1]]),
        "selected_credit_value": torch.tensor([[1.0, 0.0, 0.0]]),
        "selected_effect_all_targets": selected_effect_all,
    }

    rows = build_target_evidence_transfer_rows(
        epoch=0,
        target_type="reason",
        deletion_stats=stats,
        credit=torch.ones(1, 1, 3),
        labels=labels,
    )

    assert rows[0]["target_evidence_transfer_gap"] > 0
    assert rows[0]["target_evidence_specificity"] < 0
    agg = aggregate_target_evidence_transfer(rows)
    assert agg["TFC_reason_TET_mean"] > 0
    assert agg["TFC_reason_TES_mean"] < 0


def test_valid_target_rate_counts_unique_targets_not_batch_rows():
    rows = [
        {"target_type": "action", "target_id": 0, "valid_pairs": 2, "target_evidence_transfer_gap": 1.0, "target_evidence_specificity": 1.0, "credit_causality_agreement": 1.0, "gap_positive_rate": 1.0, "specificity_positive_rate": 1.0},
        {"target_type": "action", "target_id": 0, "valid_pairs": 2, "target_evidence_transfer_gap": 1.0, "target_evidence_specificity": 1.0, "credit_causality_agreement": 1.0, "gap_positive_rate": 1.0, "specificity_positive_rate": 1.0},
        {"target_type": "action", "target_id": 1, "valid_pairs": 1, "target_evidence_transfer_gap": 1.0, "target_evidence_specificity": 1.0, "credit_causality_agreement": 1.0, "gap_positive_rate": 1.0, "specificity_positive_rate": 1.0},
    ]

    agg = aggregate_target_evidence_transfer(rows)

    assert agg["TFC_action_valid_target_rate"] == 0.5
    assert 0.0 <= agg["TFC_action_valid_target_rate"] <= 1.0


def test_deletion_contrast_returns_pair_level_transfer_tensors():
    deletion = TFCDeletionContrast(margin=0.01)
    patch_tokens = torch.randn(2, 1, 5, 4)
    topk_indices = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
    credit_norm = torch.tensor(
        [
            [[0.9, 0.0, 0.0], [0.0, 0.8, 0.0]],
            [[0.7, 0.0, 0.0], [0.0, 0.6, 0.0]],
        ],
        dtype=torch.float32,
    )
    target_logits = patch_tokens.mean(dim=(1, 2))[:, :3]

    def head_fn(tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=(1, 2))[:, :3]

    out = deletion(
        patch_tokens=patch_tokens,
        topk_indices=topk_indices,
        credit_norm=credit_norm,
        head_fn=head_fn,
        target_logits=target_logits,
        max_factors_per_sample=2,
    )

    assert out["selected_effect_all_targets"].shape == (2, 3, 3)
    assert out["random_effect_all_targets"].shape == (2, 3, 3)
    assert out["selected_pair_mask"].shape == (2, 3)
    assert out["selected_factor_id"].shape == (2, 3)
    assert out["selected_credit_value"].shape == (2, 3)
    assert int(out["selected_pair_mask"].sum().item()) > 0
