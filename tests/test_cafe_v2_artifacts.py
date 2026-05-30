from __future__ import annotations

import torch

from fate_oia.engine.train_cafe_oia import save_epoch_artifacts


def _stats() -> dict:
    logits = torch.randn(4, 25)
    labels = (torch.rand(4, 25) > 0.7).float()
    return {
        "metrics": {"Act_mF1": 0.5, "Exp_mF1": 0.4, "Exp_mAP": 0.3},
        "base_metrics": {"Act_mF1": 0.4, "Exp_mF1": 0.3},
        "tail_metrics": {"tail_F1": 0.2, "tail_AP": 0.1},
        "logits": logits,
        "labels": labels,
        "base_logits": logits * 0.9,
        "no_evidence_logits": logits * 0.8,
        "context_logits": logits * 0.7,
        "evidence_only_logits": logits * 0.6,
        "file_names": [f"{i}.jpg" for i in range(4)],
        "loss_rows": [],
        "evidence_rows": [{"evidence_object_count": 1}],
        "cf_rows": [{"direct_effect_mean": 0.1}],
        "token_rows": [{"original_tokens": 10, "reduced_tokens": 10}],
    }


def test_save_epoch_artifacts_writes_required_files(tmp_path) -> None:
    class Args:
        action_dim = 4

    save_epoch_artifacts(tmp_path, 0, _stats(), _stats(), _stats(), {"x": 1}, Args())
    required = [
        "epoch_000/metrics_summary.json",
        "epoch_000/evidence_stats.jsonl",
        "epoch_000/counterfactual_stats.jsonl",
        "epoch_000/calibration_params_test_diagnostic.json",
        "epoch_000/logits_reason_target_deleted_test.pt",
    ]
    for rel in required:
        assert (tmp_path / rel).exists()
