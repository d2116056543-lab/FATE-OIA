import json
from pathlib import Path

import torch

from fate_oia.engine.score_v2_diagnostics import write_score_v2_epoch_diagnostics


def test_score_v2_diagnostics_backfills_required_epoch_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    epoch_dir = run_dir / "epoch_019"
    epoch_dir.mkdir(parents=True)
    action_logits = torch.tensor([[3.0, -3.0, -3.0, -3.0], [-3.0, 3.0, -3.0, -3.0]])
    action_labels = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32)
    reason_logits = torch.zeros(2, 21)
    reason_logits[0, 12] = 4.0
    reason_logits[1, 9] = 4.0
    reason_labels = torch.zeros(2, 21)
    reason_labels[0, 12] = 1
    reason_labels[1, 9] = 1
    names = ["sample0.jpg", "sample1.jpg"]
    (run_dir / "run_manifest.json").write_text(json.dumps({"branch": "ScoreV2", "n_last_blocks": 4}), encoding="utf-8")

    write_score_v2_epoch_diagnostics(
        epoch_dir,
        run_dir=run_dir,
        split="test",
        action_logits=action_logits,
        reason_logits=reason_logits,
        labels_action=action_labels,
        labels_reason=reason_labels,
        file_names=names,
        tail_reason_indices=[12, 9, 5],
        n_last_blocks=4,
    )

    for name in [
        "logits_action.pt",
        "logits_reason.pt",
        "logits_action_reason.pt",
        "labels_action.pt",
        "labels_reason.pt",
        "file_names.json",
        "per_label_reason_audit.json",
        "tail_group_metrics.json",
        "label_query_stats.json",
        "multilayer_feature_stats.json",
        "run_manifest.json",
    ]:
        assert (epoch_dir / name).exists(), name

    tail = json.loads((epoch_dir / "tail_group_metrics.json").read_text(encoding="utf-8"))
    assert tail["tail_reason_indices"] == [12, 9, 5]
    assert tail["tail_positive_support"] == 2
    audit = json.loads((epoch_dir / "per_label_reason_audit.json").read_text(encoding="utf-8"))
    assert audit["per_label"][12]["support"] == 1
