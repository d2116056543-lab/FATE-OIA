import json

import torch

from fate_oia.engine.train_complementary_logit_fusion import train_complementary_fusion
from fate_oia.models.complementary_logit_fusion import ComplementaryLogitFusionAdapter


def test_complementary_adapter_learns_per_label_mix_bias_and_temperature():
    model = ComplementaryLogitFusionAdapter(reason_dim=3, init_mix=0.25)
    run_c = torch.zeros(2, 3)
    score_v2 = torch.ones(2, 3)
    out = model(run_c, score_v2)
    assert out.shape == (2, 3)
    assert model.mix_weight().shape == (3,)
    assert torch.all((model.mix_weight() >= 0) & (model.mix_weight() <= 1))
    assert torch.all(model.temperature() > 0)


def test_cached_logit_fusion_training_writes_metrics_and_checkpoints(tmp_path):
    run_c = tmp_path / "run_c"
    s1 = tmp_path / "s1"
    out = tmp_path / "fusion"
    run_c.mkdir()
    s1.mkdir()
    names = ["a", "b", "c", "d"]
    action_labels = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
    action_logits = torch.where(action_labels > 0, torch.full_like(action_labels, 4.0), torch.full_like(action_labels, -4.0))
    reason_labels = torch.zeros(4, 21)
    reason_labels[0, 0] = 1
    reason_labels[1, 0] = 1
    reason_labels[2, 1] = 1
    reason_labels[3, 1] = 1
    run_c_reason = torch.zeros(4, 21)
    s1_reason = torch.where(reason_labels > 0, torch.full_like(reason_labels, 4.0), torch.full_like(reason_labels, -4.0))
    for root, reason in [(run_c, run_c_reason), (s1, s1_reason)]:
        torch.save(action_logits, root / "logits_action_fused_best_test.pt")
        torch.save(action_logits, root / "logits_action_best_test.pt")
        torch.save(reason, root / "logits_reason_best_test.pt")
        torch.save(action_labels, root / "labels_action_best_test.pt")
        torch.save(reason_labels, root / "labels_reason_best_test.pt")
        (root / "file_names_best_test.json").write_text(json.dumps(names), encoding="utf-8")

    result = train_complementary_fusion(
        run_c_dir=run_c,
        s1_dir=s1,
        output_dir=out,
        split="test",
        epochs=3,
        lr=0.1,
        init_mix=0.25,
        reference_exp_mf1=0.0,
    )

    assert (out / "checkpoint_latest.pth").exists()
    assert (out / "checkpoint_best_test.pth").exists()
    assert (out / "metrics_summary.jsonl").exists()
    assert result["best"]["Exp_mF1"] > 0.0
