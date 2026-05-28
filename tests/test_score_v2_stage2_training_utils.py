from pathlib import Path

import torch

from fate_oia.engine.train_score_v2_oia import build_score_v2_optimizer, load_checkpoint
from fate_oia.models.score_v2_oia_model import ScoreV2OIAConfig, ScoreV2OIAModel


def test_stage2_optimizer_uses_separate_head_and_adapter_lrs() -> None:
    model = ScoreV2OIAModel(
        ScoreV2OIAConfig(dim=8, action_dim=4, reason_dim=3, n_last_blocks=2, num_heads=2, use_adaptformer=True)
    )
    optimizer = build_score_v2_optimizer(model, lr_head=1e-4, lr_adapter=2e-5, weight_decay=0.01)
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert "head" in groups
    assert "adapter" in groups
    assert groups["head"]["lr"] == 1e-4
    assert groups["adapter"]["lr"] == 2e-5
    assert groups["adapter"]["params"]


def test_stage2_can_load_stage1_checkpoint_model_only_with_partial_keys(tmp_path: Path) -> None:
    stage1 = ScoreV2OIAModel(
        ScoreV2OIAConfig(dim=8, action_dim=4, reason_dim=3, n_last_blocks=2, num_heads=2, use_adaptformer=False)
    )
    ckpt = tmp_path / "stage1.pth"
    torch.save({"model": stage1.state_dict(), "epoch": 19, "best_test_score": 0.51}, ckpt)
    stage2 = ScoreV2OIAModel(
        ScoreV2OIAConfig(dim=8, action_dim=4, reason_dim=3, n_last_blocks=2, num_heads=2, use_adaptformer=True)
    )
    optimizer = build_score_v2_optimizer(stage2, lr_head=1e-4, lr_adapter=2e-5, weight_decay=0.0)
    start_epoch, best, info = load_checkpoint(
        ckpt,
        stage2,
        optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        resume_optimizer=False,
        allow_partial=True,
        model_only=True,
    )
    assert start_epoch == 1
    assert best == float("-inf")
    assert any("layer_adapters" in key for key in info["missing_keys"])
