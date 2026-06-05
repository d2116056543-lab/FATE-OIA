import torch
from fate_oia.models.egcaf_factor_judge import FactorJudge


def test_factor_judge_uses_action_gt_and_shapes():
    j = FactorJudge()
    y = torch.rand(2,4).round()
    out = j(torch.randn(2,4), torch.randn(2,4), torch.randn(2,4), y)
    assert "loss_sufficiency" in out and "loss_comprehensiveness" in out
    assert out["loss_sufficiency"].ndim == 0
