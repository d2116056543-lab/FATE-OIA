import torch
from fate_oia.losses.acpr_pmt_losses import predicate_patch_alignment_loss


def test_patch_alignment_lower_when_attention_on_target():
    attn_good = torch.zeros(1, 1, 6); attn_good[0, 0, :2] = 0.5
    attn_bad = torch.zeros(1, 1, 6); attn_bad[0, 0, 2:] = 0.25
    target = torch.zeros(1, 1, 6); target[0, 0, :2] = 1
    mask = torch.ones(1, 1)
    rel = torch.ones(1, 1)
    good, stats_good = predicate_patch_alignment_loss(attn_good, target, mask, rel)
    bad, stats_bad = predicate_patch_alignment_loss(attn_bad, target, mask, rel)
    assert good < bad
    assert stats_good["mass_mean"] > stats_bad["mass_mean"]


def test_patch_alignment_ignores_missing_mask():
    attn = torch.full((1, 1, 4), 0.25)
    target = torch.zeros(1, 1, 4)
    mask = torch.zeros(1, 1)
    rel = torch.ones(1, 1)
    loss, stats = predicate_patch_alignment_loss(attn, target, mask, rel)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
    assert stats["valid_predicate_mask_rate"] == 0.0
