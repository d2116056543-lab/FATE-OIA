import torch

from fate_oia.models.acpr_oia_model import ACPROIAModel


def test_gem_zero_init_model_matches_calalign_with_mock_dino():
    torch.manual_seed(7)
    base = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, gem_enabled=False, dim=384)
    gem = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, gem_enabled=True, dim=384)
    gem.load_state_dict(base.state_dict(), strict=False)
    image = torch.randn(2, 3, 360, 640)

    out_base = base(image, epoch=0)
    out_gem = gem(image, epoch=0)

    for key in ["action_logits_base", "reason_logits_base", "logits_deploy", "action_set_logits", "predicate_probs"]:
        assert torch.allclose(out_base[key], out_gem[key], atol=1e-6), key
    assert out_gem["evidence_tokens"].shape[:2] == (2, 20)
    assert out_gem["evidence_oracle_mode"] is False
