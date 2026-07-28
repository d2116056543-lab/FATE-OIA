import torch

from fate_oia.models.acpr_oia_model import ACPROIAModel
from fate_oia.models.meter_calalign_foundation import METERCalAlignFoundation


def test_meter_foundation_matches_threshold_disabled_acpr_with_same_state() -> None:
    torch.manual_seed(7)
    source = ACPROIAModel(use_mock_dino=True, threshold_enabled=False)
    foundation = METERCalAlignFoundation(use_mock_dino=True)
    foundation.load_acpr_compatible_state_dict(source.state_dict())
    source.eval()
    foundation.eval()
    images = torch.randn(2, 3, 360, 640)

    with torch.no_grad():
        expected = source(images)
        actual = foundation(images)

    torch.testing.assert_close(actual["action_logits_calalign"], expected["action_logits_base"], atol=1e-6, rtol=0)
    torch.testing.assert_close(actual["reason_logits_calalign"], expected["reason_logits_base"], atol=1e-6, rtol=0)
    torch.testing.assert_close(actual["label_nodes"], expected["label_nodes"], atol=1e-6, rtol=0)
    torch.testing.assert_close(actual["label_attention"], expected["label_attention"], atol=1e-6, rtol=0)


def test_meter_foundation_keeps_single_frozen_dino_field_contract() -> None:
    foundation = METERCalAlignFoundation(use_mock_dino=True)
    images = torch.randn(1, 3, 360, 640)
    field = foundation.encode_images(images)
    decoded = foundation.decode_foundation(field)

    assert field["patch_tokens_by_layer"].shape == (1, 3, 3600, 384)
    assert field["cls_tokens_by_layer"].shape == (1, 3, 384)
    assert decoded["foundation_patch"].shape == (1, 3600, 384)
    assert decoded["factor_base_nodes"].shape == (1, 21, 384)
    assert all(not parameter.requires_grad for parameter in foundation.dino.parameters())
    assert foundation.ordinary_dino_calls == 1
    assert not hasattr(foundation, "pair_memory")
    assert not hasattr(foundation, "threshold_head")
