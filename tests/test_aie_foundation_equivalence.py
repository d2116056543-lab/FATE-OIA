import torch

from fate_oia.models.acpr_oia_model import ACPROIAModel
from fate_oia.models.aie_calalign_foundation import AIECalAlignFoundation


def test_foundation_matches_source_raw_path():
    source = ACPROIAModel(use_mock_dino=True, threshold_enabled=False).eval()
    foundation = AIECalAlignFoundation(use_mock_dino=True).eval()
    foundation.load_from_acpr_state_dict(source.state_dict())
    image = torch.randn(1, 3, 360, 640)
    with torch.no_grad():
        expected = source(image)
        actual = foundation(image)
    pairs = {
        "action_logits_primary": "action_logits_base",
        "reason_logits_primary": "reason_logits_base",
        "label_nodes": "label_nodes",
        "label_attention": "label_attention",
        "predicate_logits": "predicate_logits",
        "predicate_attention": "predicate_attention",
    }
    for actual_key, expected_key in pairs.items():
        torch.testing.assert_close(actual[actual_key], expected[expected_key], atol=1e-6, rtol=0)


