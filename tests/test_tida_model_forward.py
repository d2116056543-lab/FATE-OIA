from types import SimpleNamespace

import torch
from torch import nn

from fate_oia.models.acpr_dino_field import ACPRDinoFieldExtractor
from fate_oia.models.tida_oia_model import TIDAOIAModel


class _ImageBase(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.foundation = nn.Module()
        self.foundation.dino = ACPRDinoFieldExtractor(use_mock_dino=True, mock_dim=dim)
        self.foundation.predicate_head = SimpleNamespace(names=[f"p{i}" for i in range(32)])

    def encode_images(self, images):
        return self.foundation.dino(images)

    def decode_from_field(self, field, **kwargs):
        b, _, _, d = field["patch_tokens_by_layer"].shape
        action_logits = torch.randn(b, 4, device=field["patch_tokens_by_layer"].device)
        reason_logits = torch.randn(b, 21, device=field["patch_tokens_by_layer"].device)
        return {
            **field,
            "action_nodes_primary": torch.randn(b, 4, d, device=field["patch_tokens_by_layer"].device),
            "reason_nodes_primary": torch.randn(b, 21, d, device=field["patch_tokens_by_layer"].device),
            "predicate_tokens": torch.randn(b, 32, d, device=field["patch_tokens_by_layer"].device),
            "predicate_attention": torch.softmax(torch.randn(b, 32, 3600, device=field["patch_tokens_by_layer"].device), -1),
            "action_logits_primary": action_logits,
            "reason_logits_primary": reason_logits,
            "action_logits_final": action_logits,
            "reason_logits_final": reason_logits,
            "cls_tokens_by_layer": field["cls_tokens_by_layer"],
        }


def test_full_model_returns_formal_shapes_and_zero_scale_fallback():
    roles = {"static_anchor": [f"p{i}" for i in range(8)], "dynamic_actor": [f"p{i}" for i in range(8, 24)], "terminal_context": [f"p{i}" for i in range(24, 32)]}
    model = TIDAOIAModel(_ImageBase(), dim=8, predicate_roles=roles, context_chunk_size=7).eval()
    out = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 14, 3, 192, 344),
        torch.linspace(-5, 0, 15).unsqueeze(0), torch.ones(1, 15, dtype=torch.bool),
        temporal_action_scale=0.0, temporal_reason_scale=0.0,
    )
    assert out["video_action_logits"].shape == (1, 4)
    assert out["video_reason_logits"].shape == (1, 21)
    assert out["history_query_tokens"].shape == (1, 14, 36, 8)
    assert torch.equal(out["action_temporal_route"], out["action_route"])
    assert torch.equal(out["video_action_logits"], out["image_action_logits"])
    assert torch.equal(out["video_reason_logits"], out["image_reason_logits"])
