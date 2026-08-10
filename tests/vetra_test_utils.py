from pathlib import Path

import torch
from torch import nn
import yaml

from fate_oia.models.vetra_oia_model import VETRAOIAModel
from fate_oia.models.vetra_visual_factor_transport import VETRAVisualFactorTransport


def predicate_names():
    cfg = yaml.safe_load(Path("configs/aie_scene_predicates.yaml").read_text(encoding="utf-8"))
    return [row["name"] for row in cfg["predicates"]]


def transport(dim=8):
    return VETRAVisualFactorTransport(
        predicate_names(), "configs/acpr_reason_predicate_grammar.yaml", dim=dim, num_layers=3)


def inputs(batch=3, dim=8, patches=12):
    torch.manual_seed(11)
    names = predicate_names(); p = len(names)
    attention = torch.softmax(torch.randn(batch, p, patches), -1)
    return {
        "patch_tokens_by_layer_raw": torch.randn(batch, 3, patches, dim),
        "action_nodes_primary": torch.randn(batch, 4, dim),
        "reason_nodes_primary": torch.randn(batch, 21, dim),
        "predicate_tokens": torch.randn(batch, p, dim),
        "predicate_attention": attention,
        "predicate_probs": torch.sigmoid(torch.randn(batch, p)),
        "predicate_layer_weights": torch.softmax(torch.randn(p, 3), -1),
    }


class DummyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.sentinel = nn.Parameter(torch.ones(()))
        self.forward_calls = 0

    def forward(self, images, **_kwargs):
        self.forward_calls += 1
        return fake_base(batch=images.shape[0])


def build_model(dim=8):
    return VETRAOIAModel(
        DummyBase(), predicate_names(), "configs/acpr_reason_predicate_grammar.yaml",
        dim=dim, num_layers=3,
    )


def fake_base(batch=3, dim=8, patches=12):
    payload = inputs(batch=batch, dim=dim, patches=patches)
    payload.update({
        "action_logits_final": torch.randn(batch, 4),
        "reason_logits_final": torch.randn(batch, 21),
    })
    return payload
