from __future__ import annotations

import torch

from fate_oia.models.mosaic_icdor_action_decoder import _CategorySparseCrossLayer
from fate_oia.models.mosaic_masked_target_rereader import MOSAICMaskedTargetRereader


def test_action_sparse_attention_scatter_is_bf16_safe() -> None:
    layer = _CategorySparseCrossLayer(dim=8, highres_topk=4, midres_topk=4)
    nodes = torch.randn(2, 4, 8)
    feature = torch.randn(2, 8, 3, 3)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        read, attention = layer._sparse_read(nodes, feature, layer.high_key, layer.high_value, 4)
    assert torch.isfinite(read).all()
    assert torch.isfinite(attention).all()
    assert attention.shape == (2, 4, 9)


def test_target_rereader_scatter_is_bf16_safe() -> None:
    rereader = MOSAICMaskedTargetRereader(dim=8, action_count=4, topk=4)
    feature = torch.randn(2, 8, 3, 3)
    queries = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, 3, 3)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        nodes, attention, active = rereader._read(feature, queries, mask, rereader.support_query)
    assert torch.isfinite(nodes).all()
    assert torch.isfinite(attention).all()
    assert active.all()
