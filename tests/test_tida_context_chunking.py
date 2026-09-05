import torch

from fate_oia.models.acpr_dino_field import ACPRDinoFieldExtractor
from fate_oia.models.tida_context_encoder import TIDAContextEncoder
from fate_oia.models.tida_terminal_query_reader import TIDATerminalQueryReader


def test_context_encoder_immediately_aggregates_each_chunk():
    dino = ACPRDinoFieldExtractor(use_mock_dino=True, mock_dim=8)
    reader = TIDATerminalQueryReader(dim=8)
    context = TIDAContextEncoder(dino, reader, context_chunk_size=3)
    out = context(torch.randn(1, 7, 3, 192, 344), torch.randn(1, 4, 8), torch.randn(1, 32, 8), torch.randn(32, 8))
    assert out["history_query_tokens"].shape == (1, 7, 36, 8)
    assert out["history_query_attention"].shape[:3] == (1, 7, 36)
    assert not any(value.ndim == 5 and value.shape[2] == 3 for value in out.values() if torch.is_tensor(value))


def test_context_encoder_surfaces_private_reason_history_without_changing_base_queries():
    dino = ACPRDinoFieldExtractor(use_mock_dino=True, mock_dim=8)
    reader = TIDATerminalQueryReader(dim=8)
    context = TIDAContextEncoder(dino, reader, context_chunk_size=2)
    out = context(
        torch.randn(1, 4, 3, 192, 344),
        torch.randn(1, 4, 8),
        torch.randn(1, 32, 8),
        torch.randn(32, 8),
        reason_nodes=torch.randn(1, 21, 8),
    )
    assert out["history_query_tokens"].shape == (1, 4, 36, 8)
    assert out["history_reason_query_tokens"].shape == (1, 4, 21, 8)
    assert out["history_reason_query_attention"].shape[:3] == (1, 4, 21)
    assert out["history_reason_patch_tokens"].shape[:4] == (1, 4, 21, 12)
    assert out["history_reason_patch_xy"].shape == (1, 4, 21, 12, 2)
    assert out["history_reason_patch_weight"].shape == (1, 4, 21, 12)
    assert torch.allclose(
        out["history_reason_patch_weight"].sum(-1),
        torch.ones(1, 4, 21),
        atol=1e-5,
    )
