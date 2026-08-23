import torch

from fate_oia.models.acpr_dino_field import ACPRDinoFieldExtractor
from fate_oia.models.tida_context_encoder import TIDAContextEncoder
from fate_oia.models.tida_terminal_query_reader import TIDATerminalQueryReader


def test_context_encoder_preserves_sparse_action_grounded_patch_evidence():
    dino = ACPRDinoFieldExtractor(use_mock_dino=True, mock_dim=8)
    reader = TIDATerminalQueryReader(dim=8, num_actions=4, num_predicates=2)
    encoder = TIDAContextEncoder(dino, reader, context_chunk_size=1, motion_topk=6)
    result = encoder(
        torch.randn(1, 2, 3, 192, 344),
        torch.randn(1, 4, 8),
        torch.randn(1, 2, 8),
        torch.randn(2, 8),
    )
    assert result["history_action_patch_tokens"].shape == (1, 2, 4, 6, 8)
    assert result["history_action_patch_xy"].shape == (1, 2, 4, 6, 2)
    assert result["history_action_patch_weight"].shape == (1, 2, 4, 6)
    assert result["history_patch_tokens_last"].shape == (1, 2, 24 * 43, 8)
    assert result["history_grid_hw"] == (24, 43)
    torch.testing.assert_close(result["history_action_patch_weight"].sum(-1), torch.ones(1, 2, 4))
    assert torch.all((result["history_action_patch_xy"] >= -1) & (result["history_action_patch_xy"] <= 1))
