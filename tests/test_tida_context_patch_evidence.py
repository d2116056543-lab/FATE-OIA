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
        predicate_reliability=torch.tensor([[0.8, 0.2]]),
    )
    assert result["history_action_patch_tokens"].shape == (1, 2, 4, 6, 8)
    assert result["history_action_patch_xy"].shape == (1, 2, 4, 6, 2)
    assert result["history_action_patch_weight"].shape == (1, 2, 4, 6)
    assert result["history_patch_tokens_last"].shape == (1, 2, 24 * 43, 8)
    assert result["history_grid_hw"] == (24, 43)
    assert result["history_semantic_patch_tokens"].shape == (1, 2, 1, 6, 8)
    assert result["history_semantic_patch_xy"].shape == (1, 2, 1, 6, 2)
    assert result["history_semantic_patch_weight"].shape == (1, 2, 1, 6)
    assert result["history_semantic_predicate_ids"].shape == (1, 2, 6)
    torch.testing.assert_close(result["history_action_patch_weight"].sum(-1), torch.ones(1, 2, 4))
    assert torch.all((result["history_action_patch_xy"] >= -1) & (result["history_action_patch_xy"] <= 1))


def test_legacy_action_patch_selection_remains_exact_topk():
    dino = ACPRDinoFieldExtractor(use_mock_dino=True, mock_dim=4)
    encoder = TIDAContextEncoder(dino, None, motion_topk=2)
    attention = torch.tensor([[[0.1, 0.7, 0.2, 0.0]]])
    field = {
        "patch_tokens_last": torch.arange(16, dtype=torch.float32).reshape(1, 4, 4),
        "grid_hw": (2, 2),
    }

    selected = encoder.select_action_patches(field, attention)

    assert selected["indices"].tolist() == [[[1, 2]]]
    torch.testing.assert_close(selected["weights"], torch.tensor([[[7.0 / 9.0, 2.0 / 9.0]]]))


def test_contrastive_diverse_action_patch_selection_avoids_duplicate_texture_anchors():
    dino = ACPRDinoFieldExtractor(use_mock_dino=True, mock_dim=4)
    encoder = TIDAContextEncoder(
        dino,
        None,
        motion_topk=3,
        action_patch_selection="contrastive_diverse",
        action_patch_nms_radius=1,
        action_patch_specificity_power=2.0,
    )
    attention = torch.full((1, 4, 16), 1e-4)
    # Patches 0/1 are high but shared by every action. Patch 15 is lower yet
    # uniquely supports action 0, so target-conditioned selection must retain it.
    attention[:, :, 0] = 0.50
    attention[:, :, 1] = 0.30
    attention[0, 0, 15] = 0.20
    attention[0, 1:, 15] = 1e-4
    field = {
        "patch_tokens_last": torch.randn(1, 16, 4),
        "grid_hw": (4, 4),
    }

    selected = encoder.select_action_patches(field, attention)

    action_zero = selected["indices"][0, 0].tolist()
    assert 15 in action_zero
    coordinates = [(index // 4, index % 4) for index in action_zero]
    for left, right in zip(coordinates, coordinates[1:]):
        assert max(abs(left[0] - right[0]), abs(left[1] - right[1])) > 1
    torch.testing.assert_close(selected["weights"].sum(-1), torch.ones(1, 4))
    assert selected["specificity"].shape == (1, 4, 3)


def test_context_encoder_reuses_each_dino_field_for_frozen_frame_logits():
    dino = ACPRDinoFieldExtractor(use_mock_dino=True, mock_dim=8)
    reader = TIDATerminalQueryReader(dim=8, num_actions=4, num_predicates=2)
    encoder = TIDAContextEncoder(dino, reader, context_chunk_size=2, motion_topk=4)
    decoder_calls = []

    def decode(field):
        assert field["grid_hw"] == (45, 80)
        assert field["patch_tokens_by_layer"].shape[2] == 45 * 80
        decoder_calls.append(field["patch_tokens_last"].data_ptr())
        batch = field["patch_tokens_last"].shape[0]
        mean = field["patch_tokens_last"].mean((1, 2))
        return {
            "action_logits_final": mean[:, None].expand(batch, 4),
            "reason_logits_final": mean[:, None].expand(batch, 21),
        }

    result = encoder(
        torch.randn(1, 3, 3, 192, 344),
        torch.randn(1, 4, 8),
        torch.randn(1, 2, 8),
        torch.randn(2, 8),
        predicate_reliability=torch.tensor([[0.8, 0.2]]),
        frozen_frame_decoder=decode,
    )

    assert len(decoder_calls) == 2
    assert result["history_image_action_logits"].shape == (1, 3, 4)
    assert result["history_image_reason_logits"].shape == (1, 3, 21)
