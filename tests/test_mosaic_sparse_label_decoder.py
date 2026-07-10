from __future__ import annotations

import inspect

import torch

from fate_oia.models.mosaic_sparse_label_decoder import MOSAICSparseLabelDecoder


def _pyramid(batch: int = 2, dim: int = 8) -> dict[str, torch.Tensor]:
    return {
        "F_hi": torch.randn(batch, dim, 45, 80),
        "F_mid": torch.randn(batch, dim, 23, 40),
        "F_ctx": torch.randn(batch, dim, 12, 20),
    }


def test_sparse_decoder_uses_context_retrieval_self_attention_and_per_label_logits() -> None:
    decoder = MOSAICSparseLabelDecoder(
        num_labels=4,
        dim=8,
        decoder_layers=2,
        self_attention_heads=2,
        highres_topk=16,
        midres_topk=8,
    )
    output = decoder(_pyramid())
    assert output["label_logits"].shape == (2, 4)
    assert output["label_nodes"].shape == (2, 4, 8)
    assert output["highres_indices"].shape == (2, 4, 16)
    assert output["midres_indices"].shape == (2, 4, 8)
    assert output["retrieval_attention"].shape == (2, 4, 24)
    assert torch.allclose(output["retrieval_attention"].sum(-1), torch.ones(2, 4), atol=1e-5)
    assert decoder.classifier_weight.shape == (4, 8)


def test_soft_factor_mask_constrains_sparse_retrieval_with_global_fallback() -> None:
    decoder = MOSAICSparseLabelDecoder(
        num_labels=2,
        dim=4,
        decoder_layers=1,
        self_attention_heads=2,
        highres_topk=8,
        midres_topk=4,
        mask_fallback_floor=1e-4,
    )
    with torch.no_grad():
        decoder.high_key.weight.zero_()
        decoder.mid_key.weight.zero_()
    masks = torch.zeros(1, 2, 45, 80)
    masks[:, :, :2, :4] = 1.0
    output = decoder(_pyramid(batch=1, dim=4), highres_masks=masks)
    assert torch.all(output["highres_indices"] < 2 * 80 + 4)
    assert torch.isfinite(output["label_logits"]).all()


def test_sparse_decoder_query_seed_changes_visual_nodes_and_all_paths_get_gradients() -> None:
    torch.manual_seed(37)
    decoder = MOSAICSparseLabelDecoder(
        num_labels=3,
        dim=8,
        decoder_layers=2,
        self_attention_heads=2,
        highres_topk=12,
        midres_topk=6,
    )
    pyramid = _pyramid(batch=2)
    seed = torch.randn(2, 3, 8, requires_grad=True)
    unseeded = decoder(pyramid)["label_nodes"]
    output = decoder(pyramid, query_seed=seed)
    assert not torch.allclose(unseeded, output["label_nodes"])
    output["label_logits"].square().mean().backward()
    assert seed.grad is not None and seed.grad.abs().sum() > 0
    for parameter in (
        decoder.label_queries,
        decoder.context_attention.in_proj_weight,
        decoder.high_key.weight,
        decoder.mid_key.weight,
        decoder.classifier_weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_sparse_decoder_does_not_materialize_dense_label_token_feature_tensor() -> None:
    source = inspect.getsource(MOSAICSparseLabelDecoder)
    assert "expand(batch_size, self.num_labels, token_count, self.dim)" not in source
    assert "[B,L,N,D]" not in source

