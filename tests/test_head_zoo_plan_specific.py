import torch

from fate_oia.models.head_zoo.ctran_head import CTranMaskedHead
from fate_oia.models.head_zoo.ml_decoder_head import MLDecoderHead
from fate_oia.models.head_zoo.q2l_decoder_head import Q2LDecoderHead
from fate_oia.models.head_zoo.run_c_calibrated_head import RunCCalibratedHead
from fate_oia.engine.train_head_zoo_oia import eval_variants_from_head_output


def _labels(batch_size=2):
    labels = torch.zeros(batch_size, 25)
    labels[:, 0] = 1
    labels[:, 4] = 1
    labels[:, 12] = 1
    return labels


def test_q2l_decoder_has_two_self_layers_and_label_attention():
    torch.manual_seed(11)
    tokens = torch.randn(2, 13, 384)
    head = Q2LDecoderHead(dim=384, action_dim=4, reason_dim=21, num_heads=6, self_layers=2)
    out = head(tokens)
    assert isinstance(head.decoder.decoder, torch.nn.TransformerDecoder)
    assert len(head.decoder.decoder.layers) == 2
    assert out["logits"].shape == (2, 25)
    assert out["label_tokens"].shape == (2, 25, 384)
    assert out["attention"].shape == (2, 25, 13)
    assert torch.allclose(out["attention"].sum(dim=-1), torch.ones(2, 25), atol=1e-5)


def test_ml_decoder_has_label_specific_tokens_at_initialization():
    torch.manual_seed(12)
    tokens = torch.randn(2, 17, 384)
    head = MLDecoderHead(dim=384, action_dim=4, reason_dim=21, groups=8)
    out = head(tokens)
    assert out["logits"].shape == (2, 25)
    assert out["attention"].shape == (2, 25, 17)
    diff = (out["label_tokens"][:, 0] - out["label_tokens"][:, 1]).abs().max().item()
    assert diff > 1e-6


def test_ctran_train_uses_partial_labels_but_eval_masks_all_labels():
    torch.manual_seed(13)
    tokens = torch.randn(2, 11, 384)
    labels_a = _labels()
    labels_b = 1.0 - labels_a
    head = CTranMaskedHead(dim=384, action_dim=4, reason_dim=21, reveal_prob=1.0)
    head.train()
    train_a = head(tokens, labels=labels_a)["logits"]
    train_b = head(tokens, labels=labels_b)["logits"]
    assert not torch.allclose(train_a, train_b)
    head.eval()
    eval_a = head(tokens, labels=labels_a)["logits"]
    eval_b = head(tokens, labels=labels_b)["logits"]
    assert torch.allclose(eval_a, eval_b, atol=1e-6)


def test_calibrated_head_exposes_raw_and_calibrated_eval_variants():
    torch.manual_seed(14)
    tokens = torch.randn(2, 12, 384)
    head = RunCCalibratedHead(dim=384, action_dim=4, reason_dim=21)
    with torch.no_grad():
        head.bias[4:] = 0.25
        head.log_temp[4:] = 0.10
    out = head(tokens)
    variants = eval_variants_from_head_output(out)
    assert set(variants) == {"calibrated", "raw"}
    assert variants["calibrated"][0].shape == (2, 4)
    assert variants["calibrated"][1].shape == (2, 21)
    assert variants["raw"][0].shape == (2, 4)
    assert variants["raw"][1].shape == (2, 21)
    assert not torch.allclose(variants["calibrated"][1], variants["raw"][1])


def test_q2l_uses_real_transformer_decoder_not_simplified_encoder():
    head = Q2LDecoderHead(dim=384, action_dim=4, reason_dim=21, num_heads=6, self_layers=2)
    assert isinstance(head.decoder.decoder, torch.nn.TransformerDecoder)
    assert len(head.decoder.decoder.layers) == 2


def test_ml_decoder_uses_fixed_label_group_projection_matrix():
    head = MLDecoderHead(dim=384, action_dim=4, reason_dim=21, groups=8)
    param_names = {name for name, _ in head.named_parameters()}
    assert "label_group_logits" not in param_names
    assert hasattr(head, "label_to_group_projection")
    projection = head.label_to_group_projection
    assert projection.shape == (25, 8)
    assert torch.allclose(projection.sum(dim=1), torch.ones(25))
