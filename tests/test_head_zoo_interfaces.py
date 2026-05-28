import torch

from fate_oia.models.head_zoo.base import BaseOIAHead
from fate_oia.models.head_zoo.ctran_head import CTranMaskedHead
from fate_oia.models.head_zoo.ml_decoder_head import MLDecoderHead
from fate_oia.models.head_zoo.q2l_decoder_head import Q2LDecoderHead
from fate_oia.models.head_zoo.run_c_mrc_head import RunCMRCAuxHead
from fate_oia.models.head_zoo.run_c_compatible_head import RunCCompatibleHead


def _labels(batch_size=2):
    labels = torch.zeros(batch_size, 25)
    labels[:, 0] = 1
    labels[:, 4] = 1
    labels[:, 9] = 1
    return labels


def _assert_common_output(out, batch_size=2):
    assert out["logits"].shape == (batch_size, 25)
    assert out["action_logits"].shape == (batch_size, 4)
    assert out["reason_logits"].shape == (batch_size, 21)
    assert "aux_losses" in out


def test_head_zoo_heads_share_common_api():
    tokens = torch.randn(2, 12, 384)
    labels = _labels()
    heads = [
        RunCCompatibleHead(dim=384, action_dim=4, reason_dim=21),
        Q2LDecoderHead(dim=384, action_dim=4, reason_dim=21),
        MLDecoderHead(dim=384, action_dim=4, reason_dim=21, groups=4),
        CTranMaskedHead(dim=384, action_dim=4, reason_dim=21, reveal_prob=0.5),
        RunCMRCAuxHead(dim=384, action_dim=4, reason_dim=21),
    ]
    for head in heads:
        assert isinstance(head, BaseOIAHead)
        out = head(tokens, labels=labels)
        _assert_common_output(out)


def test_ctran_eval_does_not_use_ground_truth_labels():
    torch.manual_seed(7)
    tokens = torch.randn(2, 10, 384)
    labels_a = _labels()
    labels_b = 1.0 - labels_a
    head = CTranMaskedHead(dim=384, action_dim=4, reason_dim=21)
    head.eval()
    out_a = head(tokens, labels=labels_a)
    out_b = head(tokens, labels=labels_b)
    assert torch.allclose(out_a["logits"], out_b["logits"], atol=1e-6)


def test_mrc_aux_loss_only_exists_in_training_with_labels():
    tokens = torch.randn(2, 12, 384)
    labels = _labels()
    head = RunCMRCAuxHead(dim=384, action_dim=4, reason_dim=21, mrc_mask_ratio=0.5)
    out = head(tokens, labels=labels)
    assert "mrc_loss" in out["aux_losses"]
    assert out["aux_losses"]["mrc_loss"].requires_grad
