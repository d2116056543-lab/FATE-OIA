import torch

from fate_oia.models.meter_reason_decoder import METERPrivateReasonDecoder


def test_annotation_residual_is_bounded() -> None:
    decoder = METERPrivateReasonDecoder(dim=16, reason_dim=21, action_dim=4)
    assert decoder.annotation_head[-1].out_features == 1
    assert decoder.tail_gain.shape == (21,)
