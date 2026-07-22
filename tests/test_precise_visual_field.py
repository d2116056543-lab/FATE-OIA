import torch

from fate_oia.models.precise_visual_field import PRECISEVisualField


def _dino_batch(batch=2):
    return {"patch_tokens_by_layer": torch.randn(batch, 3, 3600, 384), "cls_tokens_by_layer": torch.randn(batch, 3, 384), "grid_hw": (45, 80)}


def test_visual_field_retains_all_layers_tokens_and_context():
    field = PRECISEVisualField()
    output = field(_dino_batch())
    assert output.action_layers.shape == (2, 3, 3600, 384)
    assert output.reason_layers.shape == (2, 3, 3600, 384)
    assert output.evidence_layers.shape == (2, 3, 3600, 384)
    assert output.action_context.shape[1] == 3 * 9 * 16 + 3


def test_reason_loss_does_not_update_action_foundation():
    field = PRECISEVisualField()
    output = field(_dino_batch(1))
    output.reason_layers.square().mean().backward()
    assert all(parameter.grad is None for parameter in field.action_foundation.parameters())
    assert any(parameter.grad is not None for parameter in field.reason_private.parameters())
