import torch
from vetra_test_utils import inputs, transport


def test_zero_visual_field_cannot_create_action_bias():
    model, data = transport(), inputs()
    data["patch_tokens_by_layer_raw"].zero_()
    out = model(**data)
    assert torch.equal(out["vetra_action_delta"], torch.zeros_like(out["vetra_action_delta"]))
