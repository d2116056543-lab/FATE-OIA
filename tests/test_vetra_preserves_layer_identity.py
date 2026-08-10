import torch
from vetra_test_utils import inputs, transport


def test_layer_values_are_not_averaged_before_factor_construction():
    model, data = transport(), inputs(batch=1)
    data["patch_tokens_by_layer_raw"][:, 0].fill_(1)
    data["patch_tokens_by_layer_raw"][:, 1].fill_(2)
    data["patch_tokens_by_layer_raw"][:, 2].fill_(3)
    values, _, _ = model._visual_values(data["patch_tokens_by_layer_raw"], data["predicate_attention"])
    assert not torch.equal(values[:, :, 0], values[:, :, 1])
    assert not torch.equal(values[:, :, 1], values[:, :, 2])
