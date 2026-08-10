import torch
from vetra_test_utils import inputs, transport


def test_visual_owner_shuffle_changes_prediction():
    model, data = transport(), inputs()
    formal = model(**data)
    shuffled = model(**data, visual_shuffle=True)
    assert not torch.equal(formal["vetra_action_delta"], shuffled["vetra_action_delta"])
