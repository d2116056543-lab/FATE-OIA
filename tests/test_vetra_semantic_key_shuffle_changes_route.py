import torch
from vetra_test_utils import inputs, transport


def test_semantic_shuffle_changes_route():
    model, data = transport(), inputs()
    formal = model(**data)
    shuffled = model(**data, semantic_shuffle=True)
    assert not torch.equal(formal["support_route"], shuffled["support_route"])
