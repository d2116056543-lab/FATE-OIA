import torch

from fate_oia.models.cage_label_nodes import CAGELabelNodes


def test_cage_label_nodes_are_first_class_action_and_reason_nodes():
    nodes = CAGELabelNodes(action_dim=4, reason_dim=21, hidden_dim=32)
    out = nodes(batch_size=2)
    assert out["label_queries"].shape == (25, 32)
    assert out["batched_label_queries"].shape == (2, 25, 32)
    assert out["label_type_ids"].shape == (25,)
    assert out["label_type_ids"][:4].tolist() == [0, 0, 0, 0]
    assert out["label_type_ids"][4:].unique().tolist() == [1]
    assert not torch.allclose(out["label_queries"][0], out["label_queries"][4])
