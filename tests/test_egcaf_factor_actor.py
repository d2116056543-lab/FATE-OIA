import torch
from fate_oia.models.egcaf_factor_actor import FactorActor


def test_actor_selected_factor_only_and_prototypes():
    actor = FactorActor(hidden_dim=32)
    assert hasattr(actor, "prototype_vectors")
    x = torch.randn(2,4,3,32, requires_grad=True)
    w = torch.softmax(torch.randn(2,4,3), -1)
    out = actor(x, w)
    assert out["action_core_logits"].shape == (2,4)
    out["action_core_logits"].sum().backward()
    assert x.grad is not None
