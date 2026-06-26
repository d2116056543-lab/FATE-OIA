import torch

from fate_oia.models.acpr_oia_model import ACPROIAModel


def test_gem_second_backward_reaches_queries_and_qkv():
    model = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, gem_enabled=True, dim=384)
    image = torch.randn(2, 3, 360, 640)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)

    out = model(image, epoch=0)
    loss = out["action_logits_base"].sum() + out["reason_logits_base"].sum() + out["predicate_logits"].sum()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    out = model(image, epoch=0)
    loss = out["action_logits_base"].sum() + out["reason_logits_base"].sum() + out["predicate_logits"].sum()
    loss.backward()

    assert model.evidence_memory.evidence_queries.grad is not None
    assert model.evidence_memory.evidence_queries.grad.abs().sum() > 0
    assert model.evidence_memory.query_proj.weight.grad is not None
    assert model.evidence_memory.query_proj.weight.grad.abs().sum() > 0
