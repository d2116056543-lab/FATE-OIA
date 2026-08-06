import torch

from fate_oia.models.aie_evidence_interface import AIEEvidenceInterface


def test_predicate_inputs_are_detached_and_bias_is_bounded():
    module = AIEEvidenceInterface(dim=32, num_predicates=32, grid_hw=(4, 5), local_points_per_layer=2)
    action = torch.randn(1, 4, 32, requires_grad=True)
    field = torch.randn(1, 3, 20, 32, requires_grad=True)
    pattn = torch.softmax(torch.randn(1, 32, 20), -1).requires_grad_()
    pprob = torch.sigmoid(torch.randn(1, 32)).requires_grad_()
    out = module(action, field, pattn, pprob)
    out["evidence_token"].sum().backward()
    assert pattn.grad is None and pprob.grad is None
    assert float(out["predicate_bias_strength"].min()) >= 0
    assert float(out["predicate_bias_strength"].max()) <= 0.25 + 1e-7


