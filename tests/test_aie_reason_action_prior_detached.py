import torch

from fate_oia.models.aie_reason_rereader import AIEReasonRereader


def test_reason_rereader_cannot_update_action_evidence_or_primary():
    module = AIEReasonRereader(dim=32, reason_dim=21, action_dim=4, probes_per_action=4, num_predicates=32)
    reason_nodes = torch.randn(2, 21, 32, requires_grad=True)
    field = torch.randn(2, 3, 20, 32)
    evidence = torch.randn(2, 4, 4, 32, requires_grad=True)
    maps = torch.softmax(torch.randn(2, 4, 4, 20), -1).requires_grad_()
    contrib = torch.randn(2, 4, 4, requires_grad=True)
    pattn = torch.softmax(torch.randn(2, 32, 20), -1).requires_grad_()
    pprob = torch.sigmoid(torch.randn(2, 32)).requires_grad_()
    primary_logits = torch.randn(2, 21, requires_grad=True)
    out = module(reason_nodes, field, evidence, maps, contrib, pattn, pprob, primary_logits, reason_scale=1.0)
    out["reason_logits_final_train"].sum().backward()
    assert reason_nodes.grad is None and primary_logits.grad is None
    assert evidence.grad is None and maps.grad is None and contrib.grad is None
    assert pattn.grad is None and pprob.grad is None


