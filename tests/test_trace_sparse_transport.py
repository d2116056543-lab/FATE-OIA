import torch
from fate_oia.models.reason_causal_prototype_transport import ReasonCausalPrototypeTransport, masked_sparsemax

def test_sparsemax_sums_and_sparsity():
    out = masked_sparsemax(torch.tensor([[1.0, 0.0, -1.0, -99.0]]), torch.tensor([[1, 1, 1, 0]], dtype=torch.bool))
    assert out[0, 3].item() == 0 and torch.allclose(out.sum(-1), torch.ones(1), atol=1e-6)

def test_transport_shape_and_gradients():
    m = ReasonCausalPrototypeTransport(dim=16)
    ev = {"evidence_tokens": torch.randn(2, 5, 16), "evidence_mask": torch.ones(2, 5, dtype=torch.bool), "evidence_source_type": torch.zeros(2, 5, dtype=torch.long), "reason_evidence_mask": torch.zeros(2, 21, 5, dtype=torch.bool)}
    out = m(torch.randn(2, 21, 16), torch.randn(2, 21), ev)
    assert out["T"].shape == (2, 21, 6, 5)
    out["evidence_reason_logits"].sum().backward()
    assert m.prototypes.grad is not None and m.prototypes.grad.norm() > 0
