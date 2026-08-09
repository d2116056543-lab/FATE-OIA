import torch

from fate_oia.models.pact_reason_rereader import PACTReasonRereader


def test_reason_delta_uses_true_literal_budget():
    model = PACTReasonRereader(dim=16, num_layers=3, num_predicates=4, predicate_names=["a", "b", "c", "d"])
    batch, reasons, actions, probes, patches = 2, 21, 4, 4, 12
    out = model(
        torch.randn(batch, reasons, 16), torch.randn(batch, 3, patches, 16),
        torch.randn(batch, actions, probes, 16), torch.softmax(torch.randn(batch, actions, probes, patches), -1),
        torch.randn(batch, actions, probes), torch.softmax(torch.randn(batch, 4, patches), -1),
        torch.sigmoid(torch.randn(batch, 4)), torch.randn(batch, reasons), reason_budget=0.07,
    )
    assert float(out["reason_delta"].abs().max()) <= 0.0700001
    assert float(out["reason_delta_to_budget_max"]) <= 1.000001
