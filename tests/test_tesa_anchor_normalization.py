import torch

from fate_oia.models.meter_signed_factors import TypedEvidenceStateHead
from tests.tesa_helpers import typed_inputs


def test_anchor_plus_null_is_normalized_and_differentiable() -> None:
    nodes, patches = typed_inputs()
    head = TypedEvidenceStateHead(dim=32)
    out = head(nodes, patches, progress=1)
    torch.testing.assert_close(
        out["factor_anchor_map"].sum(-1) + out["factor_null_mass"],
        torch.ones(2, 21),
        atol=1e-5,
        rtol=0,
    )
    out["factor_typed_token"].sum().backward()
    assert torch.isfinite(head.anchor_query.weight.grad).all()
