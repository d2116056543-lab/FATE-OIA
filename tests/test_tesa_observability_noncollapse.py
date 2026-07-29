import torch

from fate_oia.models.meter_signed_factors import TypedEvidenceStateHead
from tesa_helpers import typed_inputs


def test_reliability_matches_observability_null_and_entropy() -> None:
    nodes, patches = typed_inputs()
    out = TypedEvidenceStateHead(dim=32)(nodes, patches)
    cardinality = torch.full((1, 21), 3.0)
    expected = out["factor_observability"] * (1 - out["factor_null_mass"]) * (
        1 - out["factor_state_entropy"] / cardinality.log()
    )
    torch.testing.assert_close(out["factor_reliability"], expected.clamp(0, 1))
    assert out["factor_observability"].std() > 0
