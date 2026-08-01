import torch

from fate_oia.models.meter_signed_factors import TypedEvidenceStateHead
from tesa_helpers import typed_inputs


def test_reliability_matches_visual_confidence_from_null_and_entropy() -> None:
    nodes, patches = typed_inputs()
    head = TypedEvidenceStateHead(dim=32)
    out = head(nodes, patches)
    cardinality = head.state_cardinalities.to(out["factor_state_entropy"])
    expected = (1 - out["factor_null_mass"]) * (
        1 - out["factor_state_entropy"] / cardinality.log().view(1, -1)
    )
    torch.testing.assert_close(out["factor_reliability"], expected.clamp(0, 1))
    torch.testing.assert_close(out["factor_visual_confidence"], expected.clamp(0, 1))
    torch.testing.assert_close(
        out["factor_observability"], out["factor_visual_confidence"]
    )
