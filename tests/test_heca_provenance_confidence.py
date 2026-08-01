import torch

from fate_oia.models.meter_signed_factors import TypedEvidenceStateHead
from tesa_helpers import typed_inputs


def test_reliability_is_visual_confidence_not_learned_source_availability() -> None:
    """The formal reliability route must not learn BDD100K source coverage."""
    nodes, patches = typed_inputs()
    head = TypedEvidenceStateHead(dim=32)
    output = head(nodes, patches)

    cardinality = head.state_cardinalities.to(output["factor_state_entropy"])
    expected = (1 - output["factor_null_mass"]) * (
        1 - output["factor_state_entropy"] / cardinality.log().view(1, -1)
    )
    torch.testing.assert_close(output["factor_visual_confidence"], expected.clamp(0, 1))
    torch.testing.assert_close(output["factor_reliability"], expected.clamp(0, 1))
    assert not hasattr(head, "obs_head")
    assert "factor_observability_tau" not in output


def test_typed_targets_expose_provenance_without_claiming_visual_label() -> None:
    """Weak-source availability is training-only eligibility, not a visual target."""
    from fate_oia.datasets.meter_typed_targets import METERTypedTargetBuilder

    builder = METERTypedTargetBuilder("configs/meter_factor_schema.yaml")
    target = builder.build(
        {
            "source_complete": True,
            "image_size": (1280, 720),
            "objects": [],
        }
    )

    assert target["factor_provenance_valid"][5]
    assert target["factor_provenance_valid"][6]
    # Deprecated aliases may remain for old artifact readers, but they are the
    # same static provenance bit and are never a learned model target.
    assert target["factor_observability_valid"][5]
    assert target["factor_observability"][5].item() == 1.0
