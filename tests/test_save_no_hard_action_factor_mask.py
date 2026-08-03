import torch

from fate_oia.models import save_predicate_measurement as predicate_module
from fate_oia.models.save_predicate_measurement import SAVEPredicateMeasurement
from tesa_helpers import typed_inputs


def test_save_predicates_remain_soft_candidates_without_hard_action_mask() -> None:
    factor_nodes, patches = typed_inputs(batch=2, dim=16, patches=20)
    measurement = SAVEPredicateMeasurement(dim=16)

    output = measurement(factor_nodes, patches, progress=0.0)

    assert output["predicate_map"].shape == (2, 21, 20)
    assert output["predicate_map_action"].shape == (2, 21, 20)
    assert torch.isfinite(output["predicate_map_action"]).all()
    # Latent predicates cannot be named, but they remain available to the
    # action reader as soft visual candidates.
    assert torch.count_nonzero(output["predicate_map_action"][:, 14]) > 0
    assert torch.count_nonzero(output["predicate_map_action"][:, 20]) > 0
    assert output["predicate_groundable_mask"][14].item() == 0.0
    assert output["predicate_named_mask"][14].item() == 0.0
    assert output["predicate_named_mask"][1].item() == 0.5
    torch.testing.assert_close(
        output["predicate_state_prob"].sum(dim=-1),
        torch.ones(2, 21),
    )


def test_save_predicate_output_schema_matches_required_contract() -> None:
    factor_nodes, patches = typed_inputs(batch=2, dim=16, patches=20)
    output = SAVEPredicateMeasurement(dim=16)(
        factor_nodes, patches, progress=1.0
    )
    expected_shapes = {
        "predicate_map": (2, 21, 20),
        "predicate_null_mass": (2, 21),
        "predicate_token": (2, 21, 16),
        "predicate_state_prob": (2, 21, 3),
        "predicate_state_entropy": (2, 21),
        "predicate_reliability": (2, 21),
        "predicate_groundable_mask": (21,),
        "predicate_named_mask": (21,),
    }

    assert predicate_module.REQUIRED_PREDICATE_OUTPUT_KEYS == (
        "predicate_map",
        "predicate_null_mass",
        "predicate_token",
        "predicate_state_prob",
        "predicate_state_entropy",
        "predicate_reliability",
        "predicate_groundable_mask",
        "predicate_named_mask",
        "predicate_mirror_pairs",
    )
    assert {
        key: tuple(output[key].shape) for key in expected_shapes
    } == expected_shapes
    assert output["predicate_mirror_pairs"] == (
        (9, 15),
        (10, 16),
        (11, 17),
        (12, 18),
        (13, 19),
        (14, 20),
    )
