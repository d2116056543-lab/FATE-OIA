from fate_oia.models.aie_evidence_interface import AIEEvidenceInterface


def test_predicate_bias_configuration_is_hard_bounded():
    module = AIEEvidenceInterface(dim=32, predicate_bias_max=0.25, grid_hw=(4, 5), local_points_per_layer=2)
    assert module.predicate_bias_max == 0.25

