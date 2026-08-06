from fate_oia.models.aie_evidence_interface import AIEEvidenceInterface


def test_group_attention_sequence_is_four_probes_per_action():
    module = AIEEvidenceInterface(dim=32, grid_hw=(4, 5), local_points_per_layer=2)
    assert module.action_dim == 4 and module.probes_per_action == 4
    assert module.group_attention.embed_dim == 32

