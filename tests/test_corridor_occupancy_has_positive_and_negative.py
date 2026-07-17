from fate_oia.datasets.mosaic_icdor_grounding import corridor_occupancy_observation


def test_corridor_occupancy_exposes_positive_and_reliable_negative():
    positive = corridor_occupancy_observation([(0.45, 0.50, 0.65, 0.90)], corridor=(0.35, 0.45, 0.65, 1.0))
    negative = corridor_occupancy_observation([], corridor=(0.35, 0.45, 0.65, 1.0))
    assert positive["presence"] == 1.0 and positive["reliable_negative"] is False
    assert negative["presence"] == 0.0 and negative["reliable_negative"] is True

