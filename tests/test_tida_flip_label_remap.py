from fate_oia.datasets.bdd_oia_video import (
    ACTION_FLIP_PERMUTATION,
    REASON_FLIP_PERMUTATION,
    remap_horizontal_flip_labels,
)


def test_action_and_reason_flip_maps_are_involutions():
    for size, permutation in ((4, ACTION_FLIP_PERMUTATION), (21, REASON_FLIP_PERMUTATION)):
        values = tuple(range(size))
        flipped = remap_horizontal_flip_labels(values, permutation)
        assert remap_horizontal_flip_labels(flipped, permutation) == values


def test_directional_reason_pairs_are_swapped():
    assert REASON_FLIP_PERMUTATION[9] == 15
    assert REASON_FLIP_PERMUTATION[12] == 18
    assert REASON_FLIP_PERMUTATION[13] == 19
    assert REASON_FLIP_PERMUTATION[14] == 20
