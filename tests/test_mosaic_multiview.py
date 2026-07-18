from __future__ import annotations

import unittest

import torch

try:
    from fate_oia.datasets.mosaic_multiview import MOSAICWeakMultiView
except ModuleNotFoundError:
    MOSAICWeakMultiView = None


FACTOR_NAMES = ("left_lane", "right_lane", "traffic_light")


class MOSAICWeakMultiViewTests(unittest.TestCase):
    def test_returns_two_reproducible_weak_views_without_hue_jitter(self) -> None:
        self.assertIsNotNone(MOSAICWeakMultiView)
        image = torch.linspace(0.0, 1.0, 3 * 4 * 6).reshape(3, 4, 6)
        first = MOSAICWeakMultiView(FACTOR_NAMES, seed=17, brightness_jitter=0.08, contrast_jitter=0.12)
        second = MOSAICWeakMultiView(FACTOR_NAMES, seed=17, brightness_jitter=0.08, contrast_jitter=0.12)

        first_output = first(image)
        second_output = second(image)

        self.assertEqual(len(first_output["images"]), 2)
        self.assertEqual(len(first_output["metadata"]), 2)
        for first_image, second_image, first_metadata, second_metadata in zip(
            first_output["images"],
            second_output["images"],
            first_output["metadata"],
            second_output["metadata"],
        ):
            self.assertTrue(torch.equal(first_image, second_image))
            self.assertEqual(first_image.shape, image.shape)
            self.assertEqual(first_metadata, second_metadata)
            self.assertEqual(first_metadata["hue_delta"], 0.0)
            self.assertLessEqual(abs(first_metadata["brightness_delta"]), 0.08)
            self.assertGreaterEqual(first_metadata["contrast_factor"], 0.88)
            self.assertLessEqual(first_metadata["contrast_factor"], 1.12)

    def test_flip_metadata_inverts_masks_coordinates_and_left_right_semantics(self) -> None:
        self.assertIsNotNone(MOSAICWeakMultiView)
        transform = MOSAICWeakMultiView(
            FACTOR_NAMES,
            flip_probability=1.0,
            brightness_jitter=0.0,
            contrast_jitter=0.0,
            seed=3,
        )
        output = transform(torch.zeros(3, 3, 5))
        # View zero is deliberately canonical. The weak companion view is
        # the one that receives the optional horizontal flip.
        metadata = output["metadata"][1]
        self.assertTrue(metadata["horizontal_flip"])
        self.assertEqual(metadata["factor_permutation"], (1, 0, 2))
        self.assertTrue(torch.equal(output["images"][1], torch.zeros(3, 3, 5).flip(-1)))

        base_masks = torch.zeros(3, 3, 5)
        base_masks[0, 1, 0] = 1.0
        base_masks[1, 1, 4] = 1.0
        base_masks[2, 0, 2] = 1.0
        permutation = torch.tensor(metadata["factor_permutation"])
        view_masks = base_masks.index_select(0, permutation).flip(-1)
        restored_masks = output["invert_factor_masks"](view_masks, metadata)
        self.assertTrue(torch.equal(restored_masks, base_masks))

        base_coordinates = torch.tensor([[-0.8, 0.1], [0.7, -0.2], [0.0, 0.4]])
        view_coordinates = base_coordinates.index_select(0, permutation).clone()
        view_coordinates[:, 0].neg_()
        restored_coordinates = output["invert_factor_coordinates"](view_coordinates, metadata)
        self.assertTrue(torch.equal(restored_coordinates, base_coordinates))

    def test_coordinate_inversion_supports_batched_factor_axes(self) -> None:
        self.assertIsNotNone(MOSAICWeakMultiView)
        transform = MOSAICWeakMultiView(FACTOR_NAMES, flip_probability=1.0, seed=9)
        metadata = transform(torch.zeros(3, 2, 4))["metadata"][1]
        permutation = torch.tensor(metadata["factor_permutation"])
        base_coordinates = torch.tensor(
            [[[[-0.6, 0.2]], [[0.5, -0.3]], [[0.1, 0.4]]]]
        )
        view_coordinates = base_coordinates.index_select(1, permutation).clone()
        view_coordinates[..., 0].neg_()

        restored = transform.invert_factor_coordinates(view_coordinates, metadata, factor_dim=1)

        self.assertTrue(torch.equal(restored, base_coordinates))

    def test_explicit_mirror_pair_is_normalized_for_both_factor_directions(self) -> None:
        self.assertIsNotNone(MOSAICWeakMultiView)
        transform = MOSAICWeakMultiView(
            ("port_boundary", "starboard_boundary", "center"),
            mirror_pairs={"port_boundary": "starboard_boundary"},
            flip_probability=1.0,
            seed=5,
        )

        metadata = transform(torch.zeros(3, 2, 4))["metadata"][1]

        self.assertEqual(metadata["factor_permutation"], (1, 0, 2))

    def test_normalized_dino_input_is_not_clamped_to_unit_interval(self) -> None:
        transform = MOSAICWeakMultiView(
            FACTOR_NAMES,
            flip_probability=0.0,
            brightness_jitter=0.0,
            contrast_jitter=0.0,
            seed=3,
        )
        image = torch.linspace(-2.0, 2.0, 3 * 4 * 5).reshape(3, 4, 5)
        output = transform(image)
        self.assertTrue(torch.equal(output["images"][0], image))
        self.assertLess(float(output["images"][0].min()), 0.0)

    def test_factor_values_restore_left_right_ontology_after_flip(self) -> None:
        transform = MOSAICWeakMultiView(FACTOR_NAMES, flip_probability=1.0, seed=1)
        metadata = transform(torch.zeros(3, 2, 4))["metadata"][1]
        canonical = torch.tensor([1.0, 2.0, 3.0])
        view = canonical.index_select(0, torch.tensor(metadata["factor_permutation"]))
        self.assertTrue(torch.equal(transform.invert_factor_values(view, metadata), canonical))
