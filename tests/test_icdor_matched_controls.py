from __future__ import annotations

import importlib

import torch


def test_matched_controls_have_four_real_nonoverlap_arms_with_mass_tolerance() -> None:
    module = importlib.import_module("fate_oia.engine.mosaic_icdor_audit_collectors")
    selected = torch.zeros(1, 8, 8)
    selected[:, 2:4, 2:4] = 1.0
    controls = module.build_matched_factor_controls(
        selected,
        selected_factor_type="object",
        selected_region="front",
        identity_masks=torch.stack([selected.roll(3, -1), selected.roll(-3, -1)]),
        identity_names=("front_object_a", "front_object_b"),
        identity_types=("object", "object"),
        identity_regions=("front", "front"),
        min_controls=4,
    )
    assert len(controls) >= 4
    kinds = {row["control_type"] for row in controls}
    assert {"spatial_roll", "same_type_identity"} <= kinds
    selected_mass = selected.sum().item()
    for row in controls:
        assert row["overlap"] == 0.0
        assert abs(row["mask_sum"] - selected_mass) / selected_mass <= 0.05
        assert row["factor_type"] == "object"
        assert row["region"] == "front"


def test_dense_continuous_masks_use_topk_slots_and_wrong_factor_identity() -> None:
    module = importlib.import_module("fate_oia.engine.mosaic_icdor_audit_collectors")
    grid = torch.linspace(0.1, 1.0, 100).reshape(10, 10)
    factor_masks = torch.stack((grid, grid.flip(-1))).unsqueeze(0)
    selected, controls, arm_rows = module.build_batch_matched_factor_selected_and_control_overrides(
        factor_masks,
        factor_index=0,
        factor="signal",
        factor_type="point",
        region="unspecified",
        identity_factor_names=("signal", "other_signal"),
        identity_factor_types=("point", "point"),
        identity_regions=("unspecified", "unspecified"),
        evidence_slots=4,
    )

    selected_mask = selected[0, 0]
    assert torch.count_nonzero(selected_mask) == 4
    assert torch.isclose(selected_mask.sum(), grid.flatten().topk(4).values.sum())
    selected_support = selected_mask.bool()
    control_supports = []
    for arm_index, override in enumerate(controls):
        replacement = override[0, 0]
        control_support = replacement.bool()
        control_supports.append(control_support)
        assert torch.count_nonzero(control_support & selected_support) == 0
        assert torch.isclose(replacement.sum(), selected_mask.sum(), atol=1e-6)
        row = arm_rows[arm_index][0]
        assert row["available"] is True
        assert row["selected_support_count"] == 4
        assert row["control_support_method"] == "topk_continuous_evidence"
    assert arm_rows[0][0]["identity_source_factor_index"] == 1
    for left, right in zip(control_supports, control_supports[1:]):
        assert torch.count_nonzero(left & right) == 0


def test_transfer_admission_must_beat_identity_and_spatial_controls_separately() -> None:
    module = importlib.import_module("fate_oia.engine.mosaic_target_transfer_metrics")
    selected = torch.tensor([[[0.9]], [[0.8]], [[0.2]], [[0.1]]])
    deleted = torch.tensor([[[0.5]], [[0.4]], [[0.9]], [[0.8]]])
    arm_prob = torch.tensor(
        [
            [[0.2, 0.8, 0.8, 0.8]],
            [[0.1, 0.7, 0.7, 0.7]],
            [[0.2, 0.2, 0.2, 0.2]],
            [[0.1, 0.1, 0.1, 0.1]],
        ]
    ).unsqueeze(-1)
    random_mean = arm_prob.mean(dim=2)
    arm_metadata = [[
        {
            "arm_index": index,
            "control_type": "same_type_identity" if index == 0 else "spatial_roll",
            "available_sample_count": 4,
            "max_mass_error": 0.0,
            "max_overlap": 0.0,
        }
        for index in range(4)
    ]]
    result = module.compute_target_transfer_metrics(module.TargetTransferInputs(
        factor_ids=("signal",),
        target_ids=("action:brake",),
        directions=(("support",),),
        factor_visual_evidence=torch.ones(4, 1),
        selected_factor_mask=torch.ones(4, 1, dtype=torch.bool),
        matched_random_factor_mask=torch.ones(4, 1, dtype=torch.bool),
        target_evaluation_mask=torch.ones(4, 1, dtype=torch.bool),
        target_labels=torch.tensor([[1.0], [1.0], [0.0], [0.0]]),
        selected_target_prob=selected,
        matched_random_target_prob=random_mean,
        matched_random_target_prob_by_arm=arm_prob,
        deleted_target_prob=deleted,
        matched_control_arms=arm_metadata,
    ))
    row = result["per_target"][0]
    assert row["tes"] > 0.0
    assert row["tes_spatial"] > 0.0
    assert row["tes_identity"] < 0.0
    assert row["admitted"] is False


def test_transfer_without_category_controls_is_diagnostic_only() -> None:
    module = importlib.import_module("fate_oia.engine.mosaic_target_transfer_metrics")
    result = module.compute_target_transfer_metrics(module.TargetTransferInputs(
        factor_ids=("signal",),
        target_ids=("action:brake",),
        directions=(("support",),),
        factor_visual_evidence=torch.ones(4, 1),
        selected_factor_mask=torch.ones(4, 1, dtype=torch.bool),
        matched_random_factor_mask=torch.ones(4, 1, dtype=torch.bool),
        target_evaluation_mask=torch.ones(4, 1, dtype=torch.bool),
        target_labels=torch.tensor([[1.0], [1.0], [0.0], [0.0]]),
        selected_target_prob=torch.tensor([[[0.9]], [[0.8]], [[0.2]], [[0.1]]]),
        matched_random_target_prob=torch.tensor([[[0.6]], [[0.5]], [[0.2]], [[0.1]]]),
        deleted_target_prob=torch.tensor([[[0.5]], [[0.4]], [[0.9]], [[0.8]]]),
    ))

    row = result["per_target"][0]
    assert row["available"] is False
    assert row["unavailable_reason"] == "matched_controls_unavailable"
    assert row["tet"] is None
    assert row["tes"] is None
    assert row["tes_identity"] is None
    assert row["tes_spatial"] is None
    assert row["cca"] is None
    assert row["ap_delta"] is None
    assert row["admitted"] is False


def test_transfer_with_zero_available_control_arm_abstains() -> None:
    module = importlib.import_module("fate_oia.engine.mosaic_target_transfer_metrics")
    arms = [[
        {
            "arm_index": index,
            "control_type": "same_type_identity" if index == 0 else "spatial_roll",
            "available_sample_count": 0 if index == 0 else 4,
            "max_mass_error": None if index == 0 else 0.0,
            "max_overlap": None if index == 0 else 0.0,
        }
        for index in range(4)
    ]]
    result = module.compute_target_transfer_metrics(module.TargetTransferInputs(
        factor_ids=("signal",), target_ids=("action:brake",),
        directions=(("support",),), factor_visual_evidence=torch.ones(4, 1),
        selected_factor_mask=torch.ones(4, 1, dtype=torch.bool),
        matched_random_factor_mask=torch.ones(4, 1, dtype=torch.bool),
        target_evaluation_mask=torch.ones(4, 1, dtype=torch.bool),
        target_labels=torch.tensor([[1.0], [1.0], [0.0], [0.0]]),
        selected_target_prob=torch.tensor([[[0.9]], [[0.8]], [[0.2]], [[0.1]]]),
        matched_random_target_prob=torch.tensor([[[0.8]], [[0.7]], [[0.2]], [[0.1]]]),
        matched_random_target_prob_by_arm=torch.zeros(4, 1, 4, 1),
        deleted_target_prob=torch.tensor([[[0.5]], [[0.4]], [[0.9]], [[0.8]]]),
        matched_control_arms=arms,
    ))
    row = result["per_target"][0]
    assert row["available"] is False
    assert row["unavailable_reason"] == "matched_controls_unavailable"
    assert all(row[key] is None for key in ("tet", "tes", "tes_identity", "tes_spatial", "cca", "ap_delta"))
    assert row["admitted"] is False
