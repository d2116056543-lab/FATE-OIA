import torch

from fate_oia.engine.tesa_diagnostics import (
    build_patch_audit_summary,
    run_stratified_patch_audit,
    source_eligible_factor_mask,
    select_geometry_matched_control,
)


def test_geometry_matched_control_preserves_count_side_depth_validity_and_disjointness() -> None:
    selected = torch.tensor([2 * 80 + 10, 2 * 80 + 11, 2 * 80 + 12])
    valid = torch.ones(45 * 80, dtype=torch.bool)
    anchor = torch.linspace(0.0, 1.0, 45 * 80)

    control, metadata = select_geometry_matched_control(
        anchor,
        selected,
        grid_hw=(45, 80),
        valid_mask=valid,
    )

    assert control.numel() == selected.numel()
    assert not bool(torch.isin(control, selected).any())
    assert metadata["selected_side"] == metadata["control_side"]
    assert metadata["selected_depth_bin"] == metadata["control_depth_bin"]
    assert metadata["control_valid_fraction"] == 1.0
    assert metadata["overlap_count"] == 0


def test_patch_audit_summary_separates_source_coverage_from_model_top_faithfulness() -> None:
    records = [
        {
            "sample_id": "a.jpg",
            "action_id": 0,
            "factor_id": 3,
            "selected_effect": 0.10,
            "control_effect": 0.02,
            "selected_minus_control": 0.08,
        },
        {
            "sample_id": "b.jpg",
            "action_id": 1,
            "factor_id": 4,
            "selected_effect": 0.08,
            "control_effect": 0.03,
            "selected_minus_control": 0.05,
        },
    ]
    result = build_patch_audit_summary(
        records,
        sample_ids={"a.jpg", "b.jpg"},
        cumulative_sample_ids={"old.jpg", "a.jpg", "b.jpg"},
        eligible_factor_ids={0, 1, 2, 3, 4},
        requested_factor_ids={3, 4},
        model_top_factor_ids={2},
        bootstrap_samples=200,
        bootstrap_seed=17,
    )

    assert result["eligible_factor_coverage"] == [0, 1, 2, 3, 4]
    assert result["requested_factor_coverage"] == [3, 4]
    assert result["executed_factor_coverage"] == [3, 4]
    assert result["model_top_factor_coverage"] == [2]
    assert result["action_coverage"] == [0, 1]
    assert result["selected_minus_control_mean"] == 0.065
    assert result["selected_minus_control_ci"]["low"] > 0.0
    assert result["selected_minus_control_ci"]["cluster_count"] == 2


def test_source_coverage_does_not_depend_on_model_observability() -> None:
    source_weight = torch.tensor([0.0, 0.8, 1.0, 0.7])
    anchor_valid = torch.tensor([True, True, False, True])
    groundable = torch.tensor([True, True, True, False])
    model_observability = torch.zeros(4)

    eligible = source_eligible_factor_mask(
        source_weight, anchor_valid, groundable
    )

    assert eligible.tolist() == [False, True, False, False]
    assert model_observability[1].item() == 0.0


def test_patch_audit_uses_source_validity_when_model_observability_is_zero() -> None:
    class SourceOnlyModel:
        def encode_images(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "patch_tokens_by_layer": torch.zeros(
                    images.shape[0], 1, 45 * 80, 2
                )
            }

        def decode_from_field(
            self, field: dict[str, torch.Tensor], *, progress: float
        ) -> dict[str, torch.Tensor]:
            batch_size = field["patch_tokens_by_layer"].shape[0]
            contribution = torch.zeros(batch_size, 4, 21)
            contribution[:, 0, 1] = 0.1
            return {
                "action_factor_contributions": contribution,
                "factor_action_ownership": torch.ones(21),
                "factor_observability": torch.zeros(batch_size, 21),
                "factor_groundable_mask": torch.ones(21),
                "factor_anchor_map": torch.arange(45 * 80).view(1, 1, -1)
                .float()
                .expand(batch_size, 21, -1),
                "action_logits_final": torch.zeros(batch_size, 4),
            }

    batch = {
        "image": torch.zeros(1, 3, 360, 640),
        "action": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "file_name": ["source-valid.jpg"],
        "meter_grounding": {
            "factor_source_weight": torch.tensor([[0.0, 1.0] + [0.0] * 19]),
            "factor_anchor_valid": torch.tensor([[False, True] + [False] * 19]),
        },
    }

    result = run_stratified_patch_audit(
        SourceOnlyModel(), [batch], torch.device("cpu"), progress=1.0,
        max_unique=1, patches_per_factor=3, factors_per_action=1,
    )

    assert result["eligible_factor_coverage"] == [1]
    assert result["requested_factor_coverage"] == [1]
    assert result["executed_factor_coverage"] == [1]


def test_patch_audit_uses_current_actions_compatibility_row() -> None:
    class ActionSpecificModel:
        def encode_images(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "patch_tokens_by_layer": torch.zeros(
                    images.shape[0], 1, 45 * 80, 2
                )
            }

        def decode_from_field(
            self, field: dict[str, torch.Tensor], *, progress: float
        ) -> dict[str, torch.Tensor]:
            batch_size = field["patch_tokens_by_layer"].shape[0]
            contribution = torch.zeros(batch_size, 4, 21)
            contribution[:, 2, 1] = 10.0
            contribution[:, 2, 12] = 1.0
            ownership = torch.zeros(4, 21)
            ownership[2, 12] = 1.0
            action_logits = torch.full((batch_size, 4), -10.0)
            action_logits[:, 2] = 10.0
            route = torch.zeros(batch_size, 4, 21)
            route[:, 2, 12] = 1.0
            return {
                "action_factor_contributions": contribution,
                "factor_action_ownership": ownership,
                "factor_observability": torch.ones(batch_size, 21),
                "factor_groundable_mask": torch.ones(21),
                "factor_anchor_map": torch.arange(45 * 80).view(1, 1, -1)
                .float()
                .expand(batch_size, 21, -1),
                "action_factor_weights": route,
                "action_logits_final": action_logits,
            }

    source = torch.zeros(1, 21)
    source[:, [1, 12]] = 1.0
    valid = source.bool()
    batch = {
        "image": torch.zeros(1, 3, 360, 640),
        "action": torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
        "file_name": ["left-action.jpg"],
        "meter_grounding": {
            "factor_source_weight": source,
            "factor_anchor_valid": valid,
        },
    }
    result = run_stratified_patch_audit(
        ActionSpecificModel(),
        [batch],
        torch.device("cpu"),
        progress=1.0,
        max_unique=1,
        patches_per_factor=3,
        factors_per_action=1,
    )
    assert result["requested_factor_coverage"] == [12]
    assert result["executed_factor_coverage"] == [12]
    assert all(record["schema_compatible"] for record in result["records"])
