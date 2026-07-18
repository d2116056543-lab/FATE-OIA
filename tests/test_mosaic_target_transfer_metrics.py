from __future__ import annotations

import pytest

from fate_oia.engine.mosaic_target_transfer_metrics import (
    TargetTransferInputs,
    collect_joint_target_transfer_metrics,
    compute_target_transfer_metrics,
)
import torch
from torch import nn


def _support_kwargs() -> dict[str, object]:
    control_arms = (
        {
            "control_type": "same_type_identity",
            "available_sample_count": 4,
            "max_mass_error": 0.0,
            "max_overlap": 0.0,
        },
        {
            "control_type": "spatial_roll",
            "available_sample_count": 4,
            "max_mass_error": 0.0,
            "max_overlap": 0.0,
        },
        {
            "control_type": "score_shuffle",
            "available_sample_count": 4,
            "max_mass_error": 0.0,
            "max_overlap": 0.0,
        },
        {
            "control_type": "image_shuffle",
            "available_sample_count": 4,
            "max_mass_error": 0.0,
            "max_overlap": 0.0,
        },
    )
    random_by_arm = (
        (((0.55,), (0.55,), (0.55,), (0.55,)),),
        (((0.50,), (0.50,), (0.50,), (0.50,)),),
        (((0.50,), (0.50,), (0.50,), (0.50,)),),
        (((0.35,), (0.35,), (0.35,), (0.35,)),),
    )
    return {
        "factor_ids": ("pedestrian",),
        "target_ids": ("brake",),
        "directions": (("support",),),
        "factor_visual_evidence": ((0.9,), (0.8,), (0.7,), (0.6,)),
        "selected_factor_mask": ((True,), (True,), (True,), (True,)),
        "matched_random_factor_mask": ((True,), (True,), (True,), (True,)),
        "target_evaluation_mask": ((True,), (True,), (True,), (True,)),
        "target_labels": ((1.0,), (0.0,), (1.0,), (0.0,)),
        "selected_target_prob": (((0.9,),), ((0.2,),), ((0.8,),), ((0.1,),)),
        "matched_random_target_prob": (((0.55,),), ((0.5,),), ((0.5,),), ((0.35,),)),
        "deleted_target_prob": (((0.5,),), ((0.6,),), ((0.4,),), ((0.3,),)),
        "matched_control_arms": (control_arms,),
        "matched_random_target_prob_by_arm": random_by_arm,
    }


def _support_inputs() -> TargetTransferInputs:
    return TargetTransferInputs(**_support_kwargs())


def test_target_transfer_uses_real_selected_random_and_deletion_counterfactuals() -> None:
    result = compute_target_transfer_metrics(_support_inputs())

    edge = result["per_target"][0]
    assert edge["factor_id"] == "pedestrian"
    assert edge["target_id"] == "brake"
    assert edge["selected_effect"] == pytest.approx(0.4)
    assert edge["matched_random_effect"] == pytest.approx(0.325)
    assert edge["signed_effect"] == pytest.approx(0.4)
    assert edge["tet"] == pytest.approx(0.4)
    assert edge["tes"] == pytest.approx(0.075)
    # CCA is the fraction of ontology-correct signed interventions, not the
    # old effect-magnitude surrogate.
    assert edge["cca"] == pytest.approx(1.0)
    assert edge["ap_delta"] == pytest.approx(5.0 / 12.0)
    assert result["summary"]["pair_count"] == 1
    assert result["summary"]["mean_tes"] == pytest.approx(0.075)


def test_target_transfer_veto_signs_effect_against_negative_target_probability() -> None:
    kwargs = _support_kwargs()
    kwargs["directions"] = (("veto",),)
    kwargs["target_labels"] = ((0.0,), (1.0,), (0.0,), (1.0,))
    kwargs["selected_target_prob"] = (((0.1,),), ((0.8,),), ((0.2,),), ((0.9,),))
    kwargs["matched_random_target_prob"] = (((0.35,),), ((0.5,),), ((0.4,),), ((0.4,),))
    kwargs["deleted_target_prob"] = (((0.5,),), ((0.4,),), ((0.6,),), ((0.3,),))
    inputs = TargetTransferInputs(**kwargs)

    result = compute_target_transfer_metrics(inputs)

    edge = result["per_target"][0]
    assert edge["signed_effect"] == pytest.approx(0.4)
    assert edge["tet"] == pytest.approx(0.4)
    assert edge["matched_random_effect"] == pytest.approx(0.225)
    assert edge["tes"] == pytest.approx(0.175)


def test_target_transfer_fails_closed_when_random_masks_are_not_matched() -> None:
    kwargs = _support_kwargs()
    kwargs["matched_random_factor_mask"] = ((True,), (True,), (True,), (False,))
    inputs = TargetTransferInputs(**kwargs)

    with pytest.raises(ValueError, match="matched random"):
        compute_target_transfer_metrics(inputs)


def test_target_transfer_records_unavailable_when_ap_has_no_real_negative_examples() -> None:
    kwargs = _support_kwargs()
    kwargs["target_labels"] = ((1.0,), (1.0,), (1.0,), (1.0,))
    inputs = TargetTransferInputs(**kwargs)

    result = compute_target_transfer_metrics(inputs)
    edge = result["per_target"][0]
    assert edge["available"] is False
    assert edge["unavailable_reason"] == "target_one_class_on_matched_control_rows"


def test_target_transfer_skips_non_candidate_factor_target_pairs() -> None:
    kwargs = _support_kwargs()
    kwargs["directions"] = (("none",),)
    with pytest.raises(ValueError, match="no candidate"):
        compute_target_transfer_metrics(TargetTransferInputs(**kwargs))


def test_joint_collector_uses_one_real_deletion_sweep_for_action_and_reason() -> None:
    class Model(nn.Module):
        def forward(self, images, *, factor_intervention_keep_mask=None, **_):
            keep = factor_intervention_keep_mask
            if keep is None:
                keep = torch.ones(images.shape[0], 2, device=images.device)
            base = images[:, 0, 0, 0]
            return {
                "factor_presence_prob": torch.full((images.shape[0], 2), 0.8, device=images.device),
                "factor_soft_masks": torch.ones(images.shape[0], 2, 2, 2, device=images.device),
                "action_final_logits": (base + 1.5 * keep[:, 0]).unsqueeze(1),
                "reason_observed_logits": (base - 1.5 * keep[:, 1]).unsqueeze(1),
            }

    batch = {
        "split": ["train_audit"] * 4,
        "image": torch.tensor([1.0, -1.0, 0.5, -0.5]).view(4, 1, 1, 1),
        "action": torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
        "reason": torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
    }
    result = collect_joint_target_transfer_metrics(
        Model(), [batch], factor_ids=("f0", "f1"), action_ids=("a0",), reason_ids=("r0",),
        action_directions=(("support",), ("none",)),
        reason_directions=(("none",), ("veto",)),
        device=torch.device("cpu"), route_mode="admitted", latent_enabled=True,
    )
    assert {(row["factor_id"], row["target_id"]) for row in result["per_target"]} == {
        ("f0", "action:a0"), ("f1", "reason:r0")
    }
