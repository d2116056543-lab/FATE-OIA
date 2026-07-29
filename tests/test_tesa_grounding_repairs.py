from pathlib import Path

import torch
from PIL import Image

from fate_oia.datasets.meter_typed_targets import METERTypedTargetBuilder
from fate_oia.losses.meter_grounding_losses import mirror_equivariance_loss
from fate_oia.losses.meter_grounding_losses import discrimination_and_mirror_loss
from fate_oia.models.meter_signed_factors import TypedEvidenceStateHead


def _schema_copy_with_overrides(tmp_path: Path) -> Path:
    source = Path("configs/meter_factor_schema.yaml").read_text(encoding="utf-8")
    source = source.replace('action_owned: 1.0, observability_source', 'action_owned: 0.25, observability_source', 1)
    source = source.replace('groundability: partial, action_owned: 0.5', 'groundability: none, action_owned: 0.5', 1)
    path = tmp_path / "schema.yaml"
    path.write_text(source, encoding="utf-8")
    return path


def test_typed_head_uses_runtime_schema_for_ownership_and_groundability(tmp_path: Path):
    schema = _schema_copy_with_overrides(tmp_path)
    head = TypedEvidenceStateHead(dim=8, schema_path=schema)

    assert head.action_ownership[0].item() == 0.25
    assert head.groundable_mask[1].item() == 0.0


def test_geometry_targets_rasterize_real_lane_and_drivable_evidence(tmp_path: Path):
    drivable = tmp_path / "drive.png"
    pixels = torch.zeros(12, 16, dtype=torch.uint8)
    pixels[8:, 2:6] = 1
    Image.fromarray(pixels.numpy(), mode="L").save(drivable)
    builder = METERTypedTargetBuilder("configs/meter_factor_schema.yaml", grid_hw=(6, 8))
    record = {
        "source_complete": True,
        "objects": [],
        "lanes": [{"poly2d": [{"vertices": [[2, 10], [5, 7], [7, 5]]}]}],
        "drivable_map_path": str(drivable),
        "image_size": [16, 12],
    }

    target = builder.build(record)
    assert target["factor_anchor_valid"][2]
    assert target["factor_anchor_map"][2, :, :4].sum() > 0
    assert target["factor_anchor_map"][2, :, 4:].sum() == 0
    assert target["factor_anchor_map"][9].sum() > 0
    assert int((target["factor_anchor_map"][9] > 0).sum()) < 6 * 8

    unknown = builder.build({"source_complete": False, "objects": [], "lanes": []})
    assert not bool(unknown["factor_anchor_valid"][2])
    assert not bool(unknown["factor_state_valid"][2])


def test_object_poly2d_lane_and_explicit_style_are_used_conservatively():
    builder = METERTypedTargetBuilder(
        "configs/meter_factor_schema.yaml", grid_hw=(6, 8)
    )
    record = {
        "source_complete": True,
        "objects": [
            {
                "category": "lane",
                "poly2d": [{"vertices": [[2, 11], [3, 8], [4, 5]]}],
                "attributes": {"laneStyle": "solid"},
            }
        ],
        "image_size": [16, 12],
    }
    target = builder.build(record)

    assert target["factor_anchor_valid"][11]
    assert target["factor_anchor_map"][11].sum() > 0
    assert target["factor_state_valid"][11]
    assert target["factor_state_target"][11].item() == 0
    assert not target["factor_state_valid"][12]

    unknown = builder.build(
        {
            "source_complete": True,
            "objects": [
                {
                    "category": "lane",
                    "poly2d": [{"vertices": [[2, 11], [3, 8], [4, 5]]}],
                }
            ],
            "image_size": [16, 12],
        }
    )
    assert not unknown["factor_state_valid"][11]


def test_mirror_equivariance_report_requires_paired_forward_outputs():
    original = {
        "factor_anchor_map": torch.zeros(1, 21, 2, 4),
        "factor_state_prob": torch.zeros(1, 21, 3),
        "action_logits_final": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "reason_logits_final": torch.arange(21, dtype=torch.float32).view(1, 21),
    }
    mirrored = {
        "factor_anchor_map": torch.zeros(1, 21, 2, 4),
        "factor_state_prob": torch.zeros(1, 21, 3),
        "action_logits_final": torch.tensor([[1.0, 2.0, 4.0, 3.0]]),
        "reason_logits_final": torch.arange(21, dtype=torch.float32).view(1, 21),
    }
    mirrored["factor_anchor_map"][:, 9] = 1.0
    original["factor_anchor_map"][:, 15] = torch.flip(mirrored["factor_anchor_map"][:, 9], dims=[-1])
    mirrored["factor_state_prob"][:, 9] = 1.0
    original["factor_state_prob"][:, 15] = mirrored["factor_state_prob"][:, 9]
    for left, right in ((9, 15), (10, 16), (11, 17), (12, 18), (13, 19)):
        mirrored["reason_logits_final"][:, left], mirrored["reason_logits_final"][:, right] = (
            original["reason_logits_final"][:, right].clone(),
            original["reason_logits_final"][:, left].clone(),
        )

    loss, report = mirror_equivariance_loss(original, mirrored)
    assert float(loss) == 0.0
    assert report["paired_forward"] is True
    mirrored["action_logits_final"][0, 2] += 1.0
    loss, report = mirror_equivariance_loss(original, mirrored)
    assert float(loss) > 0.0
    assert report["action_l1"] > 0.0


def test_paired_mirror_ignores_scalar_diagnostics_in_model_output():
    output = {
        "factor_typed_token": torch.zeros(2, 21, 4),
        "factor_state_prob": torch.zeros(2, 21, 3),
        "factor_anchor_map": torch.full((2, 21, 8), 1 / 8),
        "action_logits_final": torch.zeros(2, 4),
        "reason_logits_final": torch.zeros(2, 21),
        "scalar_diagnostic": torch.zeros(()),
    }
    mirrored = {
        key: value[:1].clone()
        for key, value in output.items()
        if isinstance(value, torch.Tensor) and value.ndim > 0
    }
    targets = {
        "factor_source_weight": torch.ones(2, 21),
        "factor_anchor_map": torch.full((2, 21, 8), 1 / 8),
        "factor_anchor_valid": torch.ones(2, 21, dtype=torch.bool),
    }

    _, mirror = discrimination_and_mirror_loss(
        output, targets, mirrored_output=mirrored
    )
    assert torch.isfinite(mirror)
