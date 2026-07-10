from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

from fate_oia.engine.export_mosaic_visual_audit import run_visual_audit


class _FakeVisualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.schema_bundle = {
            "factors": [{"name": "left_object"}, {"name": "right_object"}],
        }

    def forward(self, images, *, prior_mode="full", return_masks=False):
        batch = images.shape[0]
        factor = torch.tensor([[0.8, 0.3]], device=images.device).expand(batch, -1)
        if prior_mode == "content_only":
            factor = factor * 0.9
        elif prior_mode == "prior_only":
            factor = factor * 0.2
        left = torch.linspace(1.0, 0.0, 80, device=images.device).view(1, 1, 1, 80).expand(batch, 1, 45, 80)
        right = left.flip(-1)
        masks = torch.cat((left, right), dim=1)
        return {
            "factor_presence_prob": factor,
            "factor_visibility_prob": factor,
            "factor_soft_masks": masks,
            "prior_scale": torch.tensor([0.05, 0.05], device=images.device),
            "measurement_stats": {
                "dominant_prototype_rate": torch.tensor([0.2, 0.3], device=images.device),
                "prototype_effective_count": torch.tensor([2.0, 2.2], device=images.device),
                "dead_prototype_count": torch.tensor([0.0, 0.0], device=images.device),
            },
        }


class _GroundingBuilder:
    def __call__(self, reasons, records, *, split):
        batch = reasons.shape[0]
        geometry = torch.zeros(batch, 2, 45, 80)
        geometry[:, 0, :, :40] = 1
        return {
            "presence_target": torch.ones(batch, 2),
            "presence_mask": torch.ones(batch, 2),
            "visibility_target": torch.ones(batch, 2),
            "visibility_mask": torch.ones(batch, 2),
            "source_reliability": torch.ones(batch, 2),
            "geometry_mask": geometry,
            "geometry_mask_valid": torch.ones(batch, 2),
        }


class _Index:
    def lookup(self, file_name):
        return SimpleNamespace(label_json=None, drivable_map=None)


class _Loader(list):
    @property
    def dataset(self):
        return [0, 1]


def test_visual_audit_populates_every_required_directory(tmp_path) -> None:
    image_paths = []
    for index in range(2):
        path = tmp_path / f"image_{index}.jpg"
        Image.new("RGB", (64, 36), (30 + index * 20, 40, 50)).save(path)
        image_paths.append(str(path))
    loader = _Loader(
        [
            {
                "image": torch.randn(2, 3, 36, 64),
                "reason": torch.zeros(2, 21),
                "file_name": ["a.jpg", "b.jpg"],
                "image_path": image_paths,
            }
        ]
    )
    summary = run_visual_audit(
        _FakeVisualModel(), loader, _GroundingBuilder(), _Index(), torch.device("cpu"), tmp_path / "audit"
    )
    for directory in (
        "factor_attention_overlays", "factor_content_only", "factor_prior_only",
        "factor_query_shuffle", "factor_image_shuffle", "left_right_flip", "geometry_alignment",
    ):
        assert any((tmp_path / "audit" / directory).iterdir()), directory
    assert summary["geometry_is_forward_input"] is False
    assert (tmp_path / "audit" / "summary.json").exists()
