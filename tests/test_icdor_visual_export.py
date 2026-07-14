from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from fate_oia.engine.export_mosaic_trust_visual_audit import ICDORVisualExportError, export_visual_audit


class _EvidenceModel(nn.Module):
    def forward(self, images: torch.Tensor, *, return_masks: bool, **_: object) -> dict[str, torch.Tensor]:
        assert return_masks is True
        value = images.mean((1, 2, 3), keepdim=False).unsqueeze(1)
        mask = torch.sigmoid(images[:, :1])
        return {
            "action_final_logits": value,
            "reason_observed_logits": value + 1.0,
            "factor_soft_masks": mask,
            "action_support_mask": mask,
            "action_veto_mask": 1.0 - mask,
            "support_weights": value.unsqueeze(-1),
            "veto_weights": (value + 2.0).unsqueeze(-1),
        }


def test_visual_export_serializes_actual_logits_masks_and_edges(tmp_path) -> None:
    images = torch.stack((torch.zeros(3, 4, 4), torch.ones(3, 4, 4)))
    result = export_visual_audit(
        _EvidenceModel(),
        [{"image": images, "file_name": ["black.jpg", "white.jpg"], "split": ["train_audit", "train_audit"]}],
        tmp_path,
        device=torch.device("cpu"),
    )

    manifest = json.loads((tmp_path / "visual_audit_manifest.json").read_text(encoding="utf-8"))
    assert result["sample_count"] == 2
    assert manifest["samples"][0]["action_final_logits"] == [0.0]
    assert manifest["samples"][1]["reason_observed_logits"] == [2.0]
    assert manifest["samples"][1]["support_weights"] == [[1.0]]
    assert (tmp_path / "masks" / "0000_factor_00.pt").exists()
    assert manifest["fixed_sample_ids"] == ["black.jpg", "white.jpg"]
    assert manifest["samples"][0]["matched_random_factor_mask_files"]
    random_path = tmp_path / manifest["samples"][0]["matched_random_factor_mask_files"][0]
    assert random_path.exists()
    original = torch.load(tmp_path / manifest["samples"][0]["factor_mask_files"][0], weights_only=True)
    matched_random = torch.load(random_path, weights_only=True)
    expected = torch.roll(
        original,
        shifts=(original.shape[-2] // 3, original.shape[-1] // 3),
        dims=(-2, -1),
    )
    assert torch.equal(matched_random, expected)
    assert torch.isclose(original.sum(), matched_random.sum())


def test_visual_export_uses_a_fixed_bounded_case_set(tmp_path) -> None:
    images = torch.arange(4.0).view(4, 1, 1, 1).expand(-1, 3, 4, 4)
    result = export_visual_audit(
        _EvidenceModel(),
        [{"image": images, "file_name": [f"{i}.jpg" for i in range(4)], "split": ["train_audit"] * 4}],
        tmp_path,
        device=torch.device("cpu"),
        max_samples=2,
    )
    assert result["sample_count"] == 2


def test_visual_export_rejects_missing_real_edge_tensors(tmp_path) -> None:
    class MissingVeto(_EvidenceModel):
        def forward(self, images: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
            output = super().forward(images, **kwargs)
            output.pop("veto_weights")
            return output

    with pytest.raises(ICDORVisualExportError, match="veto_weights"):
        export_visual_audit(
            MissingVeto(),
            [{"image": torch.ones(1, 3, 4, 4), "file_name": ["only.jpg"], "split": ["train_audit"]}],
            tmp_path,
            device=torch.device("cpu"),
        )

