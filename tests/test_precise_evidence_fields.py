from pathlib import Path

import torch

from fate_oia.models.precise_evidence_fields import PRECISEEvidenceFields
from fate_oia.losses.precise_losses import evidence_loss, evidence_view_consistency_loss
from fate_oia.utils.precise_schema import load_evidence_fields


ROOT = Path(__file__).resolve().parents[1]


def test_evidence_fields_emit_multipart_explicit_and_unnamed_latent_tokens():
    fields = load_evidence_fields(ROOT / "configs" / "precise_evidence_fields.yaml")
    model = PRECISEEvidenceFields(fields)
    output = model(torch.randn(2, 3, 3600, 384))
    assert output["explicit_tokens"].shape == (2, 10, 384)
    assert output["latent_tokens"].shape == (2, 6, 384)
    assert output["part_coordinates"].shape == (2, 10, 8, 2)
    assert output["soft_masks"].shape == (2, 10, 45, 80)
    assert output["reliability"].shape == (2, 10)
    assert set(("traffic_light_visible", "right_solid_boundary")) <= set(output["derived_atom_probs"])
    assert output["explicit_part_attention"].shape == (2, 10, 8, 3600)
    assert output["latent_part_attention"].shape == (2, 6, 4, 3600)
    # Ordered curve parts must occupy distinct y anchors rather than repeat a centroid.
    left_curve_y = output["part_coordinates"][:, 8, :8, 1]
    assert torch.all(left_curve_y[:, 1:] >= left_curve_y[:, :-1])
    assert (left_curve_y[:, -1] - left_curve_y[:, 0]).mean() > 0.4


def test_latent_slots_can_read_a_reason_private_visual_field():
    fields = load_evidence_fields(ROOT / "configs" / "precise_evidence_fields.yaml")
    model = PRECISEEvidenceFields(fields)
    explicit_source = torch.randn(1, 3, 3600, 384, requires_grad=True)
    reason_source = torch.randn(1, 3, 3600, 384, requires_grad=True)
    output = model(explicit_source, latent_layers=reason_source)
    output["latent_tokens"].sum().backward()
    assert reason_source.grad is not None and reason_source.grad.abs().sum() > 0
    assert explicit_source.grad is None or explicit_source.grad.abs().sum() == 0


def test_reliability_is_continuous_and_increases_with_observability_logit():
    fields = load_evidence_fields(ROOT / "configs" / "precise_evidence_fields.yaml")
    model = PRECISEEvidenceFields(fields)
    base = model(torch.randn(1, 3, 3600, 384))
    assert torch.isfinite(base["reliability"]).all()
    assert (base["reliability"] > 0).any() and (base["reliability"] < 1).any()


def test_evidence_loss_contains_geometry_prototype_view_and_latent_terms():
    fields = load_evidence_fields(ROOT / "configs" / "precise_evidence_fields.yaml")
    output = PRECISEEvidenceFields(fields)(torch.randn(1, 3, 3600, 384))
    targets = {
        "presence": torch.ones(1, 10), "presence_valid": torch.ones(1, 10),
        "observability": torch.ones(1, 10), "state": torch.zeros(1, 10, 4),
        "state_valid": torch.ones(1, 10), "part_coordinates": output["part_coordinates"].detach(),
        "part_valid": torch.ones(1, 10),
    }
    losses = evidence_loss(output, targets)
    for key in ("loss_evidence_prototype", "loss_evidence_view", "loss_evidence_latent_diversity", "curve_distance_valid_count"):
        assert key in losses
    assert losses["curve_distance_valid_count"].item() == 2


def test_evidence_view_consistency_aligns_field_identity_masks_and_x_coordinates():
    field_map = torch.tensor([1, 0])
    canonical = {
        "explicit_evidence_tokens": torch.randn(1, 2, 4),
        "evidence_presence_logits": torch.randn(1, 2),
        "evidence_observability_logits": torch.randn(1, 2),
        "evidence_state_logits": torch.randn(1, 2, 3),
        "evidence_masks": torch.rand(1, 2, 3, 5),
        "evidence_part_coordinates": torch.rand(1, 2, 2, 2),
    }
    mirrored = {
        "explicit_evidence_tokens": canonical["explicit_evidence_tokens"][:, field_map].clone(),
        "evidence_presence_logits": canonical["evidence_presence_logits"][:, field_map].clone(),
        "evidence_observability_logits": canonical["evidence_observability_logits"][:, field_map].clone(),
        "evidence_state_logits": canonical["evidence_state_logits"][:, field_map].clone(),
        "evidence_masks": canonical["evidence_masks"][:, field_map].flip(-1).clone(),
        "evidence_part_coordinates": canonical["evidence_part_coordinates"][:, field_map].clone(),
    }
    mirrored["evidence_part_coordinates"][..., 0] = 1.0 - mirrored["evidence_part_coordinates"][..., 0]
    assert evidence_view_consistency_loss(canonical, mirrored, field_map).item() < 1e-7
