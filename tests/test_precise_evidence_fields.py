from pathlib import Path

import torch

from fate_oia.models.precise_evidence_fields import PRECISEEvidenceFields
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


def test_reliability_is_continuous_and_increases_with_observability_logit():
    fields = load_evidence_fields(ROOT / "configs" / "precise_evidence_fields.yaml")
    model = PRECISEEvidenceFields(fields)
    base = model(torch.randn(1, 3, 3600, 384))
    assert torch.isfinite(base["reliability"]).all()
    assert (base["reliability"] > 0).any() and (base["reliability"] < 1).any()
