from pathlib import Path

import torch
import yaml

from fate_oia.models.precise_evidence_fields import PRECISEEvidenceFields


def test_view_consistency_update_does_not_invalidate_reliability_graph():
    config_path = Path(__file__).parents[1] / "configs" / "precise_evidence_fields.yaml"
    fields = yaml.safe_load(config_path.read_text(encoding="utf-8"))["explicit_fields"]
    model = PRECISEEvidenceFields(fields, dim=384, grid_hw=(45, 80))
    evidence_layers = torch.randn(1, 3, 3600, 384, requires_grad=True)
    output = model(evidence_layers)
    loss = output["reliability"].mean() + output["certificate_probability"].mean()

    # The trainer updates this EMA after the forward and before target-credit
    # gradients. This must not invalidate the graph that consumed reliability.
    model.update_view_consistency(
        output["explicit_tokens"].detach(),
        output["explicit_tokens"].detach(),
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    assert any(gradient is not None for gradient in gradients)
