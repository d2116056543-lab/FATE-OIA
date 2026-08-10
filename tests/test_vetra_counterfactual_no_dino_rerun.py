import torch

from fate_oia.utils.vetra_counterfactual_audit import VETRACounterfactualAudit
from vetra_test_utils import build_model, fake_base


def test_counterfactual_audit_operates_in_projected_space_without_base_forward():
    model = build_model()
    output = model.decode_base_output(fake_base(batch=2), alpha=1.0)
    result = VETRACounterfactualAudit().run(model, output, torch.zeros(2, 4), max_samples=2)
    assert model.base_model.forward_calls == 0
    assert result["dino_rerun_count"] == 0
