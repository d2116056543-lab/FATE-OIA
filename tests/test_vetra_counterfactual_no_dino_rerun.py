import torch

from fate_oia.utils.vetra_counterfactual_audit import VETRACounterfactualAudit
from vetra_test_utils import build_model, fake_base


def test_counterfactual_audit_operates_in_projected_space_without_base_forward():
    model = build_model()
    output = model.decode_base_output(fake_base(batch=2), alpha=1.0)
    result = VETRACounterfactualAudit().run(model, output, torch.zeros(2, 4), max_samples=2)
    assert model.base_model.forward_calls == 0
    assert result["dino_rerun_count"] == 0


def test_counterfactual_donor_distance_uses_full_vector_geometry():
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.0]])
    distance = VETRACounterfactualAudit._donor_distances(values, selected=0, null_index=2)
    assert torch.isinf(distance[0])
    assert torch.isinf(distance[2])
    assert torch.allclose(distance[1], torch.tensor(2.0).sqrt())
