import torch

from fate_oia.utils.dice_counterfactual import directional_certificate


def test_certificate_uses_median_and_mad_of_all_four_controls():
    selected = torch.tensor([.5])
    out = directional_certificate(selected, torch.tensor([.1]), torch.tensor([.2]), torch.tensor([.3]), torch.tensor([.4]))
    assert torch.allclose(out["control_median"], torch.tensor([.2]))
    assert out["license_support_cf"].item() > out["license_counter_cf"].item()
