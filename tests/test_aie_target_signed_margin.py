import torch

from fate_oia.utils.aie_counterfactual import target_signed_margin


def test_target_signed_margin_handles_positive_and_negative_targets():
    logits = torch.tensor([[2.0, 2.0]])
    targets = torch.tensor([[1.0, 0.0]])
    torch.testing.assert_close(target_signed_margin(logits, targets), torch.tensor([[2.0, -2.0]]))


