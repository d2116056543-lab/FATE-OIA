import torch

from fate_oia.models.dice_directional_head import DICEDirectionalHead


def test_directional_head_has_no_bias_and_zero_init_has_zero_net_correction():
    head = DICEDirectionalHead(dim=8, action_dim=4, probes_per_action=4)
    assert not hasattr(head, "support_bias") and not hasattr(head, "counter_bias")
    out = head(torch.randn(3, 4, 4, 8), torch.zeros(3, 4), torch.zeros(3, 4, 4))
    assert torch.equal(out["dice_action_delta"], torch.zeros_like(out["dice_action_delta"]))
    assert out["atom_correction"].abs().max() <= 0.080001
