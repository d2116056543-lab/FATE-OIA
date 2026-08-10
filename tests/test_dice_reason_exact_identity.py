import torch
from torch import nn

from fate_oia.models.dice_oia_model import DICEOIAModel


class FakeBase(nn.Module):
    def forward(self, images, **kwargs):
        b, d, n, p = images.shape[0], 8, 6, 5
        return {"evidence_token": torch.randn(b,4,4,d), "conditioned_patch_layers": torch.randn(b,2,n,d),
                "predicate_attention": torch.softmax(torch.randn(b,p,n),-1), "predicate_probs": torch.rand(b,p),
                "ego_region_masks": {k: torch.rand(n) for k in ("front_center","left_corridor","right_corridor","upper_traffic_region","bottom_drivable_region")},
                "action_logits_final": torch.randn(b,4), "reason_logits_final": torch.randn(b,21),
                "bounded_contribution": torch.zeros(b,4,4)}


def test_reason_logits_are_bitwise_identical():
    model = DICEOIAModel(FakeBase(), dim=8, num_layers=2, num_predicates=5)
    out = model(torch.randn(2,3,4,4))
    assert torch.equal(out["reason_logits_final"], out["reason_logits_base"])
    assert out["reason_identity_max_abs"].item() == 0
