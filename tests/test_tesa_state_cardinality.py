import torch

from fate_oia.models.meter_signed_factors import TypedEvidenceStateHead
from tesa_helpers import typed_inputs


def test_state_head_masks_factor_specific_padding() -> None:
    nodes, patches = typed_inputs()
    head = TypedEvidenceStateHead(dim=32, state_cardinalities=(2,) + (3,) * 20)
    out = head(nodes, patches)
    assert torch.isneginf(out["factor_state_logits"][:, 0, 2]).all()
    assert out["factor_state_prob"][:, 0, 2].eq(0).all()
