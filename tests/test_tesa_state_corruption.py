import torch

from fate_oia.engine.tesa_diagnostics import corrupt_state_keep_anchor


def test_state_corruption_keeps_anchor() -> None:
    anchor = torch.randn(2, 21, 8)
    state = torch.randn(2, 21, 3).softmax(-1)
    new_anchor, new_state = corrupt_state_keep_anchor(anchor, state)
    torch.testing.assert_close(new_anchor, anchor)
    assert not torch.equal(new_state, state)
