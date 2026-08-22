import torch

from fate_oia.engine.train_tida_oia import checkpoint_trainable_state


def test_checkpoint_view_selects_requested_state():
    payload = {
        "tida_trainable_state": {"weight": torch.tensor([1.0])},
        "ema": {"weight": torch.tensor([2.0])},
    }
    assert checkpoint_trainable_state(payload, "online")["weight"].item() == 1.0
    assert checkpoint_trainable_state(payload, "ema")["weight"].item() == 2.0
