import random
import numpy as np
import torch

from fate_oia.utils.aie_artifacts import capture_rng_state, restore_rng_state


def test_rng_state_roundtrip_is_exact():
    random.seed(9); np.random.seed(9); torch.manual_seed(9); state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), torch.rand(3))
    assert expected[:2] == actual[:2]; torch.testing.assert_close(expected[2], actual[2])

