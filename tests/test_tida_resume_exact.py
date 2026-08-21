import random

import numpy as np
import torch

from fate_oia.utils.tida_artifacts import capture_rng_state, restore_rng_state


def test_rng_state_restores_python_numpy_and_torch_exactly():
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert expected[0] == actual[0] and expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])
