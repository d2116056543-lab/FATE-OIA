import random

import numpy as np
import torch

from fate_oia.utils.tida_artifacts import seed_tida_run


def _draw():
    return random.random(), float(np.random.rand()), torch.rand(4)


def test_fresh_run_seed_repeats_python_numpy_torch_and_cuda_rng():
    seed_tida_run(20260821)
    expected = _draw()
    seed_tida_run(20260821)
    actual = _draw()

    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])
