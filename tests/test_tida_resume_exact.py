import random
from copy import deepcopy
from itertools import islice

import numpy as np
import torch

from fate_oia.utils.tida_artifacts import capture_rng_state, restore_rng_state
from fate_oia.utils.tida_stateful_sampler import TIDAStatefulRandomSampler


def test_rng_state_restores_python_numpy_and_torch_exactly():
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert expected[0] == actual[0] and expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


def test_stateful_sampler_resume_matches_uninterrupted_sequence_exactly():
    full = TIDAStatefulRandomSampler(range(11), seed=17)
    full_values = list(full)

    interrupted = TIDAStatefulRandomSampler(range(11), seed=17)
    prefix = list(interrupted)[:4]
    interrupted.mark_consumed(4)
    state = interrupted.state_dict()

    resumed = TIDAStatefulRandomSampler(range(11), seed=17)
    resumed.load_state_dict(state)
    assert prefix + list(resumed) == full_values


def test_sampler_checkpoint_tracks_consumed_not_prefetched_items():
    sampler = TIDAStatefulRandomSampler(range(9), seed=3)
    iterator = iter(sampler)
    prefetched = [next(iterator) for _ in range(6)]
    sampler.mark_consumed(2)
    resumed = TIDAStatefulRandomSampler(range(9), seed=3)
    resumed.load_state_dict(sampler.state_dict())
    assert list(resumed)[0] == prefetched[2]


def _train_updates(model, optimizer, sampler, count):
    for index, augmentation_seed in islice(iter(sampler), count):
        value = torch.tensor([[index / 10.0, (augmentation_seed % 19) / 19.0]])
        loss = model(value).square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sampler.mark_consumed(1)


def test_four_updates_equal_two_save_resume_two_for_parameters_and_optimizer():
    torch.manual_seed(29)
    initial = torch.nn.Linear(2, 1)
    initial_state = deepcopy(initial.state_dict())

    continuous = torch.nn.Linear(2, 1); continuous.load_state_dict(initial_state)
    continuous_optimizer = torch.optim.AdamW(continuous.parameters(), lr=0.01)
    continuous_sampler = TIDAStatefulRandomSampler(range(8), seed=31)
    _train_updates(continuous, continuous_optimizer, continuous_sampler, 4)

    interrupted = torch.nn.Linear(2, 1); interrupted.load_state_dict(initial_state)
    interrupted_optimizer = torch.optim.AdamW(interrupted.parameters(), lr=0.01)
    interrupted_sampler = TIDAStatefulRandomSampler(range(8), seed=31)
    _train_updates(interrupted, interrupted_optimizer, interrupted_sampler, 2)
    checkpoint = {
        "model": deepcopy(interrupted.state_dict()),
        "optimizer": deepcopy(interrupted_optimizer.state_dict()),
        "sampler": deepcopy(interrupted_sampler.state_dict()),
        "scheduler_update": 2,
    }

    resumed = torch.nn.Linear(2, 1); resumed.load_state_dict(checkpoint["model"])
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_sampler = TIDAStatefulRandomSampler(range(8), seed=31)
    resumed_sampler.load_state_dict(checkpoint["sampler"])
    _train_updates(resumed, resumed_optimizer, resumed_sampler, 2)

    for left, right in zip(continuous.parameters(), resumed.parameters()):
        assert torch.equal(left, right)
    assert continuous_sampler.state_dict() == resumed_sampler.state_dict()
    assert checkpoint["scheduler_update"] + 2 == 4
    continuous_state = continuous_optimizer.state_dict()
    resumed_state = resumed_optimizer.state_dict()
    assert continuous_state["param_groups"] == resumed_state["param_groups"]
    for key, value in continuous_state["state"].items():
        for field, tensor in value.items():
            other = resumed_state["state"][key][field]
            assert torch.equal(tensor, other) if torch.is_tensor(tensor) else tensor == other
