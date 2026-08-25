from torch.utils.data import TensorDataset
import torch

from fate_oia.engine.train_tida_oia import _partition_sample_limit, make_loader


def _config():
    return {
        "data": {
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        }
    }


def test_only_training_loader_keeps_persistent_workers():
    dataset = TensorDataset(torch.zeros(8, 1))
    train = make_loader(dataset, 2, True, 2, _config(), partition="train_core")
    evaluation = make_loader(dataset, 2, False, 2, _config(), partition="test")

    assert train.persistent_workers is True
    assert evaluation.persistent_workers is False


def test_train_and_evaluation_sample_limits_are_independent():
    assert _partition_sample_limit("train_core", 1024, 256) == 1024
    assert _partition_sample_limit("test", 1024, 256) == 256
    assert _partition_sample_limit("expanded_test", 1024, None) is None
