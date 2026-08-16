import torch
from torch.utils.data import TensorDataset

from fate_oia.engine.train_aie_oia import aie_worker_init_fn, make_loader


def _config() -> dict:
    return {
        "data": {
            "pin_memory": False,
            "persistent_workers": True,
            "prefetch_factor": 2,
        }
    }


def test_loader_persistence_can_be_disabled_for_transient_eval_loaders() -> None:
    dataset = TensorDataset(torch.arange(4))

    train_loader = make_loader(
        dataset, 2, False, 1, _config(), persistent_workers=True
    )
    eval_loader = make_loader(
        dataset, 2, False, 1, _config(), persistent_workers=False
    )

    assert train_loader.persistent_workers is True
    assert eval_loader.persistent_workers is False


def test_worker_init_limits_pytorch_cpu_threads() -> None:
    original_threads = torch.get_num_threads()
    try:
        aie_worker_init_fn(0)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(original_threads)
