import torch

from fate_oia.engine.train_acpr_meter_oia import _apply_pu_gate
from fate_oia.utils.meter_config import load_meter_config


def _audit() -> dict:
    return {
        "labels": [{"eligible": True, "auprc_delta": 0.03} for _ in range(21)],
        "lambda": [0.08] * 21,
    }


def test_pu_requires_train_audit_view_gate_and_two_epoch_streak() -> None:
    config = load_meter_config("configs/fate_oia_train_360x640_acpr_meter_oia_v3_heca.yaml")
    first = _apply_pu_gate(_audit(), torch.full((21,), 0.8), config, {"pass_streak": [0] * 21}, epoch=1)
    assert first["active_labels"] == []
    second = _apply_pu_gate(_audit(), torch.full((21,), 0.8), config, first, epoch=2)
    assert second["active_labels"] == list(range(21))
    failed_view = _apply_pu_gate(_audit(), torch.zeros(21), config, second, epoch=3)
    assert failed_view["active_labels"] == []
