from __future__ import annotations

import inspect
import re

import torch

from fate_oia.engine.audit_acpr_vista_gates import _gate_c, _gate_c1


def test_gate_c_runs_real_adapter_mechanism_check():
    payload = _gate_c(torch.device("cpu"), samples=16)
    assert "initial_loss" in payload
    assert "final_loss" in payload
    assert payload["final_loss"] < payload["initial_loss"]


def test_gate_c1_uses_plan_scale_adapter_not_tiny_proxy():
    src = inspect.getsource(_gate_c1)
    dim_match = re.search(r"dim\s*=\s*(\d+)", src)
    rank_match = re.search(r"rank\s*=\s*(\d+)", src)
    assert dim_match is not None
    assert rank_match is not None
    assert int(dim_match.group(1)) >= 128
    assert int(rank_match.group(1)) >= 16
    assert "grid_hw=(8, 8)" not in src
