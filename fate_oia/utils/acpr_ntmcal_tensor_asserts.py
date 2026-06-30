from __future__ import annotations

import torch


def assert_shape(t: torch.Tensor, shape: tuple[int | None, ...], name: str) -> None:
    if t.ndim != len(shape):
        raise AssertionError(f"{name} ndim {t.ndim} != {len(shape)}")
    for got, exp in zip(t.shape, shape):
        if exp is not None and int(got) != int(exp):
            raise AssertionError(f"{name} shape {tuple(t.shape)} incompatible with {shape}")


def assert_deploy_equation(logits: torch.Tensor, theta: torch.Tensor, deploy: torch.Tensor, name: str, tol: float = 1e-6) -> None:
    err = (deploy - (logits - theta)).abs().max().item()
    if err > tol:
        raise AssertionError(f"{name} deploy equation error {err}")
