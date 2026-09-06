from __future__ import annotations

import numpy as np


def second_level_signature(increments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independent NumPy oracle for S1/S2 piecewise-linear signatures."""
    s1 = np.zeros(increments.shape[-1], dtype=np.float64)
    s2 = np.zeros((increments.shape[-1], increments.shape[-1]), dtype=np.float64)
    for delta in np.asarray(increments, dtype=np.float64):
        s2 += np.outer(s1, delta) + .5 * np.outer(delta, delta)
        s1 += delta
    return s1, s2


def directed_area(points: np.ndarray) -> float:
    _, s2 = second_level_signature(np.diff(np.asarray(points, dtype=np.float64), axis=0))
    return float(.5 * (s2[0, 1] - s2[1, 0]))


def product_integral(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    total = 0.0
    for k in range(len(t) - 1):
        dt, dx, dy = t[k + 1] - t[k], x[k + 1] - x[k], y[k + 1] - y[k]
        total += dt * (x[k] * y[k] + .5 * (x[k] * dy + y[k] * dx) + dx * dy / 3.)
    return float(total)
