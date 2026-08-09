from __future__ import annotations

import numpy as np


def paired_bootstrap(control: dict[str, np.ndarray], method: dict[str, np.ndarray], metric_fn,
                     resamples: int = 2000, seed: int = 20260809) -> dict:
    count = next(iter(control.values())).shape[0]
    if any(value.shape[0] != count for value in (*control.values(), *method.values())):
        raise ValueError("paired bootstrap sample counts differ")
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(int(resamples)):
        index = rng.integers(0, count, count)
        c = metric_fn({key: value[index] for key, value in control.items()})
        m = metric_fn({key: value[index] for key, value in method.items()})
        rows.append({key: float(m[key] - c[key]) for key in c})
    return {key: {"mean": float(np.mean([row[key] for row in rows])),
                  "p2_5": float(np.percentile([row[key] for row in rows], 2.5)),
                  "p50": float(np.percentile([row[key] for row in rows], 50)),
                  "p97_5": float(np.percentile([row[key] for row in rows], 97.5)),
                  "valid_replicates": len(rows)} for key in rows[0]}
