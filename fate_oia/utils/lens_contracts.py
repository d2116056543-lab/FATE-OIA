from __future__ import annotations

import torch


def assert_lens_forward_contract(out: dict[str, torch.Tensor], batch_size: int) -> None:
    expected = {
        "action_logits_source": (batch_size, 4), "action_logits_base": (batch_size, 4), "action_logits_final": (batch_size, 4),
        "reason_logits_source": (batch_size, 21), "reason_logits_formal": (batch_size, 21),
        "evidence_map": (batch_size, 21, 3600), "state_prob": (batch_size, 21, 3),
        "action_logits_state_substitution": (batch_size, 21, 3, 4),
    }
    for key, shape in expected.items():
        if key not in out or tuple(out[key].shape) != shape:
            raise AssertionError(f"{key} expected {shape}, got {None if key not in out else tuple(out[key].shape)}")
    if not torch.isfinite(out["action_logits_final"]).all():
        raise AssertionError("non-finite action logits")
