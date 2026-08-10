from __future__ import annotations

import torch


def assert_vetra_contract(output: dict) -> None:
    required = {
        "action_logits_base": (4,), "action_logits_final": (4,),
        "reason_logits_base": (21,), "reason_logits_final": (21,),
        "vetra_action_delta": (4,),
    }
    for key, tail in required.items():
        if key not in output or tuple(output[key].shape[1:]) != tail:
            raise AssertionError(f"VETRA output contract failed for {key}")
        if not torch.isfinite(output[key]).all():
            raise AssertionError(f"VETRA output contains nonfinite values: {key}")
    if not torch.equal(output["reason_logits_base"], output["reason_logits_final"]):
        raise AssertionError("VETRA modified formal reason logits")
    if float(output["vetra_action_delta"].abs().max()) > .200001:
        raise AssertionError("VETRA action correction exceeded cap")


def assert_base_frozen(model, expected_hash: str, hash_fn) -> None:
    if hash_fn(model.base_model) != expected_hash:
        raise AssertionError("VETRA mutated the frozen AIE base")
    if any(parameter.grad is not None and bool((parameter.grad != 0).any())
           for parameter in model.base_model.parameters()):
        raise AssertionError("VETRA leaked gradients into frozen AIE base")
