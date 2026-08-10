from __future__ import annotations

import torch


def assert_probe_contract(output: dict) -> None:
    if output["action_logits_final"].shape[-1] != 4 or output["reason_logits_final"].shape[-1] != 21:
        raise AssertionError("DICE label dimensions are invalid")
    if not torch.equal(output["reason_logits_final"], output["reason_logits_base"]):
        raise AssertionError("DICE changed formal explanation logits")
    if float(output["dice_action_delta"].abs().max()) > .250001:
        raise AssertionError("DICE action cap violated")
    if int(output["predicate_top2_count"].max()) > 2:
        raise AssertionError("DICE predicate mixture is not Top-2")


def assert_base_frozen(model) -> None:
    if any(parameter.requires_grad for parameter in model.base_model.parameters()):
        raise AssertionError("DICE probe base is trainable")
