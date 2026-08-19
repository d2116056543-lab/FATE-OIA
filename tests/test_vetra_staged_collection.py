import torch
from torch import nn

from fate_oia.engine.collect_vetra_tta_outputs import apply_optional_refiner


class DummyRefiner(nn.Module):
    def forward(self, source, action_scale, gain=None):
        delta = torch.full_like(source["action_logits_final"], 0.02)
        return {
            "action_logits_final": source["action_logits_final"] + delta,
            "reason_logits_final": source["reason_logits_final"],
            "action_delta": delta,
        }


def _source():
    return {
        "action_logits_primary": torch.randn(3, 4),
        "action_logits_final": torch.randn(3, 4),
        "reason_logits_primary": torch.randn(3, 21),
        "reason_logits_final": torch.randn(3, 21),
    }


def test_no_refiner_returns_exact_base_outputs():
    source = _source()
    output = apply_optional_refiner(source, None, action_scale=1.0)
    assert output["action_logits_final"].data_ptr() == source["action_logits_final"].data_ptr()
    assert output["reason_logits_final"].data_ptr() == source["reason_logits_final"].data_ptr()
    assert torch.count_nonzero(output["action_delta"]) == 0


def test_selected_refiner_changes_action_but_preserves_reason_exactly():
    source = _source()
    output = apply_optional_refiner(source, DummyRefiner(), action_scale=1.0)
    assert torch.count_nonzero(output["action_delta"]) > 0
    assert output["reason_logits_final"].data_ptr() == source["reason_logits_final"].data_ptr()
    torch.testing.assert_close(
        output["action_logits_final"], source["action_logits_final"] + 0.02
    )
