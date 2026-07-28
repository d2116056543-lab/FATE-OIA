import torch

from fate_oia.engine.eval_acpr_meter_oia import (
    ACTION_BRANCHES,
    REASON_BRANCHES,
    collect_outputs,
)


class _CountingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decode_calls = 0

    def encode_images(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"batch": torch.tensor(images.shape[0])}

    def decode_from_field(
        self,
        field: dict[str, torch.Tensor],
        *,
        progress: float,
        diagnostic_modes: tuple[str, ...],
    ) -> dict[str, torch.Tensor]:
        self.decode_calls += 1
        batch = int(field["batch"])
        action = torch.zeros(batch, 4)
        reason = torch.zeros(batch, 21)
        factor_map = torch.full((batch, 21, 8), 1.0 / 9.0)
        factor_null = torch.full((batch, 21), 1.0 / 9.0)
        return {
            "action_logits_visual": action,
            "action_logits_semantic": action,
            "action_logits_peer": action,
            "action_logits_final": action,
            "reason_logits_calalign": reason,
            "reason_logits_global": reason,
            "reason_logits_local": reason,
            "reason_logits_mix": reason,
            "reason_logits_final": reason,
            "factor_support_map": factor_map,
            "factor_counter_map": factor_map,
            "factor_support_null": factor_null,
            "factor_counter_null": factor_null,
            "factor_support_score": factor_null,
            "factor_counter_score": factor_null,
            "factor_layer_weights": torch.full((21, 3), 1.0 / 3.0),
            "factor_reliability": factor_null,
            "action_selector": torch.full((batch, 4), 0.5),
            "action_factor_contributions": torch.zeros(batch, 4, 21),
            "semantic_bias": action,
            "reason_mix_gate": torch.full((batch, 21), 0.5),
            "reason_annotation_delta": reason,
        }


def test_collect_outputs_decodes_each_unique_mode_once_per_batch() -> None:
    model = _CountingModel()
    batch = {
        "image": torch.zeros(2, 3, 360, 640),
        "action": torch.zeros(2, 4),
        "reason": torch.zeros(2, 21),
        "file_name": ["a.jpg", "b.jpg"],
    }

    collect_outputs(model, [batch], torch.device("cpu"), progress=1.0)

    unique_modes = set(ACTION_BRANCHES.values()) | set(REASON_BRANCHES.values())
    assert model.decode_calls == len(unique_modes)
