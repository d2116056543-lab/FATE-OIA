from __future__ import annotations

import torch

from .tida_reason_local_temporal_query import TIDAReasonLocalTemporalQuery


class TIDAActionLocalTemporalQuery(TIDAReasonLocalTemporalQuery):
    """Action-private ordered history reader with strict image fallback.

    The implementation intentionally shares the proven local temporal machinery
    with the reason reader while exposing an action-specific public contract.
    No reason logits or reason targets enter this branch.
    """

    def __init__(
        self,
        dim: int = 384,
        num_actions: int = 4,
        num_heads: int = 4,
        cap: float = 0.08,
        utility_open_prior: float = 0.10,
    ) -> None:
        super().__init__(
            dim=dim,
            num_reasons=num_actions,
            num_heads=num_heads,
            cap=cap,
            utility_open_prior=utility_open_prior,
            temporal_reason_indices=tuple(range(num_actions)),
        )
        self.num_actions = int(num_actions)

    @property
    def action_readout_weight(self) -> torch.nn.Parameter:
        return self.reason_readout_weight

    def forward(
        self,
        history_action_tokens: torch.Tensor,
        target_action_tokens: torch.Tensor,
        timestamps: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        *,
        image_logits: torch.Tensor,
        temporal_scale: float | torch.Tensor = 1.0,
    ) -> dict[str, torch.Tensor | str]:
        reason_output = super().forward(
            history_action_tokens,
            target_action_tokens,
            timestamps,
            frame_valid_mask,
            image_logits=image_logits,
            temporal_scale=temporal_scale,
        )
        output: dict[str, torch.Tensor | str] = {}
        for key, value in reason_output.items():
            if key.startswith("reason_local_"):
                key = "action_local_" + key[len("reason_local_") :]
            output[key] = value
        output["action_local_temporal_action_mask"] = output.pop(
            "action_local_temporal_reason_mask"
        )
        return output
