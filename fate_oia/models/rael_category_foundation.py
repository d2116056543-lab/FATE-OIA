"""P7 RAEL action-category foundation with a strict visual-only global branch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn


FORMAL_DIM = 384
FORMAL_ACTION_COUNT = 4
FORMAL_LAYER_COUNT = 4
ACTION_NAMES = ("forward", "stop", "left", "right")


@runtime_checkable
class MultiLayerFieldReader(Protocol):
    """The public P3 read contract consumed by this foundation."""

    def read(
        self,
        prepared: Mapping[str, Any],
        queries: Tensor,
        group_name: str | None = None,
    ) -> Mapping[str, Tensor | str | None]:
        ...


class ActionCategoryGlobalHead(nn.Module):
    """Action-wise W_A*A_tilde+b_A on the four post-attention tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(FORMAL_ACTION_COUNT, dim))
        self.bias = nn.Parameter(torch.empty(FORMAL_ACTION_COUNT))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, action_tokens: Tensor) -> Tensor:
        if action_tokens.ndim != 3 or action_tokens.shape[1:] != (FORMAL_ACTION_COUNT, self.weight.shape[1]):
            raise ValueError("global head requires post-attention action tokens [B,4,384]")
        return torch.einsum("bad,ad->ba", action_tokens, self.weight) + self.bias


class RAELActionCategoryFoundation(nn.Module):
    """Build four visual action categories from the shared P3 multi-layer field."""

    parameter_owner = "action_category"
    learning_rate = 2.0e-4

    def __init__(self, dim: int = FORMAL_DIM, num_heads: int = 6) -> None:
        super().__init__()
        if dim != FORMAL_DIM:
            raise ValueError(f"RAEL action foundation requires dim={FORMAL_DIM}, got {dim}")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("num_heads must divide the formal action embedding dimension")
        self.dim = int(dim)
        self.num_heads = int(num_heads)

        self.forward_query = nn.Parameter(torch.empty(self.dim))
        self.stop_query = nn.Parameter(torch.empty(self.dim))
        self.side_shared = nn.Parameter(torch.empty(self.dim))
        # One signed side direction keeps left/right tied around the same base.
        self.side_mirror_delta = nn.Parameter(torch.empty(self.dim))

        self.action_self_attention = nn.MultiheadAttention(
            self.dim,
            self.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.action_attention_norm = nn.LayerNorm(self.dim)
        # This is deliberately applied to the post-attention action tokens.
        self.global_head = ActionCategoryGlobalHead(self.dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in (
            self.forward_query,
            self.stop_query,
            self.side_shared,
            self.side_mirror_delta,
        ):
            nn.init.normal_(parameter, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.global_head.weight)
        nn.init.zeros_(self.global_head.bias)

    def owned_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.named_parameters())

    def _parameter_dtype(self) -> torch.dtype:
        return self.forward_query.dtype

    def query_components(self) -> dict[str, Tensor]:
        """Expose immutable-by-convention components for mirror diagnostics."""

        return {
            "forward": self.forward_query,
            "stop": self.stop_query,
            "side_shared": self.side_shared,
            "left_delta": self.side_mirror_delta,
            "right_delta": -self.side_mirror_delta,
        }

    def compositional_queries(self, batch_size: int) -> Tensor:
        """Return [forward, stop, side+delta, side-delta] for every sample."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        action_queries = torch.stack(
            (
                self.forward_query,
                self.stop_query,
                self.side_shared + self.side_mirror_delta,
                self.side_shared - self.side_mirror_delta,
            ),
            dim=0,
        )
        return action_queries.unsqueeze(0).expand(batch_size, -1, -1)

    def _validate_field_read(
        self,
        field_read: Mapping[str, Tensor | str | None],
        batch_size: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        readouts = field_read.get("readout")
        layer_weights = field_read.get("layer_weights")
        if not torch.is_tensor(readouts) or readouts.shape != (batch_size, FORMAL_ACTION_COUNT, self.dim):
            raise ValueError("P3 action readout must be [B,4,384]")
        if not torch.is_tensor(layer_weights) or layer_weights.shape != (
            batch_size,
            FORMAL_ACTION_COUNT,
            FORMAL_LAYER_COUNT,
        ):
            raise ValueError("P3 action layer_weights must be [B,4,4]")
        if readouts.device != device or layer_weights.device != device:
            raise ValueError("P3 field read must remain on the action-query device")
        if not torch.isfinite(readouts).all() or not torch.isfinite(layer_weights).all():
            raise ValueError("P3 action field read must be finite")
        if not torch.allclose(
            layer_weights.sum(dim=-1),
            torch.ones((batch_size, FORMAL_ACTION_COUNT), device=device, dtype=layer_weights.dtype),
            atol=1.0e-4,
            rtol=1.0e-4,
        ):
            raise ValueError("P3 action layer_weights must sum to one across all four layers")
        return readouts.to(dtype=self._parameter_dtype()), layer_weights.to(dtype=self._parameter_dtype())

    def project_global(self, action_tokens: Tensor) -> Tensor:
        """Project P7 visual tokens only; P8 owns the formal bridged symbol."""

        if not torch.is_tensor(action_tokens) or action_tokens.ndim != 3 or action_tokens.shape[1:] != (
            FORMAL_ACTION_COUNT,
            self.dim,
        ):
            raise ValueError("project_global requires action tokens [B,4,384]")
        global_visual = self.global_head(action_tokens)
        if not torch.is_tensor(global_visual) or global_visual.shape != (
            action_tokens.shape[0],
            FORMAL_ACTION_COUNT,
        ):
            raise RuntimeError("project_global must produce [B,4] without a trailing singleton dimension")
        if not torch.isfinite(global_visual).all():
            raise FloatingPointError("project_global produced non-finite visual logits")
        return global_visual


    def read(
        self,
        field_reader: MultiLayerFieldReader,
        prepared_field: Mapping[str, Any],
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Read all four actions through the one formal P3 field interface."""

        if not isinstance(field_reader, MultiLayerFieldReader):
            raise TypeError("field_reader must provide the formal P3 read protocol")
        batch_size: int | None = None
        field_tokens = prepared_field.get("field_tokens")
        if torch.is_tensor(field_tokens):
            batch_size = int(field_tokens.shape[0])
        elif torch.is_tensor(prepared_field.get("field_offset")):
            batch_size = int(prepared_field["field_offset"].shape[0])
        if batch_size is None or batch_size <= 0:
            raise ValueError("prepared P3 field must expose a positive batch dimension")

        queries = self.compositional_queries(batch_size)
        field_read = field_reader.read(prepared_field, queries, group_name="action_category")
        if not isinstance(field_read, Mapping):
            raise TypeError("formal P3 reader must return a mapping")
        readouts, layer_weights = self._validate_field_read(field_read, batch_size, queries.device)

        attention_delta, _ = self.action_self_attention(readouts, readouts, readouts, need_weights=False)
        action_visual_tokens = self.action_attention_norm(readouts + attention_delta)
        z_A_global_visual = self.project_global(action_visual_tokens)
        if not torch.isfinite(action_visual_tokens).all():
            raise FloatingPointError("P7 action foundation produced non-finite values")

        layer_entropy = -(layer_weights * layer_weights.clamp_min(1.0e-8).log()).sum(dim=-1)
        diagnostics = {
            "action_token_norm": torch.linalg.vector_norm(action_visual_tokens, dim=-1),
            "readout_norm": torch.linalg.vector_norm(readouts, dim=-1),
            "layer_weight_entropy": layer_entropy,
            "global_logit_rms": z_A_global_visual.square().mean(dim=-1).sqrt(),
        }
        return {
            "action_queries": queries,
            "readouts": readouts,
            "layer_weights": layer_weights,
            "action_visual_tokens": action_visual_tokens,
            "z_A_global_visual": z_A_global_visual,
            "diagnostics": diagnostics,
        }

    def forward(
        self,
        field_reader: MultiLayerFieldReader,
        prepared_field: Mapping[str, Any],
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        return self.read(field_reader, prepared_field)


__all__ = [
    "ACTION_NAMES",
    "FORMAL_ACTION_COUNT",
    "FORMAL_DIM",
    "RAELActionCategoryFoundation",
]

