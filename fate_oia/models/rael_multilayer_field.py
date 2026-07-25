"""Layer-aware shared visual reading for RAEL-OIA.

This module is deliberately downstream of the frozen DINO extractor.  It
keeps all four 45x80 patch fields separate, precomputes each layer's keys and
values once, and lets action, reason, and slot query groups reuse those
tensors.  It never averages four DINO layers and never concatenates 14,400
patches into one attention sequence.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RAELMultiLayerField(nn.Module):
    """Project four DINO fields and provide query-conditioned layer reading.

    ``precompute`` is intentionally the only place that invokes K/V
    projections.  ``read`` accepts either one query tensor or a mapping of
    named query groups, so consumers can share the same prepared field without
    re-projecting all visual tokens.
    """

    REQUIRED_LAYERS = 4
    FORMAL_DIM = 384
    FORMAL_GRID_HW = (45, 80)
    FORMAL_NUM_TOKENS = 3600

    def __init__(
        self,
        dim: int = FORMAL_DIM,
        num_layers: int = REQUIRED_LAYERS,
        formal_grid_hw: tuple[int, int] = FORMAL_GRID_HW,
        collapse_threshold: float = 0.99,
        collapse_patience: int = 3,
    ) -> None:
        super().__init__()
        if num_layers != self.REQUIRED_LAYERS:
            raise ValueError(f"RAEL requires exactly {self.REQUIRED_LAYERS} DINO layers")
        if dim <= 0:
            raise ValueError("dim must be positive")
        if collapse_threshold <= 0.0 or collapse_threshold >= 1.0:
            raise ValueError("collapse_threshold must lie in (0, 1)")
        if collapse_patience <= 0:
            raise ValueError("collapse_patience must be positive")

        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.formal_grid_hw = tuple(int(value) for value in formal_grid_hw)
        self.formal_num_tokens = self.formal_grid_hw[0] * self.formal_grid_hw[1]
        self.collapse_threshold = float(collapse_threshold)
        self.collapse_patience = int(collapse_patience)

        # W_l and DWConv_l are independent for every selected DINO layer.
        self.input_projections = nn.ModuleList(
            nn.Linear(self.dim, self.dim, bias=True) for _ in range(self.num_layers)
        )
        self.local_convs = nn.ModuleList(
            nn.Conv2d(
                self.dim,
                self.dim,
                kernel_size=3,
                padding=1,
                groups=self.dim,
                bias=True,
            )
            for _ in range(self.num_layers)
        )
        self.layer_norms = nn.ModuleList(nn.LayerNorm(self.dim) for _ in range(self.num_layers))
        self.local_gamma = nn.Parameter(torch.full((self.num_layers,), 0.02))
        self.layer_embeddings = nn.Parameter(torch.empty(self.num_layers, self.dim))
        self.positional_embedding = nn.Parameter(
            torch.empty(1, self.dim, self.formal_grid_hw[0], self.formal_grid_hw[1])
        )

        # These projections are executed exactly once by precompute().
        self.key_projections = nn.ModuleList(
            nn.Linear(self.dim, self.dim, bias=False) for _ in range(self.num_layers)
        )
        self.value_projections = nn.ModuleList(
            nn.Linear(self.dim, self.dim, bias=False) for _ in range(self.num_layers)
        )
        self.attention_query_projection = nn.Linear(self.dim, self.dim, bias=False)
        self.layer_query_projection = nn.Linear(self.dim, self.dim, bias=False)
        self.layer_global_projection = nn.Linear(self.dim, self.dim, bias=False)
        # The layer router is exactly w^T tanh(Wq q + Wg g_l): no scalar
        # intercept and no layer-specific additive bias are permitted.
        self.layer_score = nn.Linear(self.dim, 1, bias=False)

        # These diagnostics are intentionally checkpointed so a resumed run
        # cannot forget an already persistent single-layer collapse.
        self.register_buffer("_collapse_streak", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("_collapse_fail", torch.zeros((), dtype=torch.bool), persistent=True)
        self.register_buffer(
            "_last_dominant_layer",
            torch.full((), -1, dtype=torch.long),
            persistent=True,
        )
        # Batch tokens are process-local guards, not optimisation state.  They
        # prevent action/reason/slot readers from finalizing one batch twice.
        self._next_collapse_batch_token = 0
        self._finalized_collapse_tokens: set[int] = set()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in self.input_projections:
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        for conv in self.local_convs:
            nn.init.kaiming_uniform_(conv.weight, a=5**0.5)
            nn.init.zeros_(conv.bias)
        for norm in self.layer_norms:
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        for collection in (self.key_projections, self.value_projections):
            for projection in collection:
                nn.init.xavier_uniform_(projection.weight)
        for projection in (
            self.attention_query_projection,
            self.layer_query_projection,
            self.layer_global_projection,
        ):
            nn.init.xavier_uniform_(projection.weight)
        nn.init.xavier_uniform_(self.layer_score.weight)
        nn.init.normal_(self.layer_embeddings, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)

    @classmethod
    def formal_metadata(cls) -> dict[str, int | tuple[int, int]]:
        """Return the immutable full-resolution DINO contract for audits."""
        return {
            "num_layers": cls.REQUIRED_LAYERS,
            "dim": cls.FORMAL_DIM,
            "grid_hw": cls.FORMAL_GRID_HW,
            "num_tokens": cls.FORMAL_NUM_TOKENS,
        }

    def _validate_precompute_inputs(
        self,
        patch_tokens_by_layer: Tensor,
        cls_tokens_by_layer: Tensor,
        grid_hw: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        if patch_tokens_by_layer.ndim != 4:
            raise ValueError(
                "patch_tokens_by_layer must be [B,4,N,D], got "
                f"{tuple(patch_tokens_by_layer.shape)}"
            )
        batch, layers, tokens, dim = patch_tokens_by_layer.shape
        if layers != self.num_layers or dim != self.dim:
            raise ValueError(
                f"expected [B,{self.num_layers},N,{self.dim}], got "
                f"{tuple(patch_tokens_by_layer.shape)}"
            )
        if cls_tokens_by_layer.shape != (batch, self.num_layers, self.dim):
            raise ValueError(
                "cls_tokens_by_layer must be [B,4,D] matching patches, got "
                f"{tuple(cls_tokens_by_layer.shape)}"
            )
        if len(grid_hw) != 2 or min(grid_hw) <= 0:
            raise ValueError(f"grid_hw must contain two positive values, got {grid_hw}")
        height, width = (int(grid_hw[0]), int(grid_hw[1]))
        if height * width != tokens:
            raise ValueError(
                f"grid_hw={grid_hw} contains {height * width} positions, expected {tokens}"
            )
        return batch, layers, tokens, dim

    def _position_encoding(self, grid_hw: tuple[int, int], *, dtype: torch.dtype) -> Tensor:
        position = self.positional_embedding
        if tuple(grid_hw) != self.formal_grid_hw:
            position = F.interpolate(
                position,
                size=grid_hw,
                mode="bilinear",
                align_corners=False,
            )
        return position.to(dtype=dtype).permute(0, 2, 3, 1).reshape(1, -1, self.dim)

    def _parameter_dtype(self) -> torch.dtype:
        return next(self.input_projections[0].parameters()).dtype

    @staticmethod
    def _supports_internal_cuda_bfloat16(value: Tensor) -> bool:
        return (
            value.device.type == "cuda"
            and value.dtype == torch.bfloat16
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )

    def _prepare_compute_inputs(self, patch_tokens: Tensor, cls_tokens: Tensor) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """Normalize bare BF16 inputs without requiring an outer autocast context."""
        internal_autocast = self._supports_internal_cuda_bfloat16(patch_tokens)
        compute_dtype = patch_tokens.dtype if internal_autocast else self._parameter_dtype()
        working_patch = patch_tokens if patch_tokens.dtype == compute_dtype else patch_tokens.to(compute_dtype)
        working_cls = cls_tokens if cls_tokens.dtype == compute_dtype else cls_tokens.to(compute_dtype)
        return (
            working_patch,
            working_cls,
            {
                "input_dtype": str(patch_tokens.dtype),
                "compute_dtype": str(compute_dtype),
                "output_dtype": None,
                "internal_autocast": internal_autocast,
            },
        )

    @staticmethod
    def _autocast_context(enabled: bool):
        if enabled:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _new_collapse_batch_token(self) -> int:
        token = self._next_collapse_batch_token
        self._next_collapse_batch_token += 1
        return token

    def precompute(
        self,
        patch_tokens_by_layer: Tensor,
        cls_tokens_by_layer: Tensor,
        grid_hw: tuple[int, int] = FORMAL_GRID_HW,
    ) -> dict[str, Any]:
        """Construct per-layer fields and shared K/V exactly once.

        The loop holds only one layer's `[B,N,D]` local result at a time.  It
        deliberately never materializes a `[B,Q,L,N,D]` tensor.
        """
        batch, _, tokens, _ = self._validate_precompute_inputs(
            patch_tokens_by_layer,
            cls_tokens_by_layer,
            grid_hw,
        )
        working_patches, working_cls, dtype_semantics = self._prepare_compute_inputs(
            patch_tokens_by_layer,
            cls_tokens_by_layer,
        )
        position = self._position_encoding(grid_hw, dtype=working_patches.dtype)
        fields: list[Tensor] = []
        keys: list[Tensor] = []
        values: list[Tensor] = []
        global_tokens: list[Tensor] = []
        with self._autocast_context(bool(dtype_semantics["internal_autocast"])):
            for layer_index in range(self.num_layers):
                source = working_patches[:, layer_index]
                projected = self.input_projections[layer_index](source)
                local = self.local_convs[layer_index](
                    source.transpose(1, 2).reshape(batch, self.dim, grid_hw[0], grid_hw[1])
                )
                local = local.flatten(2).transpose(1, 2)
                field = self.layer_norms[layer_index](
                    projected
                    + self.local_gamma[layer_index] * local
                    + position
                    + self.layer_embeddings[layer_index].view(1, 1, self.dim)
                )
                fields.append(field)
                keys.append(self.key_projections[layer_index](field))
                values.append(self.value_projections[layer_index](field))
                # Avoid a layer mean: this is a spatial global pooling within
                # one layer, combined with that layer's DINO CLS token.
                global_tokens.append(field.sum(dim=1) / tokens + working_cls[:, layer_index])

        field_tokens = torch.stack(fields, dim=1)
        dtype_semantics["output_dtype"] = str(field_tokens.dtype)

        return {
            "field_tokens": field_tokens,
            "keys_by_layer": torch.stack(keys, dim=1),
            "values_by_layer": torch.stack(values, dim=1),
            "layer_global_tokens": torch.stack(global_tokens, dim=1),
            "grid_hw": tuple(int(value) for value in grid_hw),
            "collapse_batch_token": self._new_collapse_batch_token(),
            "dtype_semantics": dtype_semantics,
        }

    def _validate_prepared(self, prepared: Mapping[str, Any]) -> tuple[int, int, int]:
        required = ("field_tokens", "keys_by_layer", "values_by_layer", "layer_global_tokens")
        missing = [key for key in required if key not in prepared]
        if missing:
            raise KeyError(f"prepared RAEL field is missing {missing}")
        fields = prepared["field_tokens"]
        keys = prepared["keys_by_layer"]
        values = prepared["values_by_layer"]
        globals_ = prepared["layer_global_tokens"]
        if not all(torch.is_tensor(value) for value in (fields, keys, values, globals_)):
            raise TypeError("prepared visual field tensors must remain tensors")
        if fields.ndim != 4 or tuple(fields.shape[1:])[-1] != self.dim:
            raise ValueError("prepared field_tokens must be [B,4,N,D]")
        batch, layers, tokens, dim = fields.shape
        if layers != self.num_layers or dim != self.dim:
            raise ValueError("prepared field_tokens have the wrong layer or embedding dimensions")
        if keys.shape != fields.shape or values.shape != fields.shape:
            raise ValueError("prepared keys/values must exactly match field_tokens shape")
        if globals_.shape != (batch, self.num_layers, self.dim):
            raise ValueError("layer_global_tokens must be [B,4,D]")
        return batch, tokens, dim

    def _normalise_query_shape(self, queries: Tensor, batch: int) -> tuple[Tensor, tuple[int, ...]]:
        if queries.ndim < 2 or queries.shape[0] != batch or queries.shape[-1] != self.dim:
            raise ValueError(
                f"queries must be [B,...,{self.dim}], got {tuple(queries.shape)} for B={batch}"
            )
        original_query_shape = tuple(queries.shape[1:-1])
        if not original_query_shape:
            original_query_shape = (1,)
            queries = queries.unsqueeze(1)
        return queries.reshape(batch, -1, self.dim), original_query_shape

    def _read_collapse_diagnostic(self, weights: Tensor) -> dict[str, Tensor]:
        per_query_collapsed = weights.detach().amax(dim=-1) > self.collapse_threshold
        collapse_rate = per_query_collapsed.float().mean()
        group_layer_mean = weights.detach().mean(dim=(0, 1))
        dominant_weight, dominant_layer = group_layer_mean.max(dim=0)
        return {
            "layer_collapse_rate": collapse_rate,
            "batch_dominant_layer": dominant_layer.detach(),
            "batch_dominant_weight": dominant_weight.detach(),
            "batch_collapse_observed": (dominant_weight.detach() > self.collapse_threshold),
            "layer_collapse_streak": self._collapse_streak.detach().clone(),
            "layer_collapse_fail": self._collapse_fail.detach().clone(),
        }

    def _read_tensor(
        self,
        prepared: Mapping[str, Any],
        queries: Tensor,
        group_name: str | None,
    ) -> dict[str, Tensor | str | None]:
        batch, _, _ = self._validate_prepared(prepared)
        keys = prepared["keys_by_layer"]
        values = prepared["values_by_layer"]
        globals_ = prepared["layer_global_tokens"]
        dtype_semantics = prepared.get("dtype_semantics")
        if not isinstance(dtype_semantics, Mapping):
            raise TypeError("prepared field is missing dtype_semantics")
        internal_autocast = bool(dtype_semantics.get("internal_autocast", False))
        assert isinstance(keys, Tensor) and isinstance(values, Tensor) and isinstance(globals_, Tensor)
        if queries.device != keys.device:
            raise ValueError("queries and prepared field must be on the same device")
        working_queries = queries if internal_autocast else queries.to(dtype=keys.dtype)
        flat_queries, original_query_shape = self._normalise_query_shape(working_queries, batch)
        with self._autocast_context(internal_autocast):
            layer_query = self.layer_query_projection(flat_queries)
            layer_global = self.layer_global_projection(globals_)
            layer_scores = self.layer_score(
                torch.tanh(layer_query.unsqueeze(2) + layer_global.unsqueeze(1))
            ).squeeze(-1)
            layer_weights = torch.softmax(layer_scores, dim=-1)

            attention_query = self.attention_query_projection(flat_queries)
            reads: list[Tensor] = []
            for layer_index in range(self.num_layers):
                # Each layer is read independently; no [B,Q,L,N,D] allocation.
                scores = torch.einsum("bqd,bnd->bqn", attention_query, keys[:, layer_index])
                scores = scores * (self.dim**-0.5)
                attention = torch.softmax(scores, dim=-1)
                reads.append(torch.einsum("bqn,bnd->bqd", attention, values[:, layer_index]))
            layer_readouts = torch.stack(reads, dim=2)
            readout = (layer_weights.unsqueeze(-1) * layer_readouts).sum(dim=2)
        entropy = -(layer_weights * layer_weights.clamp_min(1e-8).log()).sum(dim=-1)
        collapse_diagnostic = self._read_collapse_diagnostic(layer_weights)

        restore_shape = (batch, *original_query_shape)
        restored_weights = layer_weights.reshape(*restore_shape, self.num_layers)
        # `[B,Q,D]` represents one named group, while `[B,G,Q,D]` retains one
        # diagnostic vector per group by averaging only its query axis.
        if len(original_query_shape) == 1:
            per_group_layer_weights = restored_weights.sum(dim=1) / restored_weights.shape[1]
        else:
            per_group_layer_weights = restored_weights.mean(dim=-2)
        return {
            "group_name": group_name,
            "readout": readout.reshape(*restore_shape, self.dim),
            "layer_readouts": layer_readouts.reshape(*restore_shape, self.num_layers, self.dim),
            "layer_weights": restored_weights,
            "per_group_layer_weights": per_group_layer_weights,
            "layer_entropy": entropy.reshape(*restore_shape),
            **collapse_diagnostic,
            "dtype_semantics": {
                "query_input_dtype": str(queries.dtype),
                "compute_dtype": str(flat_queries.dtype),
                "output_dtype": str(readout.dtype),
                "internal_autocast": internal_autocast,
            },
        }

    def finalize_batch_collapse(
        self,
        prepared: Mapping[str, Any],
        group_reads: Mapping[str, Any] | Mapping[str, Tensor | str | None],
    ) -> dict[str, Any]:
        """Aggregate one action/reason/slot batch and update once in training.

        ``read`` never changes collapse state.  Training callers invoke this
        method exactly once after all query groups have consumed ``prepared``.
        In eval mode it returns the same aggregate diagnostics without writing
        persistent buffers or consuming the batch token.
        """
        token = prepared.get("collapse_batch_token")
        if not isinstance(token, int):
            raise KeyError("prepared field is missing integer collapse_batch_token")
        if "layer_weights" in group_reads:
            results = [group_reads]
        else:
            results = list(group_reads.values())
        if not results:
            raise ValueError("at least one action, reason, or slot read is required")

        weight_sum: Tensor | None = None
        total_queries = 0
        collapsed_queries = 0
        for result in results:
            if not isinstance(result, Mapping) or not torch.is_tensor(result.get("layer_weights")):
                raise TypeError("each group read must contain layer_weights")
            weights = result["layer_weights"].detach()
            flat = weights.reshape(weights.shape[0], -1, self.num_layers)
            summary = flat.sum(dim=(0, 1))
            weight_sum = summary if weight_sum is None else weight_sum + summary
            total_queries += flat.shape[0] * flat.shape[1]
            collapsed_queries += int((flat.amax(dim=-1) > self.collapse_threshold).sum().item())
        assert weight_sum is not None and total_queries > 0
        aggregate_weights = weight_sum / total_queries
        dominant_weight, dominant_layer = aggregate_weights.max(dim=0)
        observed = bool(dominant_weight > self.collapse_threshold)
        collapse_rate = torch.tensor(
            collapsed_queries / total_queries,
            device=aggregate_weights.device,
            dtype=aggregate_weights.dtype,
        )

        updated = False
        if self.training:
            if token in self._finalized_collapse_tokens:
                raise RuntimeError(f"collapse batch token {token} was already finalized")
            self._finalized_collapse_tokens.add(token)
            with torch.no_grad():
                if observed:
                    if int(dominant_layer) == int(self._last_dominant_layer):
                        self._collapse_streak.add_(1)
                    else:
                        self._collapse_streak.fill_(1)
                    self._last_dominant_layer.copy_(dominant_layer)
                else:
                    self._collapse_streak.zero_()
                    self._last_dominant_layer.fill_(-1)
                self._collapse_fail.copy_(self._collapse_streak >= self.collapse_patience)
            updated = True

        return {
            "collapse_batch_token": token,
            "collapse_state_updated": updated,
            "layer_collapse_rate": collapse_rate,
            "batch_dominant_layer": dominant_layer.detach(),
            "batch_dominant_weight": dominant_weight.detach(),
            "batch_collapse_observed": observed,
            "layer_collapse_streak": self._collapse_streak.detach().clone(),
            "layer_collapse_fail": self._collapse_fail.detach().clone(),
        }

    def read(
        self,
        prepared: Mapping[str, Any],
        queries: Tensor | Mapping[str, Tensor],
        group_name: str | None = None,
    ) -> dict[str, Tensor | str | None] | dict[str, dict[str, Tensor | str | None]]:
        """Read one query tensor or named action/reason/slot query groups."""
        if isinstance(queries, Mapping):
            if group_name is not None:
                raise ValueError("group_name is only valid when reading one tensor")
            return {
                str(name): self._read_tensor(prepared, value, str(name))
                for name, value in queries.items()
            }
        if not torch.is_tensor(queries):
            raise TypeError("queries must be a Tensor or mapping of query-group tensors")
        return self._read_tensor(prepared, queries, group_name)

    def forward(
        self,
        patch_tokens_by_layer: Tensor,
        cls_tokens_by_layer: Tensor,
        queries: Tensor | Mapping[str, Tensor],
        grid_hw: tuple[int, int] = FORMAL_GRID_HW,
        group_name: str | None = None,
    ) -> tuple[
        dict[str, Tensor | tuple[int, int]],
        dict[str, Tensor | str | None] | dict[str, dict[str, Tensor | str | None]],
    ]:
        """Convenience API; callers with multiple groups should reuse precompute."""
        prepared = self.precompute(patch_tokens_by_layer, cls_tokens_by_layer, grid_hw)
        return prepared, self.read(prepared, queries, group_name=group_name)


__all__ = ["RAELMultiLayerField"]
