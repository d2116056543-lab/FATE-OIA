from __future__ import annotations

import weakref
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .tida_relational_traffic_flow import select_semantic_traffic_seeds


class TIDAContextEncoder(nn.Module):
    """Chunked history encoder that does not register a second DINO owner."""

    def __init__(
        self,
        dino_extractor: nn.Module,
        query_reader: nn.Module | None,
        context_chunk_size: int = 2,
        motion_topk: int = 12,
        action_patch_selection: str = "topk",
        action_patch_nms_radius: int = 0,
        action_patch_specificity_power: float = 1.0,
    ) -> None:
        super().__init__()
        if context_chunk_size < 1:
            raise ValueError("context_chunk_size must be positive")
        object.__setattr__(self, "_dino_reference", weakref.ref(dino_extractor))
        object.__setattr__(self, "_query_reader_reference", None if query_reader is None else weakref.ref(query_reader))
        self.context_chunk_size = int(context_chunk_size)
        self.motion_topk = int(motion_topk)
        if self.motion_topk < 1:
            raise ValueError("motion_topk must be positive")
        if action_patch_selection not in {"topk", "contrastive_diverse"}:
            raise ValueError("action_patch_selection must be topk or contrastive_diverse")
        if action_patch_nms_radius < 0 or action_patch_specificity_power < 0:
            raise ValueError("action patch diversity parameters must be non-negative")
        self.action_patch_selection = str(action_patch_selection)
        self.action_patch_nms_radius = int(action_patch_nms_radius)
        self.action_patch_specificity_power = float(action_patch_specificity_power)

    @property
    def dino_extractor(self) -> nn.Module:
        extractor = self._dino_reference()
        if extractor is None:
            raise RuntimeError("shared DINO owner has been released")
        return extractor

    @property
    def query_reader(self) -> nn.Module | None:
        reference = self._query_reader_reference
        return None if reference is None else reference()

    @staticmethod
    def _resample_field_for_decoder(
        field: dict[str, Any], target_grid_hw: tuple[int, int] = (45, 80)
    ) -> dict[str, Any]:
        """Map a low-resolution history field onto the frozen image head grid."""
        source_height, source_width = field["grid_hw"]
        target_height, target_width = target_grid_hw
        if (source_height, source_width) == target_grid_hw:
            return field
        patches = field["patch_tokens_by_layer"]
        batch, layers, patch_count, dim = patches.shape
        if patch_count != source_height * source_width:
            raise ValueError("DINO field tokens do not match source grid")
        spatial = patches.reshape(
            batch * layers, source_height, source_width, dim
        ).permute(0, 3, 1, 2)
        resized = F.interpolate(
            spatial.float(), size=target_grid_hw, mode="bilinear", align_corners=False
        ).to(patches.dtype)
        resized = resized.permute(0, 2, 3, 1).reshape(
            batch, layers, target_height * target_width, dim
        )
        return {
            **field,
            "patch_tokens_by_layer": resized,
            "patch_tokens_last": resized[:, -1],
            "grid_hw": target_grid_hw,
            "original_tokens": target_height * target_width + 1,
        }

    def forward(
        self,
        context_images: torch.Tensor,
        action_nodes: torch.Tensor,
        predicate_tokens: torch.Tensor,
        predicate_identities: torch.Tensor,
        *,
        predicate_reliability: torch.Tensor | None = None,
        canonicalize_horizontal_flip: bool = False,
        reason_nodes: torch.Tensor | None = None,
        frozen_frame_decoder: Any | None = None,
    ) -> dict[str, Any]:
        if self.query_reader is None:
            raise RuntimeError("query_reader is required for context encoding")
        if context_images.ndim != 5:
            raise ValueError("context_images must be [B,T,3,H,W]")
        batch, frames, channels, height, width = context_images.shape
        tokens, attentions, region_masses, dense_patch_fields = [], [], [], []
        action_patch_tokens, action_patch_xy, action_patch_weights = [], [], []
        action_patch_indices, action_patch_specificity = [], []
        semantic_patch_tokens, semantic_patch_xy, semantic_patch_weights = [], [], []
        semantic_predicate_ids = []
        reason_tokens, reason_attentions, reason_region_masses = [], [], []
        reason_patch_tokens, reason_patch_xy, reason_patch_weights = [], [], []
        image_action_logits, image_reason_logits = [], []
        if predicate_reliability is None:
            predicate_reliability = predicate_tokens.new_ones(
                batch, predicate_tokens.shape[1]
            )
        if predicate_reliability.shape != predicate_tokens.shape[:2]:
            raise ValueError("predicate_reliability must be [B,P]")
        for start in range(0, frames, self.context_chunk_size):
            stop = min(start + self.context_chunk_size, frames)
            images = context_images[:, start:stop].reshape(-1, channels, height, width)
            field = self.dino_extractor.forward_at_resolution(images, expected_hw=(height, width))
            if canonicalize_horizontal_flip:
                grid_height, grid_width = field["grid_hw"]
                patches = field["patch_tokens_by_layer"]
                field["patch_tokens_by_layer"] = patches.view(
                    patches.shape[0], patches.shape[1], grid_height, grid_width, patches.shape[-1]
                ).flip(3).flatten(2, 3)
            if frozen_frame_decoder is not None:
                with torch.no_grad():
                    decoded = frozen_frame_decoder(
                        self._resample_field_for_decoder(field)
                    )
                image_action_logits.append(
                    decoded["action_logits_final"].reshape(batch, stop - start, -1).detach()
                )
                image_reason_logits.append(
                    decoded["reason_logits_final"].reshape(batch, stop - start, -1).detach()
                )
            repeats = stop - start
            read = self.query_reader(
                field["patch_tokens_by_layer"],
                action_nodes[:, None].expand(-1, repeats, -1, -1).reshape(-1, action_nodes.shape[1], action_nodes.shape[2]),
                predicate_tokens[:, None].expand(-1, repeats, -1, -1).reshape(-1, predicate_tokens.shape[1], predicate_tokens.shape[2]),
                predicate_identities,
                grid_hw=field["grid_hw"],
                reason_nodes=(
                    None
                    if reason_nodes is None
                    else reason_nodes[:, None]
                    .expand(-1, repeats, -1, -1)
                    .reshape(-1, reason_nodes.shape[1], reason_nodes.shape[2])
                ),
            )
            tokens.append(read["query_tokens"].reshape(batch, repeats, -1, action_nodes.shape[-1]))
            attentions.append(read["query_attention"].reshape(batch, repeats, -1, field["patch_tokens_by_layer"].shape[2]))
            region_masses.append(read["query_region_mass"].reshape(batch, repeats, -1, 5))
            if reason_nodes is not None:
                reason_tokens.append(
                    read["reason_query_tokens"].reshape(
                        batch, repeats, reason_nodes.shape[1], reason_nodes.shape[2]
                    )
                )
                reason_attentions.append(
                    read["reason_query_attention"].reshape(
                        batch, repeats, reason_nodes.shape[1], -1
                    )
                )
                reason_region_masses.append(
                    read["reason_query_region_mass"].reshape(
                        batch, repeats, reason_nodes.shape[1], 5
                    )
                )
                reason_selected = self.select_action_patches(
                    field, read["reason_query_attention"]
                )
                reason_topk = reason_selected["tokens"].shape[2]
                reason_patch_tokens.append(
                    reason_selected["tokens"].reshape(
                        batch, repeats, reason_nodes.shape[1], reason_topk, -1
                    )
                )
                reason_patch_xy.append(
                    reason_selected["xy"].reshape(
                        batch, repeats, reason_nodes.shape[1], reason_topk, 2
                    )
                )
                reason_patch_weights.append(
                    reason_selected["weights"].reshape(
                        batch, repeats, reason_nodes.shape[1], reason_topk
                    )
                )
            dense_patch_fields.append(
                field["patch_tokens_last"].reshape(batch, repeats, -1, action_nodes.shape[-1])
            )
            selected = self.select_action_patches(
                field, read["query_attention"][:, : action_nodes.shape[1]]
            )
            topk = selected["tokens"].shape[2]
            action_patch_tokens.append(selected["tokens"].reshape(batch, repeats, action_nodes.shape[1], topk, -1))
            action_patch_xy.append(selected["xy"].reshape(batch, repeats, action_nodes.shape[1], topk, 2))
            action_patch_weights.append(selected["weights"].reshape(batch, repeats, action_nodes.shape[1], topk))
            action_patch_indices.append(
                selected["indices"].reshape(batch, repeats, action_nodes.shape[1], topk)
            )
            action_patch_specificity.append(
                selected["specificity"].reshape(batch, repeats, action_nodes.shape[1], topk)
            )
            semantic = select_semantic_traffic_seeds(
                field["patch_tokens_last"],
                read["query_attention"][:, action_nodes.shape[1] :],
                predicate_reliability[:, None].expand(-1, repeats, -1).reshape(
                    -1, predicate_tokens.shape[1]
                ),
                grid_hw=field["grid_hw"],
                topk=self.motion_topk,
            )
            semantic_topk = semantic["tokens"].shape[2]
            semantic_patch_tokens.append(
                semantic["tokens"].reshape(batch, repeats, 1, semantic_topk, -1)
            )
            semantic_patch_xy.append(
                semantic["xy"].reshape(batch, repeats, 1, semantic_topk, 2)
            )
            semantic_patch_weights.append(
                semantic["weights"].reshape(batch, repeats, 1, semantic_topk)
            )
            semantic_predicate_ids.append(
                semantic["predicate_ids"].reshape(batch, repeats, semantic_topk)
            )
        result = {
            "history_query_tokens": torch.cat(tokens, dim=1),
            "history_query_attention": torch.cat(attentions, dim=1),
            "history_query_region_mass": torch.cat(region_masses, dim=1),
            "history_action_patch_tokens": torch.cat(action_patch_tokens, dim=1),
            "history_action_patch_xy": torch.cat(action_patch_xy, dim=1),
            "history_action_patch_weight": torch.cat(action_patch_weights, dim=1),
            "history_action_patch_indices": torch.cat(action_patch_indices, dim=1),
            "history_action_patch_specificity": torch.cat(action_patch_specificity, dim=1),
            "history_semantic_patch_tokens": torch.cat(semantic_patch_tokens, dim=1),
            "history_semantic_patch_xy": torch.cat(semantic_patch_xy, dim=1),
            "history_semantic_patch_weight": torch.cat(semantic_patch_weights, dim=1),
            "history_semantic_predicate_ids": torch.cat(semantic_predicate_ids, dim=1),
            "history_patch_tokens_last": torch.cat(dense_patch_fields, dim=1),
            "history_grid_hw": (height // self.dino_extractor.patch_size, width // self.dino_extractor.patch_size),
        }
        if reason_nodes is not None:
            result.update(
                {
                    "history_reason_query_tokens": torch.cat(reason_tokens, dim=1),
                    "history_reason_query_attention": torch.cat(reason_attentions, dim=1),
                    "history_reason_query_region_mass": torch.cat(reason_region_masses, dim=1),
                    "history_reason_patch_tokens": torch.cat(reason_patch_tokens, dim=1),
                    "history_reason_patch_xy": torch.cat(reason_patch_xy, dim=1),
                    "history_reason_patch_weight": torch.cat(reason_patch_weights, dim=1),
                }
            )
        if frozen_frame_decoder is not None:
            result.update(
                {
                    "history_image_action_logits": torch.cat(image_action_logits, dim=1),
                    "history_image_reason_logits": torch.cat(image_reason_logits, dim=1),
                }
            )
        return result

    def select_action_patches(
        self,
        field: dict[str, Any],
        action_attention: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Select action-attentive patches without re-running the shared DINO field."""
        topk = min(self.motion_topk, action_attention.shape[-1])
        grid_height, grid_width = field["grid_hw"]
        if grid_height * grid_width != action_attention.shape[-1]:
            raise ValueError("action attention does not agree with the DINO patch grid")

        score = action_attention.clamp_min(0.0)
        specificity = torch.ones_like(score)
        if self.action_patch_selection == "contrastive_diverse":
            action_count = action_attention.shape[1]
            if action_count > 1:
                other_mean = (
                    score.sum(1, keepdim=True) - score
                ) / float(action_count - 1)
                specificity = score / (score + other_mean + 1e-8)
            score = score * specificity.pow(self.action_patch_specificity_power)
            top_index = self._spatially_diverse_topk(
                score, topk, grid_width, self.action_patch_nms_radius
            )
            top_weight = torch.gather(score, -1, top_index)
        else:
            top_weight, top_index = score.topk(topk, dim=-1)
        top_weight = top_weight / top_weight.sum(-1, keepdim=True).clamp_min(1e-8)
        zero_weight = top_weight.sum(-1, keepdim=True) <= 0
        if zero_weight.any():
            top_weight = torch.where(
                zero_weight,
                torch.full_like(top_weight, 1.0 / float(topk)),
                top_weight,
            )
        patch_field = field["patch_tokens_last"]
        gathered = torch.gather(
            patch_field[:, None].expand(-1, action_attention.shape[1], -1, -1),
            2,
            top_index[..., None].expand(-1, -1, -1, patch_field.shape[-1]),
        )
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, grid_height, device=patch_field.device, dtype=patch_field.dtype),
            torch.linspace(-1.0, 1.0, grid_width, device=patch_field.device, dtype=patch_field.dtype),
            indexing="ij",
        )
        coordinates = torch.stack((xx, yy), dim=-1).flatten(0, 1)
        return {
            "tokens": gathered,
            "xy": coordinates[top_index],
            "weights": top_weight,
            "indices": top_index,
            "specificity": torch.gather(specificity, -1, top_index),
        }

    @staticmethod
    def _spatially_diverse_topk(
        score: torch.Tensor,
        topk: int,
        grid_width: int,
        radius: int,
    ) -> torch.Tensor:
        """Greedy patch-grid NMS over a very small action-specific candidate set."""
        if radius <= 0:
            return score.topk(topk, dim=-1).indices
        flat_score = score.reshape(-1, score.shape[-1])
        candidate_count = min(flat_score.shape[-1], max(topk * 8, topk))
        # Transfer one compact index table instead of synchronizing the GPU once
        # per greedy comparison.
        ordered_rows = flat_score.topk(candidate_count, dim=-1).indices.detach().cpu().tolist()
        selected_rows: list[list[int]] = []
        for row in ordered_rows:
            chosen: list[int] = []
            for index in row:
                y, x = divmod(index, grid_width)
                separated = True
                for existing in chosen:
                    ey, ex = divmod(existing, grid_width)
                    if max(abs(y - ey), abs(x - ex)) <= radius:
                        separated = False
                        break
                if separated:
                    chosen.append(index)
                    if len(chosen) == topk:
                        break
            if len(chosen) < topk:
                chosen_set = set(chosen)
                for index in row:
                    if index not in chosen_set:
                        chosen.append(index)
                        if len(chosen) == topk:
                            break
            selected_rows.append(chosen)
        return torch.tensor(
            selected_rows, device=score.device, dtype=torch.long
        ).reshape(*score.shape[:-1], topk)
