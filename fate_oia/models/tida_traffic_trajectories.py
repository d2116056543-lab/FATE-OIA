from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TIDATrafficTrajectoryBuilder(nn.Module):
    """Propagate final-frame action anchors backward with cycle-consistent DINO matching."""

    def __init__(
        self,
        temperature: float = 0.07,
        cycle_scale: float = 0.20,
        local_radius: int = 4,
    ) -> None:
        super().__init__()
        if temperature <= 0 or cycle_scale <= 0:
            raise ValueError("temperature and cycle_scale must be positive")
        self.temperature = float(temperature)
        self.cycle_scale = float(cycle_scale)
        self.local_radius = int(local_radius)
        if self.local_radius < 1:
            raise ValueError("local_radius must be positive")

    @staticmethod
    def _local_offsets(
        height: int, width: int, radius: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        dy, dx = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=device, dtype=dtype),
            torch.arange(-radius, radius + 1, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack(
            (2.0 * dx.flatten() / max(width - 1, 1), 2.0 * dy.flatten() / max(height - 1, 1)),
            dim=-1,
        )

    def _local_match(
        self,
        query: torch.Tensor,
        centers: torch.Tensor,
        field: torch.Tensor,
        grid_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Soft-match each query inside a bounded spatial window."""
        batch, actions, tracks, dim = query.shape
        height, width = grid_hw
        if field.shape != (batch, height * width, dim):
            raise ValueError("dense patch field does not agree with grid_hw")
        offsets = self._local_offsets(height, width, self.local_radius, field.device, field.dtype)
        sample_grid = centers[..., None, :] + offsets.view(1, 1, 1, -1, 2)
        candidate_valid = (sample_grid.abs() <= 1.0 + 1e-6).all(-1)
        feature_map = field.transpose(1, 2).reshape(batch, dim, height, width)
        sampled = F.grid_sample(
            feature_map,
            sample_grid.reshape(batch, actions * tracks, -1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.permute(0, 2, 3, 1).reshape(batch, actions, tracks, -1, dim)
        score = torch.einsum(
            "bakd,baknd->bakn", F.normalize(query, dim=-1), F.normalize(sampled, dim=-1)
        ) / self.temperature
        score = score.masked_fill(~candidate_valid, -1e4)
        probability = score.softmax(-1)
        matched_tokens = torch.einsum("bakn,baknd->bakd", probability, sampled)
        matched_xy = torch.einsum("bakn,baknc->bakc", probability, sample_grid)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(-1)
        valid_count = candidate_valid.sum(-1).clamp_min(1).to(entropy.dtype)
        confidence = 1.0 - entropy / valid_count.log().clamp_min(1.0)
        confidence = confidence.clamp(0.0, 1.0) * probability.max(-1).values.sqrt()
        coverage = candidate_valid.float().mean(-1)
        return matched_tokens, matched_xy, confidence, coverage

    def _dense_forward(
        self,
        patch_tokens: torch.Tensor,
        patch_xy: torch.Tensor,
        patch_weights: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        dense_patch_tokens: torch.Tensor,
        dense_grid_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        batch, frames, actions, tracks, _ = patch_tokens.shape
        if dense_patch_tokens.shape[:2] != (batch, frames):
            raise ValueError("dense_patch_tokens must share batch and frame dimensions")
        final_tokens = patch_tokens[:, -1]
        final_xy = patch_xy[:, -1]
        token_steps, xy_steps = [final_tokens], [final_xy]
        confidence_steps = [
            frame_valid_mask[:, -1, None, None].to(patch_weights.dtype).expand(-1, actions, tracks)
        ]
        coverage_steps = [torch.ones_like(confidence_steps[0])]
        next_tokens, next_xy = final_tokens, final_xy

        for frame in range(frames - 2, -1, -1):
            previous_field = dense_patch_tokens[:, frame]
            next_field = dense_patch_tokens[:, frame + 1]
            _, provisional_xy, provisional_confidence, _ = self._local_match(
                next_tokens, next_xy, previous_field, dense_grid_hw
            )
            provisional_displacement = next_xy - provisional_xy
            common = provisional_displacement.reshape(batch, actions * tracks, 2).median(1).values
            predicted_previous_xy = next_xy - common[:, None, None]
            matched_tokens, matched_xy, backward_confidence, coverage = self._local_match(
                next_tokens, predicted_previous_xy, previous_field, dense_grid_hw
            )
            _, cycle_xy, forward_confidence, _ = self._local_match(
                matched_tokens, matched_xy + common[:, None, None], next_field, dense_grid_hw
            )
            cycle_error = (cycle_xy - next_xy).square().sum(-1).sqrt()
            reciprocal_confidence = (
                provisional_confidence * backward_confidence * forward_confidence
            ).clamp_min(1e-12).pow(1.0 / 3.0)
            confidence = reciprocal_confidence * torch.exp(-cycle_error / self.cycle_scale)
            pair_valid = frame_valid_mask[:, frame] & frame_valid_mask[:, frame + 1]
            confidence = confidence * pair_valid[:, None, None].to(confidence.dtype)
            token_steps.append(matched_tokens)
            xy_steps.append(matched_xy)
            confidence_steps.append(confidence)
            coverage_steps.append(coverage)
            next_tokens, next_xy = matched_tokens, matched_xy

        return self._finalize(
            patch_weights,
            frame_valid_mask,
            torch.stack(token_steps[::-1], dim=3),
            torch.stack(xy_steps[::-1], dim=3),
            torch.stack(confidence_steps[::-1], dim=3),
            torch.stack(coverage_steps[::-1], dim=3),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_xy: torch.Tensor,
        patch_weights: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        *,
        dense_patch_tokens: torch.Tensor | None = None,
        dense_grid_hw: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        if patch_tokens.ndim != 5:
            raise ValueError("patch_tokens must be [B,T,A,K,D]")
        batch, frames, actions, tracks, dim = patch_tokens.shape
        if frames < 2:
            raise ValueError("at least two frames are required")
        if patch_xy.shape != (batch, frames, actions, tracks, 2):
            raise ValueError("patch_xy must be [B,T,A,K,2]")
        if patch_weights.shape != (batch, frames, actions, tracks):
            raise ValueError("patch_weights must be [B,T,A,K]")
        if frame_valid_mask.shape != (batch, frames):
            raise ValueError("frame_valid_mask must be [B,T]")
        if dense_patch_tokens is not None:
            if dense_grid_hw is None:
                raise ValueError("dense_grid_hw is required with dense_patch_tokens")
            return self._dense_forward(
                patch_tokens,
                patch_xy,
                patch_weights,
                frame_valid_mask,
                dense_patch_tokens,
                dense_grid_hw,
            )

        final_tokens = patch_tokens[:, -1]
        final_xy = patch_xy[:, -1]
        token_steps = [final_tokens]
        xy_steps = [final_xy]
        confidence_steps = [
            frame_valid_mask[:, -1, None, None].to(patch_weights.dtype).expand(-1, actions, tracks)
        ]

        next_tokens = final_tokens
        next_xy = final_xy
        for frame in range(frames - 2, -1, -1):
            candidates = patch_tokens[:, frame]
            candidates_xy = patch_xy[:, frame]
            next_norm = F.normalize(next_tokens, dim=-1)
            candidate_norm = F.normalize(candidates, dim=-1)
            similarity = torch.einsum("baqd,bakd->baqk", next_norm, candidate_norm)
            backward = torch.softmax(similarity / self.temperature, dim=-1)
            matched_tokens = torch.einsum("baqk,bakd->baqd", backward, candidates)
            matched_xy = torch.einsum("baqk,bakc->baqc", backward, candidates_xy)

            # A reverse lookup must return to the next-frame anchor, not merely find a similar patch.
            matched_norm = F.normalize(matched_tokens, dim=-1)
            next_candidates = F.normalize(patch_tokens[:, frame + 1], dim=-1)
            forward_similarity = torch.einsum("baqd,bakd->baqk", matched_norm, next_candidates)
            forward = torch.softmax(forward_similarity / self.temperature, dim=-1)
            cycle_xy = torch.einsum("baqk,bakc->baqc", forward, patch_xy[:, frame + 1])
            cycle_error = (cycle_xy - next_xy).square().sum(-1).sqrt()
            confidence = backward.max(-1).values * torch.exp(-cycle_error / self.cycle_scale)
            pair_valid = frame_valid_mask[:, frame] & frame_valid_mask[:, frame + 1]
            confidence = confidence * pair_valid[:, None, None].to(confidence.dtype)

            token_steps.append(matched_tokens)
            xy_steps.append(matched_xy)
            confidence_steps.append(confidence)
            next_tokens, next_xy = matched_tokens, matched_xy

        trajectory_appearance = torch.stack(token_steps[::-1], dim=3)
        trajectory_xy = torch.stack(xy_steps[::-1], dim=3)
        trajectory_confidence = torch.stack(confidence_steps[::-1], dim=3)
        return self._finalize(
            patch_weights,
            frame_valid_mask,
            trajectory_appearance,
            trajectory_xy,
            trajectory_confidence,
            torch.ones_like(trajectory_confidence),
        )

    @staticmethod
    def _finalize(
        patch_weights: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        trajectory_appearance: torch.Tensor,
        trajectory_xy: torch.Tensor,
        trajectory_confidence: torch.Tensor,
        local_candidate_coverage: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, actions, tracks, frames, _ = trajectory_appearance.shape
        displacement = trajectory_xy[..., 1:, :] - trajectory_xy[..., :-1, :]
        pair_valid = (frame_valid_mask[:, 1:] & frame_valid_mask[:, :-1])[:, None, None]
        pair_valid = pair_valid.expand(batch, actions, tracks, frames - 1)
        displacement = displacement * pair_valid[..., None].to(displacement.dtype)

        # A median-centered clipped mean is robust to independently moving agents.
        by_interval = displacement.permute(0, 3, 1, 2, 4).reshape(batch, frames - 1, actions * tracks, 2)
        valid_weight = (
            trajectory_confidence[..., 1:].permute(0, 3, 1, 2).reshape(batch, frames - 1, actions * tracks)
        )
        median = by_interval.median(dim=2).values
        residual = (by_interval - median[:, :, None]).clamp(-0.25, 0.25)
        weight = valid_weight / valid_weight.sum(-1, keepdim=True).clamp_min(1e-8)
        common = median + torch.einsum("bfn,bfnc->bfc", weight, residual)
        interval_available = pair_valid.any(dim=(1, 2))
        common = common * interval_available[..., None].to(common.dtype)
        exclusive = displacement - common[:, None, None]

        return {
            "trajectory_appearance": trajectory_appearance,
            "trajectory_xy": trajectory_xy,
            "trajectory_visibility": trajectory_confidence,
            "trajectory_anchor_weight": patch_weights[:, -1],
            "trajectory_cycle_confidence": trajectory_confidence,
            "trajectory_pair_valid": pair_valid,
            "trajectory_displacement": displacement,
            "trajectory_common_displacement": common,
            "trajectory_exclusive_displacement": exclusive,
            "trajectory_local_candidate_coverage": local_candidate_coverage,
        }
