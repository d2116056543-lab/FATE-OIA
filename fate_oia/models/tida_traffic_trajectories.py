from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TIDATrafficTrajectoryBuilder(nn.Module):
    """Propagate final-frame action anchors backward with cycle-consistent DINO matching."""

    def __init__(self, temperature: float = 0.07, cycle_scale: float = 0.20) -> None:
        super().__init__()
        if temperature <= 0 or cycle_scale <= 0:
            raise ValueError("temperature and cycle_scale must be positive")
        self.temperature = float(temperature)
        self.cycle_scale = float(cycle_scale)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_xy: torch.Tensor,
        patch_weights: torch.Tensor,
        frame_valid_mask: torch.Tensor,
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
        }
