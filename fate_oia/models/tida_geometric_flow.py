from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _region_masks(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)[:, None]
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)[None, :]
    masks = torch.stack(
        (
            torch.exp(-((x - 0.5).square() / 0.08 + (y - 0.72).square() / 0.20)),
            torch.sigmoid((0.46 - x) * 12.0) * torch.sigmoid((y - 0.30) * 10.0),
            torch.sigmoid((x - 0.54) * 12.0) * torch.sigmoid((y - 0.30) * 10.0),
            torch.sigmoid((0.46 - y) * 10.0).expand(height, width),
            torch.sigmoid((y - 0.58) * 10.0).expand(height, width),
        )
    )
    return masks / masks.sum((-2, -1), keepdim=True).clamp_min(1e-8)


class TIDAGeometricFlowEncoder(nn.Module):
    """Fixed low-resolution motion measurement with no external flow network."""

    descriptor_dim = 20
    # Three temporal summaries of raw and ego-compensated (u, v, energy,
    # confidence), plus the local token coordinate.
    motion_token_dim = 26

    def __init__(
        self,
        hidden_dim: int = 64,
        flow_hw: tuple[int, int] = (45, 80),
        motion_token_hw: tuple[int, int] = (4, 8),
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.flow_hw = tuple(int(value) for value in flow_hw)
        self.motion_token_hw = tuple(int(value) for value in motion_token_hw)
        if min(self.motion_token_hw) < 1:
            raise ValueError("motion_token_hw must be positive")
        if self.hidden_dim < 3 * self.descriptor_dim:
            raise ValueError(f"hidden_dim must be at least {3 * self.descriptor_dim}")
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_x.transpose(-1, -2).contiguous(), persistent=False)
        self.register_buffer("channel_scale", torch.tensor([0.229, 0.224, 0.225]), persistent=False)
        self.register_buffer("luma", torch.tensor([0.2989, 0.5870, 0.1140]), persistent=False)

    def _build_motion_tokens(
        self,
        horizontal: torch.Tensor,
        vertical: torch.Tensor,
        energy: torch.Tensor,
        confidence: torch.Tensor,
        residual_horizontal: torch.Tensor,
        residual_vertical: torch.Tensor,
        residual_energy: torch.Tensor,
        pair_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep local motion evidence until each action/reason chooses what to read."""
        batch, intervals = horizontal.shape[:2]
        channels = torch.cat(
            (
                horizontal,
                vertical,
                energy,
                confidence,
                residual_horizontal,
                residual_vertical,
                residual_energy,
                confidence,
            ),
            dim=2,
        )
        pooled = F.adaptive_avg_pool2d(
            channels.flatten(0, 1), self.motion_token_hw
        ).view(batch, intervals, 8, *self.motion_token_hw)
        valid = pair_valid[:, :, None, None, None].to(pooled.dtype)
        valid_count = valid.sum(1).clamp_min(1.0)
        summary = (pooled * valid).sum(1) / valid_count

        # Official clips are dense, but selecting the last valid interval also
        # keeps padded or partially decoded clips semantically correct.
        last_index = (
            pair_valid.long()
            * torch.arange(1, intervals + 1, device=pair_valid.device)[None]
        ).argmax(1)
        recent = pooled[torch.arange(batch, device=pooled.device), last_index]
        trend = recent - summary

        token_features = torch.cat((summary, recent, trend), dim=1)
        token_features = token_features.flatten(2).transpose(1, 2)
        y = torch.linspace(-1.0, 1.0, self.motion_token_hw[0], device=pooled.device, dtype=pooled.dtype)
        x = torch.linspace(-1.0, 1.0, self.motion_token_hw[1], device=pooled.device, dtype=pooled.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        token_xy = torch.stack((xx, yy), dim=-1).flatten(0, 1)
        token_xy = token_xy[None].expand(batch, -1, -1)
        token_features = torch.cat((token_features, token_xy), dim=-1)
        available = pair_valid.any(1)
        token_mask = available[:, None].expand(-1, token_features.shape[1])
        token_features = token_features * token_mask[..., None].to(token_features.dtype)
        return token_features, token_xy, token_mask

    @staticmethod
    def _affine_residual_flow(
        horizontal: torch.Tensor,
        vertical: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remove camera-induced affine flow while retaining local motion."""
        if not (horizontal.shape == vertical.shape == confidence.shape):
            raise ValueError("flow and confidence tensors must share [B,T,1,H,W]")
        batch, intervals, channels, height, width = horizontal.shape
        if channels != 1:
            raise ValueError("affine flow compensation expects one flow channel")
        output_dtype = horizontal.dtype
        # CUDA/CPU LU kernels do not support bf16. The 3x3 solve is tiny, and
        # fp32 also prevents an unstable camera model from contaminating flow.
        with torch.autocast(device_type=horizontal.device.type, enabled=False):
            horizontal_fp32 = horizontal.float()
            vertical_fp32 = vertical.float()
            confidence_fp32 = confidence.float()
            y = torch.linspace(-1.0, 1.0, height, device=horizontal.device)
            x = torch.linspace(-1.0, 1.0, width, device=horizontal.device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            design = torch.stack((torch.ones_like(xx), xx, yy), dim=-1).reshape(-1, 3)
            design = design[None].expand(batch * intervals, -1, -1)
            weight = confidence_fp32.reshape(batch * intervals, -1).clamp_min(0.0)
            target = torch.stack(
                (
                    horizontal_fp32.reshape(batch * intervals, -1),
                    vertical_fp32.reshape(batch * intervals, -1),
                ),
                dim=-1,
            )
            weighted_design = design * weight[..., None]
            normal = design.transpose(1, 2) @ weighted_design
            ridge = torch.eye(3, device=normal.device, dtype=normal.dtype)[None] * 1e-4
            rhs = design.transpose(1, 2) @ (target * weight[..., None])
            coefficients = torch.linalg.solve(normal + ridge, rhs)
            fitted = design @ coefficients
            residual = target - fitted
            residual = residual.transpose(1, 2).reshape(batch, intervals, 2, height, width)
        return residual[:, :, :1].to(output_dtype), residual[:, :, 1:].to(output_dtype)

    def _gray(self, frames: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = frames.shape
        if channels != 3:
            raise ValueError("geometric flow expects RGB input")
        # ImageNet means cancel in temporal/spatial differences; restoring the
        # channel scales is sufficient and also accepts unnormalized test input.
        rgb = frames.float() * self.channel_scale.view(1, 1, 3, 1, 1)
        gray = (rgb * self.luma.view(1, 1, 3, 1, 1)).sum(2, keepdim=True)
        return F.interpolate(
            gray.flatten(0, 1), self.flow_hw, mode="bilinear", align_corners=False
        ).view(batch, steps, 1, *self.flow_hw)

    def _correlation_descriptor(self, gray: torch.Tensor) -> torch.Tensor:
        padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(padded, self.sobel_x)
        grad_y = F.conv2d(padded, self.sobel_y)
        local_mean = F.avg_pool2d(gray, 3, stride=1, padding=1)
        centered = gray - local_mean
        local_scale = F.avg_pool2d(centered.square(), 3, stride=1, padding=1).clamp_min(1e-8).sqrt()
        return F.normalize(torch.cat((centered, grad_x, grad_y, local_scale), dim=1), dim=1, eps=1e-6)

    def _local_correlation_match(
        self, previous: torch.Tensor, current: torch.Tensor, radius: int = 4
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match local descriptors without materializing a dense token feature tensor."""
        previous_descriptor = self._correlation_descriptor(previous)
        current_descriptor = self._correlation_descriptor(current)
        batch, channels, height, width = previous_descriptor.shape
        kernel = 2 * int(radius) + 1
        candidates = F.unfold(current_descriptor, kernel_size=kernel, padding=radius)
        candidates = candidates.view(batch, channels, kernel * kernel, height, width)
        score = (previous_descriptor[:, :, None] * candidates).sum(1) / 0.12
        probability = score.softmax(1)
        offset_y, offset_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=previous.device, dtype=previous.dtype),
            torch.arange(-radius, radius + 1, device=previous.device, dtype=previous.dtype),
            indexing="ij",
        )
        offset_x = offset_x.flatten().view(1, -1, 1, 1)
        offset_y = offset_y.flatten().view(1, -1, 1, 1)
        horizontal = (probability * offset_x).sum(1, keepdim=True)
        vertical = (probability * offset_y).sum(1, keepdim=True)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(1, keepdim=True)
        confidence = (1.0 - entropy / torch.log(previous.new_tensor(float(kernel * kernel)))).clamp(0.0, 1.0)
        temporal_strength = F.avg_pool2d((current - previous).abs(), 3, stride=1, padding=1)
        confidence = confidence * temporal_strength / (temporal_strength + 0.01)
        return horizontal, vertical, confidence

    def _multiscale_correlation_flow(
        self, previous: torch.Tensor, current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_x, full_y, full_confidence = self._local_correlation_match(previous, current)
        coarse_hw = (max(8, previous.shape[-2] // 2), max(8, previous.shape[-1] // 2))
        coarse_previous = F.interpolate(previous, coarse_hw, mode="bilinear", align_corners=False)
        coarse_current = F.interpolate(current, coarse_hw, mode="bilinear", align_corners=False)
        coarse_x, coarse_y, coarse_confidence = self._local_correlation_match(coarse_previous, coarse_current)
        coarse_x = 2.0 * F.interpolate(coarse_x, previous.shape[-2:], mode="bilinear", align_corners=False)
        coarse_y = 2.0 * F.interpolate(coarse_y, previous.shape[-2:], mode="bilinear", align_corners=False)
        coarse_confidence = F.interpolate(
            coarse_confidence, previous.shape[-2:], mode="bilinear", align_corners=False
        )
        weight = (full_confidence + coarse_confidence).clamp_min(1e-6)
        horizontal = (full_confidence * full_x + coarse_confidence * coarse_x) / weight
        vertical = (full_confidence * full_y + coarse_confidence * coarse_y) / weight
        confidence = 1.0 - (1.0 - full_confidence) * (1.0 - coarse_confidence)
        correlation_x = torch.tanh(horizontal / 4.0) * confidence
        correlation_y = torch.tanh(vertical / 4.0) * confidence

        # Correlation handles sparse large displacement, while the differential
        # branch preserves local radial expansion/contraction at object edges.
        midpoint = 0.5 * (previous + current)
        padded = F.pad(midpoint, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(padded, self.sobel_x)
        grad_y = F.conv2d(padded, self.sobel_y)
        grad_t = current - previous
        denominator = grad_x.square() + grad_y.square() + 2e-3
        differential_confidence = (denominator - 2e-3).clamp_min(0.0)
        differential_confidence = differential_confidence / (differential_confidence + 0.02)
        differential_x = torch.tanh(-grad_t * grad_x / denominator) * differential_confidence
        differential_y = torch.tanh(-grad_t * grad_y / denominator) * differential_confidence
        horizontal = 0.75 * correlation_x + 0.25 * differential_x
        vertical = 0.75 * correlation_y + 0.25 * differential_y
        combined_confidence = 1.0 - (1.0 - confidence) * (1.0 - differential_confidence)
        return horizontal, vertical, combined_confidence

    @staticmethod
    def _flow_grid_tracks(
        flow: torch.Tensor,
        confidence: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        grid_size: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate fixed normalized seeds through measured flow fields."""
        batch, intervals, _, height, width = flow.shape
        axis = torch.linspace(-0.75, 0.75, grid_size, device=flow.device, dtype=flow.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        position = torch.stack((xx, yy), dim=-1).flatten(0, 1)
        position = position[None].expand(batch, -1, -1).clone()
        tracks = [position]
        visible = [frame_valid_mask[:, 0, None].expand(-1, position.shape[1])]
        for step in range(intervals):
            sample_grid = position[:, :, None]
            displacement = F.grid_sample(
                flow[:, step], sample_grid, mode="bilinear",
                padding_mode="zeros", align_corners=True,
            ).squeeze(-1).transpose(1, 2)
            match_confidence = F.grid_sample(
                confidence[:, step], sample_grid, mode="bilinear",
                padding_mode="zeros", align_corners=True,
            ).squeeze(1).squeeze(-1)
            normalized = torch.stack(
                (
                    2.0 * displacement[..., 0] / max(width - 1, 1),
                    2.0 * displacement[..., 1] / max(height - 1, 1),
                ),
                dim=-1,
            )
            next_position = position + normalized
            in_bounds = next_position.abs().amax(-1) <= 1.0
            step_visible = (
                visible[-1]
                & frame_valid_mask[:, step + 1, None]
                & in_bounds
                & (match_confidence > 1e-5)
            )
            position = next_position.clamp(-1.0, 1.0)
            tracks.append(position)
            visible.append(step_visible)
        return torch.stack(tracks, dim=1), torch.stack(visible, dim=1)

    def forward(
        self,
        frames: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if frames.ndim != 5 or frame_valid_mask.shape != frames.shape[:2]:
            raise ValueError("frames and valid mask must be [B,T,3,H,W] and [B,T]")
        gray = self._gray(frames)
        previous, current = gray[:, :-1], gray[:, 1:]
        flat_previous = previous.flatten(0, 1)
        flat_current = current.flatten(0, 1)
        horizontal, vertical, confidence = self._multiscale_correlation_flow(flat_previous, flat_current)
        horizontal = horizontal.view_as(previous)
        vertical = vertical.view_as(previous)
        confidence = confidence.view_as(previous)

        pair_valid = frame_valid_mask[:, 1:].bool() & frame_valid_mask[:, :-1].bool()
        pair_weight = pair_valid[:, :, None, None, None].to(horizontal.dtype)
        if timestamps is not None:
            if timestamps.shape != frame_valid_mask.shape:
                raise ValueError("timestamps must match frame_valid_mask")
            dt = (timestamps[:, 1:] - timestamps[:, :-1]).clamp_min(1e-3)
            # Normalize irregular sampling without amplifying the shortest gaps.
            dt_scale = dt.median(1, keepdim=True).values / dt
            pair_weight = pair_weight * dt_scale.clamp(0.25, 4.0)[:, :, None, None, None]
        horizontal = horizontal * pair_weight
        vertical = vertical * pair_weight
        flow = torch.cat((horizontal, vertical), dim=2)
        flow_tracks_xy, flow_tracks_visibility = self._flow_grid_tracks(
            flow, confidence * pair_weight, frame_valid_mask
        )

        height, width = self.flow_hw
        y = torch.linspace(-1.0, 1.0, height, device=flow.device, dtype=flow.dtype).view(1, 1, 1, height, 1)
        x = torch.linspace(-1.0, 1.0, width, device=flow.device, dtype=flow.dtype).view(1, 1, 1, 1, width)
        energy_sq = horizontal.square() + vertical.square()
        energy = energy_sq.clamp_min(1e-12).sqrt() - 1e-6
        residual_horizontal, residual_vertical = self._affine_residual_flow(
            horizontal,
            vertical,
            confidence * pair_weight,
        )
        residual_energy = (
            residual_horizontal.square() + residual_vertical.square()
        ).clamp_min(1e-12).sqrt() - 1e-6
        motion_tokens, motion_token_xy, motion_token_mask = self._build_motion_tokens(
            horizontal,
            vertical,
            energy,
            confidence * pair_weight,
            residual_horizontal,
            residual_vertical,
            residual_energy,
            pair_valid,
        )
        global_horizontal = horizontal.mean((-2, -1)).squeeze(-1)
        global_vertical = vertical.mean((-2, -1)).squeeze(-1)
        expansion = (horizontal * x + vertical * y).mean((-2, -1)).squeeze(-1)
        rotation = (-horizontal * y + vertical * x).mean((-2, -1)).squeeze(-1)
        motion_energy = energy.mean((-2, -1)).squeeze(-1)

        masks = _region_masks(height, width, flow.device, flow.dtype)
        region_horizontal = torch.einsum("btchw,rhw->btr", horizontal, masks)
        region_vertical = torch.einsum("btchw,rhw->btr", vertical, masks)
        region_energy = torch.einsum("btchw,rhw->btr", energy, masks)
        region_motion = torch.stack((region_horizontal, region_vertical, region_energy), dim=-1)
        descriptor = torch.cat(
            (
                global_horizontal[..., None], global_vertical[..., None], expansion[..., None],
                rotation[..., None], motion_energy[..., None], region_motion.flatten(2),
            ),
            dim=-1,
        )
        descriptor = descriptor * pair_valid[..., None].to(descriptor.dtype)
        valid_count = pair_valid.sum(1, keepdim=True).clamp_min(1).to(descriptor.dtype)
        summary = descriptor.sum(1) / valid_count
        recent = descriptor[:, -1]
        trend = recent - summary
        measured_state = torch.cat((summary, recent, trend), dim=-1)
        padding = summary.new_zeros(summary.shape[0], self.hidden_dim - 3 * self.descriptor_dim)
        flow_state = torch.cat((measured_state, padding), dim=-1)
        history_available = pair_valid.any(1)
        flow_state = flow_state * history_available[:, None].to(flow_state.dtype)
        prefix_indices = torch.tensor(
            [max(1, round(descriptor.shape[1] * fraction)) for fraction in (0.25, 0.50, 0.75, 1.0)],
            device=descriptor.device,
        ).clamp_max(descriptor.shape[1])
        cumulative = descriptor.cumsum(1)
        valid_cumulative = pair_valid.to(descriptor.dtype).cumsum(1).clamp_min(1.0)
        prefix_summary = torch.stack(
            [cumulative[:, index - 1] / valid_cumulative[:, index - 1, None] for index in prefix_indices], dim=1
        )
        prefix_recent = torch.stack([descriptor[:, index - 1] for index in prefix_indices], dim=1)
        prefix_trend = prefix_recent - prefix_summary
        prefix_measured = torch.cat((prefix_summary, prefix_recent, prefix_trend), dim=-1)
        prefix_padding = prefix_summary.new_zeros(
            prefix_summary.shape[0], prefix_summary.shape[1], self.hidden_dim - 3 * self.descriptor_dim
        )
        prefix_states = torch.cat((prefix_measured, prefix_padding), dim=-1)
        prefix_available = torch.stack(
            [pair_valid[:, :index].any(1) for index in prefix_indices], dim=1
        )
        prefix_states = prefix_states * prefix_available[..., None].to(prefix_states.dtype)
        return {
            "flow_field": flow,
            "residual_flow_field": torch.cat((residual_horizontal, residual_vertical), dim=2),
            "residual_motion_energy": residual_energy,
            "residual_motion_energy_mean": residual_energy.mean((-2, -1)).squeeze(-1),
            "flow_match_confidence": confidence * pair_weight,
            "flow_descriptor_sequence": descriptor,
            "flow_state": flow_state,
            "prefix_flow_states": prefix_states,
            "prefix_available": prefix_available,
            "prefix_fractions": flow_state.new_tensor((0.25, 0.50, 0.75, 1.0)),
            "pair_valid_mask": pair_valid,
            "history_available": history_available,
            "flow_grid_tracks_xy": flow_tracks_xy,
            "flow_grid_tracks_visibility": flow_tracks_visibility,
            "flow_motion_tokens": motion_tokens,
            "flow_motion_token_xy": motion_token_xy,
            "flow_motion_token_mask": motion_token_mask,
            "global_horizontal": global_horizontal,
            "global_vertical": global_vertical,
            "global_expansion": expansion,
            "global_rotation": rotation,
            "motion_energy": motion_energy,
            "region_motion": region_motion,
        }


class _IndependentFlowHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        cap: float,
        motion_feature_dim: int,
        target_context_dim: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.flow_projection = nn.Linear(hidden_dim, hidden_dim)
        self.motion_key = nn.Linear(motion_feature_dim, hidden_dim)
        self.motion_value = nn.Linear(motion_feature_dim, hidden_dim)
        self.target_embedding = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.02)
        self.target_context_projection = nn.Linear(target_context_dim, hidden_dim)
        self.target_prior_mix_logit = nn.Parameter(torch.zeros(()))
        self.context_projection = nn.Linear(2, hidden_dim)
        self.hidden = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.output = nn.Linear(hidden_dim, 1)
        self.gate = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.0)
        self.output_dim = int(output_dim)
        self.cap = float(cap)

    def forward(
        self,
        state: torch.Tensor,
        available: torch.Tensor,
        base_logits: torch.Tensor | None = None,
        *,
        motion_tokens: torch.Tensor | None = None,
        motion_token_mask: torch.Tensor | None = None,
        target_context: torch.Tensor | None = None,
        target_attention_prior: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if base_logits is None:
            base_logits = state.new_zeros(*state.shape[:-1], self.output_dim)
        if base_logits.shape != (*state.shape[:-1], self.output_dim):
            raise ValueError("base logits do not match target-conditioned flow state")
        uncertainty = torch.exp(-base_logits.detach().abs())
        context = torch.stack((base_logits.detach(), uncertainty), dim=-1)
        target = self.target_embedding.view(*([1] * (state.ndim - 1)), self.output_dim, state.shape[-1])
        query = target + self.context_projection(context)
        if target_context is not None:
            if target_context.shape[:-1] != query.shape[:-1]:
                raise ValueError("target context does not match flow target dimensions")
            query = query + self.target_context_projection(target_context.detach())

        attention = state.new_zeros(*state.shape[:-1], self.output_dim, 0)
        if motion_tokens is None:
            pooled = self.flow_projection(self.norm(state))[..., None, :]
        else:
            if motion_token_mask is None or motion_token_mask.shape != motion_tokens.shape[:-1]:
                raise ValueError("motion token mask must match motion tokens")
            key = self.motion_key(motion_tokens)
            value = self.motion_value(motion_tokens)
            score = torch.einsum("...ld,...md->...lm", query, key) / (key.shape[-1] ** 0.5)
            score = score.masked_fill(~motion_token_mask[..., None, :], -1e4)
            attention = score.softmax(-1) * motion_token_mask[..., None, :].to(score.dtype)
            if target_attention_prior is not None:
                if target_attention_prior.shape != attention.shape:
                    raise ValueError("target attention prior must match motion attention")
                prior = target_attention_prior.clamp_min(0.0)
                prior = prior * motion_token_mask[..., None, :].to(prior.dtype)
                prior = prior / prior.sum(-1, keepdim=True).clamp_min(1e-8)
                mix = torch.sigmoid(self.target_prior_mix_logit).clamp(0.10, 0.90)
                attention = (1.0 - mix) * attention + mix * prior
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
            pooled = torch.einsum("...lm,...md->...ld", attention, value)
        hidden = self.hidden(pooled + query)
        value = self.cap * torch.tanh(self.output(hidden).squeeze(-1)) * torch.sigmoid(self.gate(hidden).squeeze(-1))
        value = value * available[..., None].to(value.dtype)
        if return_attention:
            return value, attention
        return value


class TIDAGeometricFlowDecisionHeads(nn.Module):
    """Owner-isolated action/reason readers over the fixed flow measurement."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_actions: int = 4,
        num_reasons: int = 21,
        action_cap: float = 0.20,
        reason_cap: float = 0.15,
        motion_feature_dim: int = TIDAGeometricFlowEncoder.motion_token_dim,
        target_context_dim: int = 384,
    ) -> None:
        super().__init__()
        self.action_head = _IndependentFlowHead(
            hidden_dim, num_actions, action_cap, motion_feature_dim, target_context_dim
        )
        self.reason_head = _IndependentFlowHead(
            hidden_dim, num_reasons, reason_cap, motion_feature_dim, target_context_dim
        )
        self.action_output = self.action_head.output
        self.reason_output = self.reason_head.output

    def action_parameters(self):
        return self.action_head.parameters()

    def reason_parameters(self):
        return self.reason_head.parameters()

    def forward(
        self,
        flow_state: torch.Tensor,
        history_available: torch.Tensor,
        action_base_logits: torch.Tensor | None = None,
        reason_base_logits: torch.Tensor | None = None,
        *,
        motion_tokens: torch.Tensor | None = None,
        motion_token_mask: torch.Tensor | None = None,
        action_target_context: torch.Tensor | None = None,
        reason_target_context: torch.Tensor | None = None,
        action_target_attention_prior: torch.Tensor | None = None,
        reason_target_attention_prior: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        action, action_attention = self.action_head(
            flow_state,
            history_available,
            action_base_logits,
            motion_tokens=motion_tokens,
            motion_token_mask=motion_token_mask,
            target_context=action_target_context,
            target_attention_prior=action_target_attention_prior,
            return_attention=True,
        )
        reason, reason_attention = self.reason_head(
            flow_state,
            history_available,
            reason_base_logits,
            motion_tokens=motion_tokens,
            motion_token_mask=motion_token_mask,
            target_context=reason_target_context,
            target_attention_prior=reason_target_attention_prior,
            return_attention=True,
        )
        return {
            "geometric_action_delta": action,
            "geometric_reason_delta": reason,
            "geometric_action_delta_rms": action.float().square().mean().sqrt(),
            "geometric_reason_delta_rms": reason.float().square().mean().sqrt(),
            "geometric_action_motion_attention": action_attention,
            "geometric_reason_motion_attention": reason_attention,
            "geometric_action_motion_attention_entropy": -(
                action_attention * action_attention.clamp_min(1e-8).log()
            ).sum(-1),
            "geometric_reason_motion_attention_entropy": -(
                reason_attention * reason_attention.clamp_min(1e-8).log()
            ).sum(-1),
        }

    def forward_prefixes(
        self,
        prefix_flow_states: torch.Tensor,
        prefix_available: torch.Tensor,
        action_base_logits: torch.Tensor | None = None,
        reason_base_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if action_base_logits is not None:
            action_base_logits = action_base_logits[:, None].expand(-1, prefix_flow_states.shape[1], -1)
        if reason_base_logits is not None:
            reason_base_logits = reason_base_logits[:, None].expand(-1, prefix_flow_states.shape[1], -1)
        return {
            "geometric_prefix_action_delta": self.action_head(prefix_flow_states, prefix_available, action_base_logits),
            "geometric_prefix_reason_delta": self.reason_head(prefix_flow_states, prefix_available, reason_base_logits),
        }
