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

    def __init__(self, hidden_dim: int = 64, flow_hw: tuple[int, int] = (45, 80)) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.flow_hw = tuple(int(value) for value in flow_hw)
        if self.hidden_dim < self.descriptor_dim:
            raise ValueError(f"hidden_dim must be at least {self.descriptor_dim}")
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_x.transpose(-1, -2).contiguous(), persistent=False)
        self.register_buffer("channel_scale", torch.tensor([0.229, 0.224, 0.225]), persistent=False)
        self.register_buffer("luma", torch.tensor([0.2989, 0.5870, 0.1140]), persistent=False)

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
        midpoint = 0.5 * (previous + current)
        flat = midpoint.flatten(0, 1)
        padded = F.pad(flat, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(padded, self.sobel_x).view_as(midpoint)
        grad_y = F.conv2d(padded, self.sobel_y).view_as(midpoint)
        grad_t = current - previous
        denominator = grad_x.square() + grad_y.square() + 2e-3
        horizontal = (-grad_t * grad_x / denominator).tanh()
        vertical = (-grad_t * grad_y / denominator).tanh()
        confidence = (denominator - 2e-3).clamp_min(0.0)
        confidence = confidence / (confidence + 0.02)
        horizontal = horizontal * confidence
        vertical = vertical * confidence

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

        height, width = self.flow_hw
        y = torch.linspace(-1.0, 1.0, height, device=flow.device, dtype=flow.dtype).view(1, 1, 1, height, 1)
        x = torch.linspace(-1.0, 1.0, width, device=flow.device, dtype=flow.dtype).view(1, 1, 1, 1, width)
        energy = (horizontal.square() + vertical.square() + 1e-8).sqrt()
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
        padding = summary.new_zeros(summary.shape[0], self.hidden_dim - self.descriptor_dim)
        flow_state = torch.cat((summary, padding), dim=-1)
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
        prefix_padding = prefix_summary.new_zeros(
            prefix_summary.shape[0], prefix_summary.shape[1], self.hidden_dim - self.descriptor_dim
        )
        prefix_states = torch.cat((prefix_summary, prefix_padding), dim=-1)
        prefix_available = torch.stack(
            [pair_valid[:, :index].any(1) for index in prefix_indices], dim=1
        )
        prefix_states = prefix_states * prefix_available[..., None].to(prefix_states.dtype)
        return {
            "flow_field": flow,
            "flow_descriptor_sequence": descriptor,
            "flow_state": flow_state,
            "prefix_flow_states": prefix_states,
            "prefix_available": prefix_available,
            "prefix_fractions": flow_state.new_tensor((0.25, 0.50, 0.75, 1.0)),
            "pair_valid_mask": pair_valid,
            "history_available": history_available,
            "global_horizontal": global_horizontal,
            "global_vertical": global_vertical,
            "global_expansion": expansion,
            "global_rotation": rotation,
            "motion_energy": motion_energy,
            "region_motion": region_motion,
        }


class _IndependentFlowHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, cap: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.output = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.cap = float(cap)

    def forward(self, state: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        value = self.cap * torch.tanh(self.output(self.hidden(self.norm(state))))
        return value * available[..., None].to(value.dtype)


class TIDAGeometricFlowDecisionHeads(nn.Module):
    """Owner-isolated action/reason readers over the fixed flow measurement."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_actions: int = 4,
        num_reasons: int = 21,
        action_cap: float = 0.20,
        reason_cap: float = 0.15,
    ) -> None:
        super().__init__()
        self.action_head = _IndependentFlowHead(hidden_dim, num_actions, action_cap)
        self.reason_head = _IndependentFlowHead(hidden_dim, num_reasons, reason_cap)
        self.action_output = self.action_head.output
        self.reason_output = self.reason_head.output

    def action_parameters(self):
        return self.action_head.parameters()

    def reason_parameters(self):
        return self.reason_head.parameters()

    def forward(self, flow_state: torch.Tensor, history_available: torch.Tensor) -> dict[str, torch.Tensor]:
        action = self.action_head(flow_state, history_available)
        reason = self.reason_head(flow_state, history_available)
        return {
            "geometric_action_delta": action,
            "geometric_reason_delta": reason,
            "geometric_action_delta_rms": action.float().square().mean().sqrt(),
            "geometric_reason_delta_rms": reason.float().square().mean().sqrt(),
        }

    def forward_prefixes(
        self, prefix_flow_states: torch.Tensor, prefix_available: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            "geometric_prefix_action_delta": self.action_head(prefix_flow_states, prefix_available),
            "geometric_prefix_reason_delta": self.reason_head(prefix_flow_states, prefix_available),
        }
