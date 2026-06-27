from __future__ import annotations

import torch


def compute_eventness(frames: torch.Tensor) -> torch.Tensor:
    """Deterministic observed-frame eventness from RGB frame deltas.

    This is intentionally lightweight and cache-free. It only uses observed
    frames and can be enabled for anchor selection after profiling.
    """

    if frames.ndim != 5:
        raise ValueError(f"Expected [B,T,3,H,W], got {tuple(frames.shape)}")
    delta = (frames[:, 1:] - frames[:, :-1]).abs().mean(dim=(2, 3, 4))
    first = delta[:, :1]
    return torch.cat([first, delta], dim=1)


def select_event_anchors(frames: torch.Tensor, anchor_count: int = 6) -> torch.Tensor:
    eventness = compute_eventness(frames)
    b, t = eventness.shape
    base = torch.tensor([0, t - 1], device=frames.device)
    k = max(0, anchor_count - 2)
    if k == 0:
        return base.view(1, -1).expand(b, -1)
    scores = eventness.clone()
    scores[:, 0] = -1
    scores[:, -1] = -1
    top = scores.topk(k=min(k, max(1, t - 2)), dim=1).indices
    anchors = torch.cat([base[:1].view(1, 1).expand(b, 1), top, base[1:].view(1, 1).expand(b, 1)], dim=1)
    return anchors.sort(dim=1).values
