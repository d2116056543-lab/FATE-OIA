from __future__ import annotations

import torch
import torch.nn.functional as F


def _resize_mask(mask: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if mask.shape[-2:] == target_hw:
        return mask.float()
    return F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=target_hw, mode="nearest").squeeze(0).squeeze(0)


def attention_grounding_bce(attention: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """BCE loss between normalized attention map and binary grounding mask."""
    if attention.ndim != 2:
        raise ValueError("attention must be [H,W]")
    target = _resize_mask(target_mask, attention.shape[-2:]).to(attention.device)
    attn = attention.float()
    attn = (attn - attn.min()) / (attn.max() - attn.min()).clamp_min(eps)
    return F.binary_cross_entropy(attn.clamp(eps, 1 - eps), target.float())


def pointing_game_hit(attention: torch.Tensor, target_mask: torch.Tensor) -> float:
    target = _resize_mask(target_mask, attention.shape[-2:]).to(attention.device)
    flat_idx = int(torch.argmax(attention.reshape(-1)).item())
    y = flat_idx // attention.shape[-1]
    x = flat_idx % attention.shape[-1]
    return float(target[y, x].item() > 0.5)


def mask_iou(pred_mask: torch.Tensor, target_mask: torch.Tensor, threshold: float = 0.5) -> float:
    target = _resize_mask(target_mask, pred_mask.shape[-2:]).to(pred_mask.device) > 0.5
    pred = pred_mask.float() >= threshold
    inter = (pred & target).float().sum()
    union = (pred | target).float().sum().clamp_min(1.0)
    return float((inter / union).item())