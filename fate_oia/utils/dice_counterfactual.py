from __future__ import annotations

import torch
from torch import Tensor

from fate_oia.losses.dice_losses import certificate_targets


def mass_matched_control(selected: Tensor, region: Tensor, offset: int) -> Tensor:
    """Deterministic same-region control with selected mass and no global substitution."""
    region = region.to(selected).clamp_min(0)
    candidate = region * torch.roll(1 - selected.clamp(0, 1), shifts=int(offset), dims=-1)
    candidate = candidate / candidate.sum(-1, keepdim=True).clamp_min(1e-8)
    return candidate * selected.sum(-1, keepdim=True)


def hard_region_topk(probability: Tensor, region: Tensor, topk: int = 64) -> Tensor:
    """Turn an attention distribution into a deletion mask with real patch support."""
    score=probability*region.to(probability)
    valid=int((region>0).sum().item()); count=min(int(topk),valid,score.shape[-1])
    if count<=0: return torch.zeros_like(probability)
    index=score.topk(count).indices
    return torch.zeros_like(probability).scatter(0,index,1.0)


def directional_certificate(selected_drop: Tensor, same_region_1: Tensor, same_region_2: Tensor,
                            wrong_probe: Tensor, wrong_action: Tensor, temperature: float = .05) -> dict[str, Tensor]:
    controls = torch.stack((same_region_1, same_region_2, wrong_probe, wrong_action), -1)
    return {**certificate_targets(selected_drop, controls, temperature), "controls": controls}


def choose_round_robin_atoms(atom_correction: Tensor, update: int, max_actions_per_sample: int = 2,
                             selected_samples: int | None = None) -> list[tuple[int, int, int]]:
    rows = []
    count=min(atom_correction.shape[0],selected_samples if selected_samples is not None else max(1,atom_correction.shape[0]//2))
    for sample in range(count):
        for step in range(min(max_actions_per_sample, atom_correction.shape[1])):
            action = (int(update) + sample + step) % atom_correction.shape[1]
            probe = int(atom_correction[sample, action].abs().argmax())
            rows.append((sample, action, probe))
    return rows
