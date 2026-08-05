from __future__ import annotations

import torch
from torch import Tensor


ACTION_PERMUTATION = torch.tensor([0, 1, 3, 2])
REASON_PERMUTATION = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 9, 10, 11, 12, 13, 14])


def mirror_lens_batch(images: Tensor, action: Tensor, reason: Tensor, state_target: Tensor | None = None, map_target: Tensor | None = None) -> dict[str, Tensor]:
    result = {
        "image": torch.flip(images, dims=(-1,)),
        "action": action[:, ACTION_PERMUTATION.to(action.device)],
        "reason": reason[:, REASON_PERMUTATION.to(reason.device)],
    }
    if state_target is not None:
        result["state_target"] = state_target[:, REASON_PERMUTATION.to(state_target.device)]
    if map_target is not None:
        result["map_target"] = torch.flip(map_target[:, REASON_PERMUTATION.to(map_target.device)], dims=(-1,))
    return result
