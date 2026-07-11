from __future__ import annotations

import torch


def remove_verified_dino_vproj_aliases(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove only runtime aliases that exactly duplicate DINO attention projections."""
    cleaned = dict(state_dict)
    alias_keys = [key for key in cleaned if ".attn.vproj." in key]
    for alias_key in alias_keys:
        projection_key = alias_key.replace(".attn.vproj.", ".attn.proj.")
        if projection_key not in cleaned or not torch.equal(cleaned[alias_key], cleaned[projection_key]):
            raise RuntimeError(f"unverified DINO vproj checkpoint alias: {alias_key}")
        del cleaned[alias_key]
    return cleaned


def load_mosaic_model_state_strict(model, state_dict: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(remove_verified_dino_vproj_aliases(state_dict), strict=True)
