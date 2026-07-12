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


def restore_verified_dino_vproj_aliases(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize checkpoints to the current DINO runtime alias contract.

    Some MOSAIC epochs were saved before ``Attention.save_proj`` registered
    ``vproj`` in the module state dict, while later epochs contain a duplicate
    alias.  The alias is semantically the same projection as ``proj``.  Verify
    any stored alias, then materialize it from ``proj`` when absent so strict
    loading is valid for both checkpoint generations.
    """
    cleaned = remove_verified_dino_vproj_aliases(state_dict)
    for key, value in list(cleaned.items()):
        if ".attn.proj." not in key:
            continue
        alias_key = key.replace(".attn.proj.", ".attn.vproj.")
        if alias_key not in cleaned:
            cleaned[alias_key] = value.clone()
    return cleaned


def load_mosaic_model_state_strict(model, state_dict: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(restore_verified_dino_vproj_aliases(state_dict), strict=True)
