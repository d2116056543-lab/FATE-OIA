from __future__ import annotations

import torch


def readout_factor_off(output: dict[str, torch.Tensor], factor: int) -> torch.Tensor:
    return output["logits"] - output["factor_contribution"][..., factor]


@torch.no_grad()
def input_intervention(model, inputs, kind: str, *, generator: torch.Generator | None = None):
    history = inputs.history_rgb.clone(); valid = inputs.valid.clone(); times = inputs.actual_t.clone()
    if kind == "history_off": valid[:, :-1] = False
    elif kind == "repeated_last": history[:] = torch.nn.functional.interpolate(inputs.target_rgb, (256, 448)).unsqueeze(1)
    elif kind in ("shuffle", "reverse"):
        count = history.shape[1]
        order = torch.arange(count, device=history.device).flip(0) if kind == "reverse" else torch.randperm(count, generator=generator, device=history.device)
        history, valid[:, :-1] = history[:, order], valid[:, order]
    elif kind.startswith("frame_delete:"):
        valid[:, int(kind.split(":")[1])] = False
    else: raise ValueError(kind)
    changed = type(inputs)(inputs.target_rgb, history, times, valid, inputs.geometry_meta)
    return model(changed)
