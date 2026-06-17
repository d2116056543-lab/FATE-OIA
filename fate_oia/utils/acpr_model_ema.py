from __future__ import annotations

import torch


class ModelEMA:
    """CPU-backed EMA helper.

    Keeping a full GPU copy of the model would distort the memory profile of the
    direct-image training run. The shadow state stays on CPU and is temporarily
    applied to the live model only for EMA evaluation.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.995) -> None:
        self.decay = float(decay)
        self.shadow = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        current = model.state_dict()
        for key, value in current.items():
            value_cpu = value.detach().cpu()
            if key not in self.shadow:
                self.shadow[key] = value_cpu.clone()
                continue
            if torch.is_floating_point(value_cpu):
                self.shadow[key].mul_(self.decay).add_(value_cpu, alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(value_cpu)
        self.num_updates += 1

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        target_state = model.state_dict()
        loaded = {}
        for key, value in target_state.items():
            shadow = self.shadow.get(key)
            loaded[key] = (shadow.to(device=value.device, dtype=value.dtype) if shadow is not None else value)
        model.load_state_dict(loaded, strict=False)
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        target_state = model.state_dict()
        restored = {k: v.to(device=target_state[k].device, dtype=target_state[k].dtype) for k, v in backup.items() if k in target_state}
        model.load_state_dict(restored, strict=False)
