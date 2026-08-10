from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from fate_oia.engine.train_aie_oia import build_model as build_aie_model, canonical_model_state_dict, collate
from fate_oia.models.vetra_oia_model import VETRAOIAModel


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def tensor_state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.named_parameters()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_vetra_model(cfg: dict, checkpoint_path: str | Path, device: torch.device,
                      use_mock_dino: bool = False) -> VETRAOIAModel:
    base = build_aie_model(cfg, device, use_mock_dino=use_mock_dino)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    result = base.load_state_dict(canonical_model_state_dict(state), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"source checkpoint mismatch: {result}")
    model = VETRAOIAModel(
        base, base.foundation.predicate_head.names, str(cfg["primary"]["reason_grammar"]),
        dim=int(cfg["primary"]["dim"]), num_layers=len(cfg["backbone"]["selected_layers"]),
        correction_cap=float(cfg["vetra"]["correction_cap"]),
        base_forward_kwargs={"action_scale": 1.0, "reason_scale": .60},
    )
    return model.to(device)


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int, cfg: dict, generator=None):
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=collate,
                  pin_memory=bool(cfg["data"]["pin_memory"]), generator=generator,
                  persistent_workers=bool(cfg["data"]["persistent_workers"]) and workers > 0)
    if workers:
        kwargs["prefetch_factor"] = int(cfg["data"]["prefetch_factor"])
    return DataLoader(dataset, **kwargs)


def alpha_schedule(update: int, total_updates: int, initial: float = .02, full_ratio: float = .20) -> float:
    progress = update / max(total_updates, 1)
    if progress <= .05:
        return initial + (.10 - initial) * progress / .05
    if progress <= full_ratio:
        return .10 + .90 * (progress - .05) / max(full_ratio - .05, 1e-8)
    return 1.0


def make_scheduler(optimizer, total_updates: int, warmup_ratio: float, min_lr_ratio: float):
    warmup = max(1, int(total_updates * warmup_ratio))
    def factor(step):
        if step < warmup:
            return max((step + 1) / warmup, 1e-8)
        progress = min(max((step - warmup) / max(total_updates - warmup, 1), 0.0), 1.0)
        return min_lr_ratio + (1 - min_lr_ratio) * .5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
