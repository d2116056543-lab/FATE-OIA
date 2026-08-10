from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import yaml

from fate_oia.engine.train_aie_oia import build_model as build_aie_model, canonical_model_state_dict
from fate_oia.models.dice_oia_model import DICEOIAModel


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def tensor_state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.named_parameters()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_dice_model(cfg: dict, checkpoint_path: str | Path, device: torch.device) -> DICEOIAModel:
    base = build_aie_model(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    result = base.load_state_dict(canonical_model_state_dict(state), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"base checkpoint mismatch: {result}")
    dice = cfg["dice"]
    return DICEOIAModel(
        base, dim=int(cfg["primary"]["dim"]), num_layers=len(cfg["backbone"]["selected_layers"]),
        num_predicates=base.foundation.predicate_head.num_predicates,
        probes_per_action=int(cfg["evidence"]["probes_per_action"]),
        predicate_strength_max=float(dice["predicate_strength_max"]),
        predicate_presence_floor=float(dice["predicate_presence_floor"]),
        c_max_per_atom=float(dice["c_max_per_atom"]), total_action_cap=float(dice["total_action_cap"]),
        base_forward_kwargs={"action_scale": 1.0, "reason_scale": 0.60},
    ).to(device)
