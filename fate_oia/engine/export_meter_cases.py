from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.transforms_meter import meter_image_transform
from fate_oia.utils.meter_artifacts import load_checkpoint
from fate_oia.utils.meter_config import load_meter_config


ACTION_NAMES = ("forward", "stop", "left", "right")


def _save_heatmap(value: torch.Tensor, path: Path) -> None:
    image = value.detach().float().reshape(45, 80)
    image = image - image.min()
    image = image / image.max().clamp_min(1e-8)
    Image.fromarray((image * 255).byte().cpu().numpy()).resize(
        (640, 360), Image.BILINEAR
    ).save(path)


@torch.no_grad()
def export_case(
    model: METEROIAModel,
    image_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image.save(output_root / "original.jpg")
    tensor, _ = meter_image_transform()(image)
    output = model(tensor.unsqueeze(0).to(device), progress=1.0)
    contribution = output["action_factor_contributions"][0].detach().cpu()
    anchor = output["factor_anchor_map"][0].detach().cpu()
    state = output["factor_state_prob"][0].detach().cpu()
    reliability = output["factor_reliability"][0].detach().cpu()
    for action_id, action_name in enumerate(ACTION_NAMES):
        top = torch.topk(contribution[action_id].abs(), k=3).indices.tolist()
        for rank, factor_id in enumerate(top):
            _save_heatmap(
                anchor[factor_id],
                output_root / f"{action_name}_factor_{rank}_{factor_id}.png",
            )
    payload = {
        "action_visual": output["action_logits_visual"][0].detach().cpu().tolist(),
        "action_final": output["action_logits_final"][0].detach().cpu().tolist(),
        "reason_global": output["reason_logits_global"][0].detach().cpu().tolist(),
        "reason_final": output["reason_logits_final"][0].detach().cpu().tolist(),
        "factor_state_probability": state.tolist(),
        "factor_observability": output["factor_observability"][0]
        .detach()
        .cpu()
        .tolist(),
        "factor_reliability": reliability.tolist(),
        "factor_to_action_contribution": contribution.tolist(),
        "factor_to_action_weight": output["action_factor_weights"][0]
        .detach()
        .cpu()
        .tolist(),
        "factor_null_mass": output["factor_null_mass"][0].detach().cpu().tolist(),
    }
    (output_root / "tesa_case.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_meter_config(args.config)
    device = torch.device(args.device)
    model = METEROIAModel(
        dim=int(config["model"]["dim"]),
        action_dim=int(config["model"]["action_dim"]),
        reason_dim=int(config["model"]["reason_dim"]),
        selected_layers=tuple(config["backbone"]["selected_layers"]),
        pretrained_weights=config["backbone"]["pretrained_weights"],
        factor_rank=int(config["model"].get("factor_rank", 16)),
    ).to(device)
    load_checkpoint(args.checkpoint, model=model)
    model.eval()
    export_case(model, args.image, args.output_dir, device)


if __name__ == "__main__":
    main()
