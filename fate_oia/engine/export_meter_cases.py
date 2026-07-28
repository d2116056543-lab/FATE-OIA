from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image

from fate_oia.datasets.meter_dataset import METERDataset
from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.transforms_meter import meter_image_transform
from fate_oia.utils.meter_artifacts import write_json


def _heatmap(values: torch.Tensor, path: Path, size: tuple[int, int]) -> None:
    values = values.detach().float().cpu().reshape(45, 80)
    values = (values - values.min()) / (values.max() - values.min()).clamp_min(1e-6)
    image = Image.fromarray((values.numpy() * 255).astype("uint8"), mode="L").resize(size, Image.Resampling.BILINEAR)
    image.save(path)


@torch.no_grad()
def export_case(model: METEROIAModel, item: dict[str, Any], out_dir: Path, device: torch.device) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = item["image"].unsqueeze(0).to(device)
    output = model(image, progress=1.0)
    Image.open(item["image_path"]).convert("RGB").save(out_dir / "original.jpg")
    names = ["forward", "stop", "left", "right"]
    for index, name in enumerate(names):
        map_value = output["action_factor_weights"][0, index].unsqueeze(-1) * output["factor_support_map"][0]
        _heatmap(map_value.sum(0), out_dir / f"evidence_{name}.png", (640, 360))
    top_reason = torch.sigmoid(output["reason_logits_final"])[0].topk(3).indices.tolist()
    for index in top_reason:
        _heatmap(output["factor_support_map"][0, index], out_dir / f"evidence_top_reason_{index}.png", (640, 360))
    table = {
        "file_name": item["file_name"],
        "action_logits": output["action_logits_final"][0].cpu().tolist(),
        "reason_logits": output["reason_logits_final"][0].cpu().tolist(),
        "action_factor_contributions": output["action_factor_contributions"][0].cpu().tolist(),
        "factor_reliability": output["factor_reliability"][0].cpu().tolist(),
    }
    write_json(out_dir / "action_set_table.json", table)
    write_json(out_dir / "graph_edges.json", {"available": False, "reason": "METER uses no graph/PMI adjacency by contract"})
    write_json(out_dir / "deletion_audit.json", {"available": True, "selected_vs_control": "see counterfactual_events.jsonl"})
    report = "<html><body><h1>METER-OIA case</h1><p>" + html.escape(str(item["file_name"])) + "</p>"
    report += "<img src='original.jpg' width='640'><h2>Action evidence</h2>"
    report += "".join(f"<img src='evidence_{name}.png' width='320'>" for name in names)
    report += "</body></html>"
    (out_dir / "report.html").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_cases", type=int, default=8)
    args = parser.parse_args()
    model = METEROIAModel().to(args.device)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload["model"])
    dataset = METERDataset(data_root=args.data_root, raw_root=args.raw_root, split="test", transform=meter_image_transform())
    for index in range(min(args.max_cases, len(dataset))):
        export_case(model, dataset[index], Path(args.output_dir) / f"case_{index:04d}", torch.device(args.device))


if __name__ == "__main__":
    main()
