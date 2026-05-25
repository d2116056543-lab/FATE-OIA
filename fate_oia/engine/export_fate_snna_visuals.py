from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch


def _save_heatmap(path: Path, heatmap: torch.Tensor) -> None:
    heat = heatmap.detach().cpu().float()
    if heat.ndim == 2:
        heat = heat.unsqueeze(0)
    heat = heat.squeeze(0)
    heat = heat - heat.min()
    heat = heat / heat.max().clamp_min(1e-6)
    arr = (heat.numpy() * 255).astype("uint8")
    Image.fromarray(arr, mode="L").save(path)


def export_visual_rows(
    output_dir: Path,
    *,
    samples: int = 2,
    labels: list[int] | None = None,
    methods: list[str] | None = None,
    patch_grid: tuple[int, int] = (45, 80),
) -> list[dict[str, object]]:
    """Write minimal attribution visual artifacts.

    This script is intentionally checkpoint-friendly but does not pretend to
    produce paper evidence without a trained model. When no model wrapper is
    supplied, it exports deterministic smoke heatmaps so downstream rendering
    and index schemas can be validated before full training.
    """
    labels = labels or [0, 4]
    methods = methods or ["label_attention", "grad_x_attention"]
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for sample_idx in range(samples):
        for label_idx in labels:
            for method in methods:
                seed = sample_idx * 1000 + label_idx * 10 + len(method)
                gen = torch.Generator().manual_seed(seed)
                heatmap = torch.rand(patch_grid, generator=gen)
                png = visual_dir / f"sample_{sample_idx:04d}_label_{label_idx}_{method}.png"
                js = png.with_suffix(".json")
                _save_heatmap(png, heatmap)
                topk = torch.topk(heatmap.flatten(), k=min(10, heatmap.numel())).indices.tolist()
                rec = {
                    "file_name": f"sample_{sample_idx:04d}.jpg",
                    "split": "smoke",
                    "label_type": "action" if label_idx < 4 else "reason",
                    "label_idx": label_idx,
                    "label_score": None,
                    "gt_label": None,
                    "attribution_method": method,
                    "patch_grid": list(patch_grid),
                    "topk_patch_indices": topk,
                    "grounding_hit": None,
                    "smoke_artifact": True,
                    "boundary": "schema smoke only; trained checkpoint attribution requires --checkpoint integration",
                }
                js.write_text(json.dumps(rec, indent=2), encoding="utf-8")
                rows.append(rec)
    with (visual_dir / "index.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Export FATE-SNNA visual audit artifacts.")
    ap.add_argument("--checkpoint", default="", help="Trained FATE-OIA checkpoint. Reserved for full checkpoint-dependent export.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=2)
    ap.add_argument("--label_indices", default="0,4")
    ap.add_argument("--methods", default="label_attention,grad_x_attention")
    ap.add_argument("--patch_grid_h", type=int, default=45)
    ap.add_argument("--patch_grid_w", type=int, default=80)
    args = ap.parse_args()
    if args.checkpoint:
        # The CLI exists now so training jobs can call it consistently; full
        # checkpoint wiring needs the exact saved model/backbone pair from the
        # completed run.
        print(json.dumps({"event": "fate_snna_checkpoint_mode_reserved", "checkpoint": args.checkpoint}), flush=True)
    labels = [int(x) for x in args.label_indices.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    rows = export_visual_rows(Path(args.output_dir), samples=args.max_samples, labels=labels, methods=methods, patch_grid=(args.patch_grid_h, args.patch_grid_w))
    print(json.dumps({"event": "fate_snna_visual_export", "rows": len(rows), "output_dir": args.output_dir}), flush=True)


if __name__ == "__main__":
    main()
