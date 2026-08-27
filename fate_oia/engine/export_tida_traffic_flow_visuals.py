from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image


def _ego_compensated_paths(track: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    pair_visible = visible[1:] & visible[:-1]
    step = track[1:] - track[:-1]
    masked = step.masked_fill(~pair_visible[..., None], float("nan"))
    camera_flow = torch.nanmedian(masked, dim=1).values.nan_to_num()
    residual = (step - camera_flow[:, None]) * pair_visible[..., None]
    return torch.cat((torch.zeros_like(track[:1]), residual.cumsum(0)), dim=0)


def _read_manifest(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["file_name"]).lower()] = row
    return rows


def _plot_case(
    output_path: Path, image_path: Path, track: torch.Tensor,
    visible: torch.Tensor, title: str,
) -> None:
    paths = _ego_compensated_paths(track.float(), visible.bool())
    speed = paths[-1].square().sum(-1).sqrt()
    keep = speed.topk(min(16, len(speed))).indices
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    if image_path.exists():
        with Image.open(image_path) as image:
            axes[0].imshow(image.convert("RGB"))
    axes[0].set_title("BDD-OIA decision frame")
    axes[0].axis("off")
    colors = plt.cm.turbo(torch.linspace(0, 1, len(keep)).numpy())
    for color, point in zip(colors, keep.tolist()):
        valid = visible[:, point].cpu().numpy()
        xy = paths[:, point].cpu().numpy()
        axes[1].plot(xy[valid, 0], -xy[valid, 1], color=color, alpha=0.85)
        if valid.any():
            axes[1].scatter(xy[valid, 0][-1], -xy[valid, 1][-1], color=color, s=18)
    axes[1].axhline(0, color="#777777", linewidth=0.6)
    axes[1].axvline(0, color="#777777", linewidth=0.6)
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].set_title("Ego-compensated traffic trajectories")
    axes[1].set_xlabel("lateral residual motion")
    axes[1].set_ylabel("longitudinal residual motion")
    figure.suptitle(title)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrector-test", required=True)
    parser.add_argument("--object-track-store", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=12)
    args = parser.parse_args()

    result = torch.load(args.corrector_test, map_location="cpu")
    store = torch.load(args.object_track_store, map_location="cpu")
    manifest = _read_manifest(Path(args.manifest))
    track_index = {str(name).lower(): index for index, name in enumerate(store["file_names"])}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = result["file_names"]
    mask = result["traffic_deployed_mask"].bool()
    target = result["action_target"].bool()
    baseline = result["no_traffic_deploy_prediction"].bool()
    deployed = result["full_prediction"].bool()
    probability = result["full_probability"].float()
    selected = mask.nonzero(as_tuple=False).tolist()
    selected.sort(key=lambda item: abs(float(probability[item[0], item[1]]) - 0.5), reverse=True)
    case_rows = []
    for case_number, (sample, action) in enumerate(selected[: args.max_cases]):
        name = str(names[sample])
        row = manifest.get(name.lower(), {})
        status = "recovered" if baseline[sample, action] != target[sample, action] and deployed[sample, action] == target[sample, action] else "damaged"
        image_name = f"case_{case_number:03d}_{status}.png"
        title = (
            f"{name} | action={action} | {status} | "
            f"base={int(baseline[sample, action])} traffic={int(deployed[sample, action])} "
            f"gt={int(target[sample, action])} p={float(probability[sample, action]):.3f}"
        )
        index = track_index[name.lower()]
        _plot_case(
            output / image_name, Path(row.get("target_image_path", "")),
            store["tracks_xy"][index], store["visibility"][index], title,
        )
        case_rows.append({
            "file_name": name, "action": int(action), "status": status,
            "base": bool(baseline[sample, action]),
            "traffic": bool(deployed[sample, action]),
            "target": bool(target[sample, action]),
            "traffic_probability": float(probability[sample, action]),
            "source_batch": row.get("source_batch"), "visual": image_name,
        })
    (output / "traffic_flow_cases.json").write_text(
        json.dumps(case_rows, indent=2), encoding="utf-8"
    )
    cards = "\n".join(
        f'<article><h2>{html.escape(row["file_name"])}: {row["status"]}</h2>'
        f'<img src="{row["visual"]}" alt="traffic flow case"></article>'
        for row in case_rows
    )
    (output / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>TIDA Traffic Flow Evidence</title>"
        "<style>body{font:16px Georgia;background:#f4efe6;color:#17221d;margin:2rem}"
        "article{max-width:1200px;margin:0 auto 3rem}img{width:100%;border:1px solid #897}</style>"
        "<h1>Traffic-flow decision evidence</h1>"
        "<p>Right panel shows ego-camera-compensated trajectories, not pixel saliency.</p>" + cards,
        encoding="utf-8",
    )
    print(json.dumps({"cases": len(case_rows), "output_dir": str(output)}))


if __name__ == "__main__":
    main()
