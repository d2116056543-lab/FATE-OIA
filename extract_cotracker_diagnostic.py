from __future__ import annotations

import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fate_oia.datasets.bdd_oia_video import BDDOIAVideoDataset, tida_video_collate
from fate_oia.transforms import IMAGENET_MEAN, IMAGENET_STD


WORKTREE = Path(r"E:\sbw\FATE_Drive\fate_oia_tida_relational_flow_v8_10k_worktree")
EPOCH = Path(r"F:\FATE_Drive_runs\tida_relational_v8_2_pilot5584x1_retry2\epoch_000")


def main() -> None:
    # Keep process-spawning work inside main: Windows DataLoader workers import this module.
    dataset = BDDOIAVideoDataset(
        WORKTREE / "artifacts/tida_10k_v8/tida_10k_primary_manifest.jsonl",
        "test",
        training=False,
    )
    loader = DataLoader(
        dataset,
        # CoTracker3 offline currently uses view() on an expanded coordinate
        # tensor for B>1. Keep B=1 rather than patching third-party semantics.
        batch_size=1,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=tida_video_collate,
    )
    model = torch.hub.load(
        r"E:\sbw\deps\co-tracker", "cotracker3_offline", source="local"
    )
    model = model.cuda().eval()
    mean = IMAGENET_MEAN.cuda()[None, None]
    std = IMAGENET_STD.cuda()[None, None]
    all_features = []
    all_tracks = []
    all_visibility = []
    all_names = []
    started = time.perf_counter()
    total_batches = len(loader)

    for batch_index, batch in enumerate(loader, start=1):
        context = batch["context_images"].cuda(non_blocking=True)
        target = batch["target_image"].cuda(non_blocking=True)
        context = (context * std + mean).clamp(0, 1)
        target = (target[:, None] * std + mean).clamp(0, 1)[:, 0]
        target = F.interpolate(
            target, size=context.shape[-2:], mode="bilinear", align_corners=False
        )
        video = torch.cat((context, target[:, None]), dim=1) * 255.0
        with torch.no_grad():
            tracks, visibility = model(video, grid_size=8)
        visibility = visibility > 0.5

        height, width = video.shape[-2:]
        xy = tracks.clone()
        xy[..., 0] = 2.0 * xy[..., 0] / max(width - 1, 1) - 1.0
        xy[..., 1] = 2.0 * xy[..., 1] / max(height - 1, 1) - 1.0
        displacement = xy[:, 1:] - xy[:, :-1]
        common = displacement.median(dim=2, keepdim=True).values
        exclusive = displacement - common
        valid_pair = visibility[:, 1:] & visibility[:, :-1]
        valid = valid_pair.to(exclusive.dtype)
        denominator = valid.sum(1).clamp_min(1.0)
        mean_velocity = (exclusive * valid[..., None]).sum(1) / denominator[..., None]
        last_velocity = exclusive[:, -1]
        acceleration = exclusive[:, 1:] - exclusive[:, :-1]
        acceleration_valid = valid[:, 1:] * valid[:, :-1]
        acceleration_denominator = acceleration_valid.sum(1).clamp_min(1.0)
        mean_acceleration = (
            acceleration * acceleration_valid[..., None]
        ).sum(1) / acceleration_denominator[..., None]
        speed = exclusive.square().sum(-1).sqrt()
        mean_speed = (speed * valid).sum(1) / denominator
        last_speed = speed[:, -1]
        path_length = (speed * valid).sum(1)
        final_xy = xy[:, -1]
        radial = (mean_velocity * final_xy).sum(-1)
        visible_fraction = visibility.float().mean(1)
        confidence = visible_fraction * torch.exp(
            -4.0 * (exclusive - mean_velocity[:, None]).abs().mean((1, 3))
        )
        features = torch.cat(
            (
                final_xy,
                mean_velocity,
                last_velocity,
                mean_acceleration,
                mean_speed[..., None],
                last_speed[..., None],
                radial[..., None],
                visible_fraction[..., None],
                confidence[..., None],
                path_length[..., None],
            ),
            dim=-1,
        )
        all_features.append(features.cpu())
        all_tracks.append(xy.cpu())
        all_visibility.append(visibility.cpu())
        all_names.extend(batch["file_name"])

        if batch_index == 1 or batch_index % 25 == 0 or batch_index == total_batches:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "batch": batch_index,
                        "total_batches": total_batches,
                        "samples": len(all_names),
                        "elapsed_seconds": elapsed,
                        "samples_per_second": len(all_names) / max(elapsed, 1e-6),
                    }
                ),
                flush=True,
            )

    expected_names = json.loads(
        (EPOCH / "file_names_test.json").read_text(encoding="utf-8")
    )
    if all_names != expected_names:
        raise RuntimeError("CoTracker diagnostic order does not match formal test artifacts")

    features_all = torch.cat(all_features)
    tracks_all = torch.cat(all_tracks)
    visibility_all = torch.cat(all_visibility)
    torch.save(features_all, EPOCH / "cotracker_motion_features_test.pt")
    torch.save(tracks_all, EPOCH / "cotracker_tracks_test.pt")
    torch.save(visibility_all, EPOCH / "cotracker_visibility_test.pt")
    print(
        json.dumps(
            {
                "samples": len(all_names),
                "elapsed_seconds": time.perf_counter() - started,
                "feature_shape": list(features_all.shape),
                "visibility_rate": float(visibility_all.float().mean()),
                "output": str(EPOCH / "cotracker_motion_features_test.pt"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
