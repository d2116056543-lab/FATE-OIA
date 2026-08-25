from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from fate_oia.datasets.bdd_oia_video import BDDOIAVideoDataset, tida_video_collate
from fate_oia.models.tida_object_tracker import TIDAFrozenPointTracker


def select_seed_tracks(
    payload: dict[str, object], requested_names: set[str]
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    names = [str(value) for value in payload["file_names"]]
    normalized = {value.lower() for value in requested_names}
    indices = [index for index, name in enumerate(names) if name.lower() in normalized]
    tracks = payload["tracks_xy"]
    visibility = payload["visibility"]
    if not torch.is_tensor(tracks) or not torch.is_tensor(visibility):
        raise TypeError("seed track payload tensors are invalid")
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return (
        [names[index] for index in indices],
        tracks.index_select(0, index_tensor),
        visibility.index_select(0, index_tensor),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--partitions", default="train_core,train_calib,test")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-new-samples", type=int)
    parser.add_argument("--seed-store")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    tracker = TIDAFrozenPointTracker.from_local_repository(args.repository).to(device).eval()
    output_path = Path(args.output)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    names: list[str] = []
    tracks: list[torch.Tensor] = []
    visibility: list[torch.Tensor] = []
    if partial_path.exists():
        partial = torch.load(partial_path, map_location="cpu", weights_only=True)
        names = list(partial["file_names"])
        tracks = [partial["tracks_xy"]]
        visibility = [partial["visibility"]]
    elif args.seed_store:
        requested_names: set[str] = set()
        for partition in tuple(
            value.strip() for value in args.partitions.split(",") if value.strip()
        ):
            limit = args.train_limit if partition == "train_core" else args.eval_limit
            requested = BDDOIAVideoDataset(
                args.manifest, partition, training=False, max_samples=limit
            )
            requested_names.update(row.file_name for row in requested.records)
        seed_payload = torch.load(args.seed_store, map_location="cpu", weights_only=True)
        names, seed_tracks, seed_visibility = select_seed_tracks(
            seed_payload, requested_names
        )
        if names:
            tracks = [seed_tracks]
            visibility = [seed_visibility]
        print(json.dumps({
            "seed_store": str(Path(args.seed_store).resolve()),
            "requested": len(requested_names),
            "reused": len(names),
            "missing": len(requested_names) - len(names),
        }), flush=True)
    completed = {name.lower() for name in names}
    new_samples = 0
    started = time.perf_counter()
    partitions = tuple(value.strip() for value in args.partitions.split(",") if value.strip())

    def save(path: Path) -> None:
        payload = {
            "schema": "tida_frozen_cotracker_coordinates_v1",
            "manifest": str(Path(args.manifest).resolve()),
            "file_names": names,
            "tracks_xy": torch.cat(tracks),
            "visibility": torch.cat(visibility),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    for partition in partitions:
        limit = args.train_limit if partition == "train_core" else args.eval_limit
        dataset = BDDOIAVideoDataset(args.manifest, partition, training=False, max_samples=limit)
        dataset.records = [
            record for record in dataset.records if record.file_name.lower() not in completed
        ]
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
            pin_memory=True, collate_fn=tida_video_collate,
        )
        for batch_index, batch in enumerate(loader, start=1):
            context = batch["context_images"].to(device, non_blocking=True)
            target = batch["target_image"].to(device, non_blocking=True)
            target = torch.nn.functional.interpolate(
                target, size=context.shape[-2:], mode="bilinear", align_corners=False
            )
            output = tracker(torch.cat((context, target[:, None]), dim=1))
            names.extend(batch["file_name"])
            completed.update(name.lower() for name in batch["file_name"])
            tracks.append(output["object_tracks_xy"].half().cpu())
            visibility.append(output["object_tracks_visibility"].cpu())
            new_samples += len(batch["file_name"])
            if len(names) % int(args.save_every) == 0:
                save(partial_path)
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(loader):
                elapsed = time.perf_counter() - started
                print(json.dumps({
                    "partition": partition, "batch": batch_index, "partition_total": len(loader),
                    "samples": len(names), "samples_per_second": len(names) / max(elapsed, 1e-6),
                }), flush=True)
            if args.max_new_samples is not None and new_samples >= args.max_new_samples:
                save(partial_path)
                print(json.dumps({"partial": True, "saved": len(names)}), flush=True)
                return
    save(output_path)
    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    partial_path.unlink(missing_ok=True)
    print(json.dumps({
        "output": str(output_path), "samples": len(names),
        "visibility_rate": float(payload["visibility"].float().mean()),
        "elapsed_seconds": time.perf_counter() - started,
    }), flush=True)


if __name__ == "__main__":
    main()
