from __future__ import annotations

import json
from pathlib import Path

import torch


def main() -> None:
    partial = Path(r"F:\FATE_Drive_runs\tida_object_tracks_1000_calib512_test512.pt.partial")
    epoch = Path(r"F:\FATE_Drive_runs\tida_relational_v8_2_pilot5584x1_retry2\epoch_000")
    output = Path(r"F:\FATE_Drive_runs\tida_object_tracks_1000_calib324_test885.pt")
    manifest_path = Path(r"artifacts\tida_10k_v8\tida_10k_primary_manifest.jsonl")
    partial_payload = torch.load(partial, map_location="cpu", weights_only=True)
    test_names = json.loads((epoch / "file_names_test.json").read_text(encoding="utf-8"))
    test_xy = torch.load(epoch / "cotracker_tracks_test.pt", map_location="cpu", weights_only=True).half()
    test_visibility = torch.load(
        epoch / "cotracker_visibility_test.pt", map_location="cpu", weights_only=True
    ).bool()
    manifest = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        manifest[row["file_name"].lower()] = row["partition"]
    counts: dict[str, int] = {}
    for name in partial_payload["file_names"]:
        partition = manifest[name.lower()]
        counts[partition] = counts.get(partition, 0) + 1
    if counts != {"train_core": 1000, "train_calib": 324}:
        raise RuntimeError(f"unexpected partial track accounting: {counts}")
    if not len(test_names) == len(test_xy) == len(test_visibility) == 885:
        raise RuntimeError("existing CoTracker test artifacts are incomplete")
    if not all(manifest[name.lower()] == "test" for name in test_names):
        raise RuntimeError("existing CoTracker test names do not match the formal test split")
    if set(map(str.lower, partial_payload["file_names"])).intersection(map(str.lower, test_names)):
        raise RuntimeError("pilot track store has cross-partition duplicate names")
    payload = {
        "schema": "tida_frozen_cotracker_coordinates_v1",
        "manifest": str(manifest_path.resolve()),
        "file_names": list(partial_payload["file_names"]) + test_names,
        "tracks_xy": torch.cat((partial_payload["tracks_xy"], test_xy)),
        "visibility": torch.cat((partial_payload["visibility"], test_visibility)),
    }
    torch.save(payload, output)
    print(json.dumps({
        "output": str(output), "counts": {**counts, "test": 885},
        "samples": len(payload["file_names"]), "shape": list(payload["tracks_xy"].shape),
        "visibility_rate": float(payload["visibility"].float().mean()),
    }))


if __name__ == "__main__":
    main()
