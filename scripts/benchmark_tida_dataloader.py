from __future__ import annotations

import argparse
import json
import time

from torch.utils.data import DataLoader

from fate_oia.datasets.bdd_oia_video import BDDOIAVideoDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--history-frames", type=int, default=5)
    parser.add_argument("--workers", default="0,1,2,4")
    args = parser.parse_args()
    rows = []
    for workers in (int(value) for value in args.workers.split(",")):
        dataset = BDDOIAVideoDataset(
            args.manifest, "train_core", max_samples=args.samples,
            history_frames=args.history_frames,
        )
        loader = DataLoader(
            dataset, batch_size=8, num_workers=workers, shuffle=False,
            persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None,
        )
        start = time.perf_counter()
        count = 0
        for batch in loader:
            count += int(batch["target_image"].shape[0])
        elapsed = time.perf_counter() - start
        rows.append({"workers": workers, "samples": count, "seconds": elapsed,
                     "samples_per_second": count / elapsed})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
