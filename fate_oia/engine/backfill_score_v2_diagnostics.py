from __future__ import annotations

import argparse
import json
from pathlib import Path

from fate_oia.engine.score_v2_diagnostics import write_score_v2_epoch_diagnostics


def _read_manifest_n_last_blocks(run_dir: Path) -> int | None:
    manifest = run_dir / "run_manifest.json"
    if not manifest.exists():
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    value = payload.get("n_last_blocks")
    return int(value) if value is not None else None


def backfill_score_v2_run(run_dir: str | Path, *, split: str = "test", tail_reason_indices: list[int] | None = None) -> dict:
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    tail = tail_reason_indices or [12, 9, 5, 14, 6, 11, 10, 13]
    n_last_blocks = _read_manifest_n_last_blocks(root)
    rows = []
    for epoch_dir in sorted(root.glob("epoch_*")):
        if not (epoch_dir / f"logits_reason_{split}.pt").exists():
            continue
        result = write_score_v2_epoch_diagnostics(
            epoch_dir,
            run_dir=root,
            split=split,
            tail_reason_indices=tail,
            n_last_blocks=n_last_blocks,
        )
        rows.append(
            {
                "epoch_dir": str(epoch_dir),
                "tail_macro_f1": result["tail_group_metrics"]["tail_macro_f1"],
                "tail_macro_ap": result["tail_group_metrics"]["tail_macro_ap"],
                "tail_positive_support": result["tail_group_metrics"]["tail_positive_support"],
            }
        )
    summary = {"run_dir": str(root), "split": split, "epoch_count": len(rows), "rows": rows}
    (root / "score_v2_diagnostics_backfill_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill ScoreV2 required per-epoch diagnostic artifacts from cached logits.")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tail_reason_indices", nargs="+", type=int, default=[12, 9, 5, 14, 6, 11, 10, 13])
    args = ap.parse_args()
    print(json.dumps(backfill_score_v2_run(args.run_dir, split=args.split, tail_reason_indices=args.tail_reason_indices), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
