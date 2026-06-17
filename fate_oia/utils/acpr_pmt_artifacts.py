from __future__ import annotations

import json
from pathlib import Path


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_pmt_epoch_artifacts(epoch_dir: Path, epoch: int, payload: dict) -> None:
    mapping = {
        "predicate_patch_alignment": "predicate_patch_alignment.jsonl",
        "predicate_coverage": "predicate_coverage.jsonl",
        "predicate_reason_alignment": "predicate_reason_alignment.jsonl",
        "triadic_mediator_stats": "triadic_mediator_stats.jsonl",
        "triadic_chain_topk": "triadic_chain_topk.jsonl",
        "pmt_hardpair_stats": "pmt_hardpair_stats.jsonl",
        "pmt_pu_stats": "pmt_pu_stats.jsonl",
        "threshold_diagnostics": "threshold_diagnostics.jsonl",
        "pmt_phase_schedule": "pmt_phase_schedule.jsonl",
        "loss_group_components": "loss_group_components.jsonl",
    }
    for key, name in mapping.items():
        row = payload.get(key, {})
        if not row:
            row = {"available": False, "reason": "not_computed"}
        row = {"epoch": int(epoch), **row}
        append_jsonl(epoch_dir / name, row)
