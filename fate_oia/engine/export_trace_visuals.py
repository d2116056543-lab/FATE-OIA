from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw
from fate_oia.utils.trace_artifacts import append_jsonl, write_json


def export_trace_case(output_dir: str | Path, epoch: int, case: dict[str, Any]) -> dict[str, Any]:
    root = Path(output_dir) / "visuals" / f"epoch_{epoch:03d}"; root.mkdir(parents=True, exist_ok=True)
    sid = str(case.get("sample_id", "sample"))
    png, js = root / f"{sid}.png", root / f"{sid}.json"
    img = Image.new("RGB", (320, 180), (18, 18, 18)); dr = ImageDraw.Draw(img)
    dr.text((8, 8), f"TRACE {sid}", fill=(255, 255, 255)); dr.text((8, 32), f"reason={case.get('reason_idx', 0)} proto={case.get('prototype_id', 0)}", fill=(180, 220, 255)); dr.text((8, 56), f"drop={case.get('drop', 0.0):.4f}", fill=(255, 220, 120)); img.save(png)
    row = {"sample_id": sid, "file_name": case.get("file_name", ""), "reason_idx": int(case.get("reason_idx", 0)), "prototype_id": int(case.get("prototype_id", 0)), "top_evidence": case.get("top_evidence", []), "transport_mass": float(case.get("transport_mass", 0.0)), "factual_logit": float(case.get("factual_logit", 0.0)), "target_deleted_logit": float(case.get("target_deleted_logit", 0.0)), "drop": float(case.get("drop", 0.0)), "png": str(png), "json": str(js)}
    write_json(js, row); append_jsonl(Path(output_dir) / f"epoch_{epoch:03d}" / "trace_visuals_index.jsonl", row)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--output_dir", required=True); ap.add_argument("--epoch", type=int, default=0); args = ap.parse_args(argv)
    export_trace_case(args.output_dir, args.epoch, {"sample_id": "smoke", "reason_idx": 0, "prototype_id": 0, "drop": 0.01, "top_evidence": [{"source_type": "object", "transport_mass": 1.0}]})


if __name__ == "__main__":
    main()
