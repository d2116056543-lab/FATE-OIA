from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def _endpoint_mse(row: dict) -> dict:
    import av

    result = {"file_name": row["file_name"], "clip_path": row["clip_path"]}
    try:
        target = np.asarray(Image.open(row["target_image_path"]).convert("RGB"), dtype=np.float32) / 255.0
        endpoint = None
        with av.open(row["clip_path"]) as container:
            for index, frame in enumerate(container.decode(container.streams.video[0])):
                if index == int(row["target_frame_index"]):
                    endpoint = frame.to_image().convert("RGB")
                    break
        if endpoint is None:
            raise RuntimeError("manifest target_frame_index is not decodable")
        decoded = np.asarray(endpoint.resize((target.shape[1], target.shape[0])), dtype=np.float32) / 255.0
        result["endpoint_mse"] = float(np.square(target - decoded).mean())
        result["error"] = None
    except Exception as error:
        result["endpoint_mse"] = None
        result["error"] = repr(error)
    return result


def _write_corrected_manifest(source: Path, output: Path, invalid_ids: set[str]) -> None:
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["file_name"] in invalid_ids:
            row["history_available"] = False
            row["history_unavailable_reason"] = "endpoint_identity_mismatch"
        rows.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    temporary = output.with_suffix(output.suffix + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    temporary.replace(output)


def _payload(source: Path, rows: list[dict], results: list[dict], max_mse: float, complete: bool) -> dict:
    failures = [row for row in results if row["error"] is not None]
    mismatches = [row for row in results if row["endpoint_mse"] is not None and row["endpoint_mse"] > max_mse]
    return {
        "schema": "coev_endpoint_identity_v1",
        "pass": complete and not failures and not mismatches,
        "complete": complete,
        "manifest": str(source),
        "checked": len(results),
        "manifest_rows": len(rows),
        "max_mse": max_mse,
        "mismatch_count": len(mismatches),
        "decode_failure_count": len(failures),
        "observed_max_mse": max((row["endpoint_mse"] for row in results if row["endpoint_mse"] is not None), default=None),
        "mismatches": sorted(mismatches, key=lambda row: row["endpoint_mse"], reverse=True),
        "decode_failures": failures,
        "results": results,
    }


def _atomic_payload(output: Path, payload: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-mse", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--corrected-manifest")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config")
    args = parser.parse_args()
    source = Path(args.manifest)
    identity = None
    if args.config:
        from fate_oia.utils.coev_preflight import source_identity

        identity = source_identity(args.config)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    available = [row for row in rows if row.get("history_available", True)]
    if args.max_rows is not None:
        available = available[: args.max_rows]
    output = Path(args.output)
    results = []
    if args.resume and output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("manifest") != str(source) or previous.get("max_mse") != args.max_mse:
            raise RuntimeError("resume artifact identity differs from current invocation")
        results = previous.get("results", [])
    completed_ids = {row["file_name"] for row in results}
    pending = [row for row in available if row["file_name"] not in completed_ids]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(_endpoint_mse, pending, chunksize=8):
            results.append(result)
            if len(results) % 200 == 0:
                progress = _payload(source, rows, results, args.max_mse, False)
                progress["audit_identity"] = identity
                _atomic_payload(output, progress)
                print(json.dumps({"checked": len(results), "total": len(available)}), flush=True)
    complete = len(results) == len(available)
    payload = _payload(source, rows, results, args.max_mse, complete)
    payload["audit_identity"] = identity
    failures = payload["decode_failures"]
    mismatches = payload["mismatches"]
    invalid_ids = {row["file_name"] for row in failures + mismatches}
    _atomic_payload(output, payload)
    if args.corrected_manifest:
        _write_corrected_manifest(source, Path(args.corrected_manifest), invalid_ids)
    if invalid_ids:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
