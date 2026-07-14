"""Build a fail-closed MOSAIC-TRUST ablation table from completed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence


_TRANSFER_FIELDS = (
    "selected_effect",
    "matched_random_effect",
    "signed_effect",
    "tet",
    "tes",
    "cca",
    "ap_delta",
    "admitted",
)
_REQUIRED_FILES = (
    "branch_metrics.json",
    "per_label_metrics.json",
    "target_transfer_summary.json",
    "target_transfer_stats.jsonl",
)


def _epoch_dir(run_dir: Path, epoch: int | None) -> Path:
    if epoch is not None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        directory = run_dir / f"epoch_{epoch:03d}"
    else:
        directories = sorted(path for path in run_dir.glob("epoch_*") if path.is_dir())
        if not directories:
            raise FileNotFoundError("no MOSAIC-TRUST epoch artifacts found")
        directory = directories[-1]
    if not directory.is_dir():
        raise FileNotFoundError(f"MOSAIC-TRUST epoch artifact directory is missing: {directory}")
    return directory


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"required MOSAIC-TRUST artifact is missing: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path.name}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required MOSAIC-TRUST artifact is missing: {path.name}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"target_transfer_stats.jsonl has an empty row at line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid target_transfer_stats.jsonl row {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError("target_transfer_stats.jsonl rows must be objects")
        rows.append(row)
    if not rows:
        raise ValueError("target_transfer_stats.jsonl must contain real transfer rows")
    return rows


def _reject_unavailable(value: Any, location: str) -> None:
    if value is None:
        raise ValueError(f"artifact value is unavailable at {location}")
    if isinstance(value, str) and value.strip().lower() in {"unavailable", "placeholder", "unknown", "nan", "n/a"}:
        raise ValueError(f"artifact value is unavailable at {location}")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"artifact value is unavailable at {location}")
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"artifact mapping is empty at {location}")
        for key, child in value.items():
            _reject_unavailable(child, f"{location}.{key}")
    elif isinstance(value, list):
        if not value:
            raise ValueError(f"artifact list is empty at {location}")
        for index, child in enumerate(value):
            _reject_unavailable(child, f"{location}[{index}]")


def _flatten_numbers(value: Any, prefix: str = "") -> dict[str, float | bool]:
    if isinstance(value, Mapping):
        result: dict[str, float | bool] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_numbers(child, child_prefix))
        return result
    if isinstance(value, bool):
        return {prefix: value}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not isfinite(float(value)):
            raise ValueError(f"artifact numeric value is unavailable at {prefix}")
        return {prefix: float(value)}
    return {}


def _transfer_pair(row: Mapping[str, Any]) -> tuple[str, str, str]:
    try:
        pair = (str(row["factor_id"]), str(row["target_id"]), str(row["direction"]))
    except KeyError as error:
        raise ValueError("transfer artifact rows require factor_id, target_id, and direction") from error
    if not all(pair) or pair[2] not in {"support", "veto"}:
        raise ValueError("transfer artifact row has an invalid factor/target/direction")
    return pair


def _validate_transfer_row(row: Mapping[str, Any]) -> None:
    _transfer_pair(row)
    missing = [field for field in _TRANSFER_FIELDS if field not in row]
    if missing:
        raise ValueError(f"transfer artifact row is missing fields: {', '.join(missing)}")
    for field in _TRANSFER_FIELDS:
        if field == "admitted":
            if not isinstance(row[field], bool):
                raise ValueError("transfer artifact admitted must be a boolean")
        elif not isinstance(row[field], (int, float)) or isinstance(row[field], bool) or not isfinite(float(row[field])):
            raise ValueError(f"transfer artifact {field} must be a finite number")


def _validate_transfer_artifacts(summary: Any, stats: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(summary, Mapping) or summary.get("schema_version") != "mosaic_target_transfer.v1":
        raise ValueError("target_transfer_summary.json must use mosaic_target_transfer.v1")
    per_target = summary.get("per_target")
    aggregate = summary.get("summary")
    if not isinstance(per_target, list) or not per_target or not isinstance(aggregate, Mapping):
        raise ValueError("target transfer summary requires non-empty per_target and summary payloads")
    summary_rows = [row for row in per_target if isinstance(row, Mapping)]
    if len(summary_rows) != len(per_target):
        raise ValueError("target transfer per_target entries must be objects")
    for row in [*summary_rows, *stats]:
        _validate_transfer_row(row)
    summary_pairs = {_transfer_pair(row): row for row in summary_rows}
    stats_pairs = {_transfer_pair(row): row for row in stats}
    if len(summary_pairs) != len(summary_rows) or len(stats_pairs) != len(stats):
        raise ValueError("transfer artifacts contain duplicate factor-target rows")
    if set(summary_pairs) != set(stats_pairs):
        raise ValueError("target transfer summary and stats rows do not describe the same factor-target pairs")
    for pair, summary_row in summary_pairs.items():
        stats_row = stats_pairs[pair]
        for field in _TRANSFER_FIELDS:
            if field == "admitted":
                if summary_row[field] != stats_row[field]:
                    raise ValueError(f"transfer artifact disagreement for {pair}: {field}")
            elif not isclose(float(summary_row[field]), float(stats_row[field]), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"transfer artifact disagreement for {pair}: {field}")
    if aggregate.get("pair_count") != len(summary_rows):
        raise ValueError("target transfer summary pair_count does not match real rows")
    mean_tes = sum(float(row["tes"]) for row in summary_rows) / len(summary_rows)
    admitted_rate = sum(1.0 if row["admitted"] else 0.0 for row in summary_rows) / len(summary_rows)
    for field, expected in (("mean_tes", mean_tes), ("admitted_rate", admitted_rate)):
        if field not in aggregate or not isinstance(aggregate[field], (int, float)) or not isfinite(float(aggregate[field])):
            raise ValueError(f"target transfer summary requires finite {field}")
        if not isclose(float(aggregate[field]), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"target transfer summary {field} does not match real rows")
    return aggregate


def build_row(run_dir: str | Path, *, epoch: int | None = None) -> dict[str, Any]:
    """Return one flat table row from complete, non-placeholder epoch artifacts."""

    root = Path(run_dir)
    directory = _epoch_dir(root, epoch)
    branch_metrics = _read_json(directory / _REQUIRED_FILES[0])
    per_label_metrics = _read_json(directory / _REQUIRED_FILES[1])
    transfer_summary = _read_json(directory / _REQUIRED_FILES[2])
    transfer_stats = _read_jsonl(directory / _REQUIRED_FILES[3])
    for name, payload in (("branch_metrics", branch_metrics), ("per_label_metrics", per_label_metrics), ("target_transfer_summary", transfer_summary)):
        _reject_unavailable(payload, name)
    if not isinstance(branch_metrics, Mapping) or not isinstance(per_label_metrics, Mapping):
        raise ValueError("branch_metrics and per_label_metrics artifacts must be objects")
    branch_values = _flatten_numbers(branch_metrics, "branch")
    label_values = _flatten_numbers(per_label_metrics, "per_label")
    if not branch_values or not label_values:
        raise ValueError("branch_metrics and per_label_metrics must contain real numeric metrics")
    aggregate = _validate_transfer_artifacts(transfer_summary, transfer_stats)
    try:
        epoch_value = int(directory.name.split("_", maxsplit=1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid MOSAIC-TRUST epoch directory name: {directory.name}") from error
    row: dict[str, Any] = {"run": root.name, "epoch": epoch_value}
    row.update(branch_values)
    row.update(label_values)
    for field, value in aggregate.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            row[f"transfer.{field}"] = float(value)
    row["transfer.stats_row_count"] = len(transfer_stats)
    return row


def write_table(rows: Sequence[Mapping[str, Any]], *, output_json: str | Path, output_csv: str | Path) -> None:
    if not rows:
        raise ValueError("ablation table requires at least one real run row")
    normalized = [dict(row) for row in rows]
    fields = ["run", "epoch"] + sorted({key for row in normalized for key in row} - {"run", "epoch"})
    json_path, csv_path = Path(output_json), Path(output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", "--run_dir", action="append", required=True, dest="run_dirs")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--output-json", "--output_json", required=True, dest="output_json")
    parser.add_argument("--output-csv", "--output_csv", required=True, dest="output_csv")
    args = parser.parse_args()
    write_table(
        [build_row(run_dir, epoch=args.epoch) for run_dir in args.run_dirs],
        output_json=args.output_json,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
