from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from fate_oia.models.mosaic_factor_certificate import MOSAICFactorCertificate, build_factor_certificate
from fate_oia.models.mosaic_native_semantics import load_icdor_ontology


def _read_audit_payload(path: str | Path) -> tuple[str, Mapping[str, Mapping[str, Any]]]:
    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"IC-DOR factor audit payload is unreadable: {input_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("IC-DOR factor audit payload must be a mapping")
    source_split = payload.get("source_split")
    factor_stats = payload.get("factor_stats")
    if source_split != "train_audit":
        raise ValueError("IC-DOR factor certificate builder only accepts train_audit payloads")
    if not isinstance(factor_stats, Mapping) or not factor_stats:
        raise ValueError("IC-DOR factor audit payload requires non-empty factor_stats")
    if any(not isinstance(name, str) or not isinstance(stats, Mapping) for name, stats in factor_stats.items()):
        raise ValueError("IC-DOR factor audit statistics must map names to mappings")
    return str(source_split), factor_stats


def build_and_write_factor_certificate(
    audit_stats_path: str | Path,
    output_path: str | Path,
    *,
    config_root: str | Path,
) -> MOSAICFactorCertificate:
    """Build the immutable certificate only from persisted train-audit statistics."""
    source_split, factor_stats = _read_audit_payload(audit_stats_path)
    ontology = load_icdor_ontology(Path(config_root))
    expected_names = {str(factor["name"]) for factor in ontology["factors"]}
    if set(factor_stats) != expected_names:
        missing = sorted(expected_names - set(factor_stats))
        unexpected = sorted(set(factor_stats) - expected_names)
        raise ValueError(f"IC-DOR factor audit names do not match ontology; missing={missing}, unexpected={unexpected}")
    certificate = build_factor_certificate(
        factor_stats,
        ontology["certificate_rules"],
        source_split=source_split,
    )
    certificate.write_json(output_path)
    return certificate


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an immutable IC-DOR factor certificate from train_audit statistics.")
    parser.add_argument("--audit_stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config_root", default="configs")
    args = parser.parse_args()
    certificate = build_and_write_factor_certificate(args.audit_stats, args.output, config_root=args.config_root)
    print(json.dumps({"source_split": certificate.source_split, "sha256": certificate.sha256, "output": str(args.output)}))


if __name__ == "__main__":
    main()
