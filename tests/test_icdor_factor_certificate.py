from __future__ import annotations

import copy
import json
from pathlib import Path

from fate_oia.engine.build_mosaic_factor_certificate import build_and_write_factor_certificate
from fate_oia.models.mosaic_factor_certificate import build_factor_certificate
from fate_oia.models.mosaic_native_semantics import load_icdor_ontology


def _record() -> dict[str, object]:
    return {
        "counts": {
            "confirmed_positive": 40,
            "reliable_negative": 40,
            "weak_negative": 12,
            "unknown": 5,
            "geometry_valid": 220,
        },
        "scores": {
            "full": 0.72,
            "content_only": 0.55,
            "prior_only": 0.40,
            "query_shuffle_drop": 0.04,
            "image_shuffle_drop": 0.05,
            "grounding_minus_random": 0.08,
            "view_consistency": 0.80,
            "mirror_consistency": 0.81,
            "ece": 0.04,
            "presence_variance": 0.03,
            "visibility_variance": 0.02,
        },
        "prototype": {"effective_count": 2.1, "dominant_rate": 0.50, "dead_count": 0},
        "bootstrap_lcb95": {
            "full_minus_prior_only": 0.05,
            "query_shuffle_drop": 0.02,
            "image_shuffle_drop": 0.02,
            "grounding_minus_random": 0.04,
        },
    }


def test_icdor_certificate_is_train_audit_only_and_hash_stable() -> None:
    ontology = load_icdor_ontology(Path("configs"))
    stats = {factor["name"]: _record() for factor in ontology["factors"]}
    first = build_factor_certificate(stats, ontology["certificate_rules"], source_split="train_audit")
    second = build_factor_certificate(stats, ontology["certificate_rules"], source_split="train_audit")

    assert first.sha256 == second.sha256
    assert all(entry["tier"] == "certified" for entry in first.entries.values())
    assert all(entry["reliability"] == 1.0 for entry in first.entries.values())
    assert first.to_dict()["source_split"] == "train_audit"


def test_icdor_certificate_downgrades_without_identifiability_and_rejects_other_splits() -> None:
    ontology = load_icdor_ontology(Path("configs"))
    stable_but_weak = _record()
    stable_but_weak["counts"] = {**stable_but_weak["counts"], "geometry_valid": 5, "reliable_negative": 2}
    no_signal = copy.deepcopy(stable_but_weak)
    no_signal["bootstrap_lcb95"]["query_shuffle_drop"] = 0.0
    stats = {
        ontology["factors"][0]["name"]: stable_but_weak,
        ontology["factors"][1]["name"]: no_signal,
    }
    certificate = build_factor_certificate(stats, ontology["certificate_rules"], source_split="train_audit")
    assert certificate.entries[ontology["factors"][0]["name"]]["tier"] == "reason_only"
    assert certificate.entries[ontology["factors"][0]["name"]]["reliability"] == 0.5
    assert certificate.entries[ontology["factors"][1]["name"]]["tier"] == "abstained"
    assert certificate.entries[ontology["factors"][1]["name"]]["reliability"] == 0.0


def test_certificate_builder_refuses_non_audit_payload_and_persists_a_frozen_digest(tmp_path: Path) -> None:
    ontology = load_icdor_ontology(Path("configs"))
    payload = {
        "source_split": "train_audit",
        "factor_stats": {factor["name"]: _record() for factor in ontology["factors"]},
    }
    input_path = tmp_path / "audit_factor_stats.json"
    output_path = tmp_path / "factor_certificate.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    certificate = build_and_write_factor_certificate(input_path, output_path, config_root=Path("configs"))

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert certificate.sha256 == persisted["sha256"]
    assert persisted["source_split"] == "train_audit"
    payload["source_split"] = "train_core"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        build_and_write_factor_certificate(input_path, output_path, config_root=Path("configs"))
    except ValueError as error:
        assert "train_audit" in str(error)
    else:
        raise AssertionError("certificate builder must reject non-audit inputs")
