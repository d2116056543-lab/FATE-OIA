from __future__ import annotations

import argparse
import json
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.utils.meter_artifacts import combined_file_hash, python_source_tree_hash, write_json
from fate_oia.utils.meter_config import load_meter_config


REQUIRED_FILES = (
    "configs/fate_oia_train_360x640_acpr_meter_oia_v3_heca.yaml",
    "configs/meter_factor_schema.yaml",
    "fate_oia/models/meter_meta_adapters.py",
    "fate_oia/models/meter_signed_factors.py",
    "fate_oia/models/meter_semantic_action.py",
    "fate_oia/models/meter_reason_decoder.py",
    "fate_oia/models/meter_oia_model.py",
    "fate_oia/losses/meter_action_losses.py",
    "fate_oia/losses/meter_reason_losses.py",
    "fate_oia/losses/meter_grounding_losses.py",
    "fate_oia/optim/heca_optimization.py",
    "fate_oia/engine/export_heca_ontology_prototypes.py",
    "fate_oia/engine/prepare_heca_static_artifacts.py",
    "fate_oia/engine/train_acpr_meter_oia.py",
    "fate_oia/engine/eval_acpr_meter_oia.py",
    "fate_oia/engine/evaluate_meter_oia_v3_heca_pilot.py",
    "fate_oia/engine/supervise_meter_oia_v3_heca_foreground.py",
    "scripts/FATE_OIA_meter_oia_v3_heca_pilot.ps1",
    "scripts/FATE_OIA_meter_oia_v3_heca_foreground.ps1",
)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _source_checks(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    active_runtime_names = (
        "fate_oia/models/meter_signed_factors.py",
        "fate_oia/models/meter_semantic_action.py",
        "fate_oia/models/meter_reason_decoder.py",
        "fate_oia/models/meter_oia_model.py",
        "fate_oia/engine/train_acpr_meter_oia.py",
        "fate_oia/engine/eval_acpr_meter_oia.py",
        "fate_oia/engine/evaluate_meter_oia_v3_heca_pilot.py",
    )
    active = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in active_runtime_names
        if (root / name).exists()
    )
    exporter = (root / "fate_oia/engine/export_heca_ontology_prototypes.py").read_text(
        encoding="utf-8"
    )
    factor_head = (root / "fate_oia/models/meter_signed_factors.py").read_text(
        encoding="utf-8"
    )
    grounding_loss = (root / "fate_oia/losses/meter_grounding_losses.py").read_text(
        encoding="utf-8"
    )
    pilot_script = (root / "scripts/FATE_OIA_meter_oia_v3_heca_pilot.ps1").read_text(
        encoding="utf-8"
    )
    offline_encoder_path = str(config["artifacts"]["offline_text_encoder_path"])
    offline_encoder_script_path = offline_encoder_path.replace("/", "\\")
    forbidden_tokens = {
        "hard_action_factor_field": "compatible" + "_" + "actions",
        "admission_parameter": "action_evidence_" + "admission",
        "legacy_dense_factor_loss": "dense_factor_" + "loss",
        "legacy_anti_monopoly": "anti_" + "monopoly",
        "legacy_action_transport_alias": "FactorSpecific" + "ActionTransport",
        "legacy_signed_factor_alias": "METER" + "signedFactors",
        "runtime_text_encoder": "Auto" + "Tokenizer",
        "run_c": "run_c_" + "logits",
        "cache": "cached_" + "logits",
    }
    forbidden = {name: token in active for name, token in forbidden_tokens.items()}
    protocol = {
        "epochs_14": int(config["training"]["epochs"]) == 14,
        "memory_safe_effective_batch_4x8": (
            int(config["training"]["batch_size"]) == 4
            and int(config["training"]["gradient_accumulation_steps"]) == 8
            and int(config["training"]["effective_batch_size"]) == 32
        ),
        "single_backward_next_window_balance": (
            config["training"].get("shared_gradient_policy") == "next_window_single_backward"
            and "shared_action_grads" not in active
            and "shared_reason_grads" not in active
            and "shared_action_gradient_scale=active_balance[\"action\"]" in active
            and "shared_reason_gradient_scale=active_balance[\"reason\"]" in active
        ),
        "bf16": str(config["training"]["precision"]).lower() == "bf16",
        "test_only": config["runtime"]["test_only"] is True,
        "no_cache": config["model"]["feature_cache_enabled"] is False,
        "no_compression": config["model"]["token_compression"] == "none",
        "same_forward_eval": config["runtime"]["sequential_eval"] is False,
        "provenance_is_not_learned_visual_target": (
            "self.obs_head" not in factor_head
            and "obs_head =" not in factor_head
            and "obs_bce, obs_coverage = observability_objective(" not in grounding_loss
            and "factor_provenance_valid" not in (root / "fate_oia/models/meter_oia_model.py").read_text(encoding="utf-8")
        ),
        "provenance_stats_are_train_only": (
            str(config["model"].get("provenance_stats_path", "")).endswith(
                "factor_provenance_stats.json"
            )
            and "observability_tau_path" not in config["model"]
        ),
        "offline_ontology_export": (
            "local_files_only=True" in exporter
            and "all-MiniLM-L6-v2" not in exporter
            and "[string]$TextEncoderPath" not in pilot_script
            and f'$TextEncoderPath = "{offline_encoder_script_path}"' in pilot_script
            and '"--encoder_id", $TextEncoderPath' in pilot_script
            and offline_encoder_path.endswith("frozen_bert_base_uncased")
        ),
    }
    return {
        "missing_files": missing,
        "forbidden_pattern_results": forbidden,
        "protocol": protocol,
        "pass": not missing and not any(forbidden.values()) and all(protocol.values()),
    }


def _dynamic_checks() -> dict[str, Any]:
    torch.manual_seed(7)
    model = METEROIAModel(dim=384, use_mock_dino=True).eval()
    image = torch.randn(2, 3, 360, 640)
    with torch.no_grad():
        field = model.encode_images(image)
        clean = model.decode_from_field(field, progress=0.0)
        factor_off = model.decode_from_field(field, progress=0.0, diagnostic_modes=("factor_off",))
        uniform = model.decode_from_field(field, progress=0.5, diagnostic_modes=("state_uniform",))
    shapes = {
        key: list(clean[key].shape)
        for key in ("action_logits_final", "reason_logits_final", "factor_anchor_map", "factor_state_prob", "factor_visual_confidence", "action_factor_contribution")
    }
    checks = {
        "action_progress_zero_equivalence": float((clean["action_logits_final"] - clean["action_logits_visual"]).abs().max()) < 1e-6,
        "reason_progress_zero_equivalence": float((clean["reason_logits_final"] - clean["reason_logits_global"]).abs().max()) < 1e-6,
        "label_nodes_progress_zero_equivalence": float(
            (clean["label_nodes"] - clean["label_nodes_base"]).abs().max()
        ) < 1e-6,
        "factor_off_action_visual": torch.equal(factor_off["action_logits_final"], factor_off["action_logits_visual"]),
        "state_uniform_recomputes_values": not torch.allclose(clean["action_factor_values"], uniform["action_factor_values"]),
        "visual_confidence_matches_reliability": torch.allclose(
            clean["factor_visual_confidence"], clean["factor_reliability"]
        ),
        "visual_confidence_is_finite": bool(torch.isfinite(clean["factor_visual_confidence"]).all()),
        "one_dino_call": model._encode_call_count == 1,
        "shapes": shapes == {
            "action_logits_final": [2, 4], "reason_logits_final": [2, 21],
            "factor_anchor_map": [2, 21, 3600], "factor_state_prob": [2, 21, 3],
            "factor_visual_confidence": [2, 21],
            "action_factor_contribution": [2, 4, 21],
        },
    }
    return {"checks": checks, "shapes": shapes, "pass": all(checks.values())}


def run_audit(config_path: str, output_dir: str, write_review_pass: bool) -> dict[str, Any]:
    root = Path.cwd()
    config = load_meter_config(config_path)
    compile_errors: list[str] = []
    for name in REQUIRED_FILES:
        path = root / name
        if path.suffix == ".py" and path.exists():
            try:
                py_compile.compile(str(path), doraise=True)
            except Exception as error:
                compile_errors.append(f"{name}: {error}")
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *[str(path) for path in sorted((root / "tests").glob("test_heca_*.py"))]],
        text=True, capture_output=True, check=False,
    )
    source = _source_checks(root, config)
    dynamic = _dynamic_checks()
    result = {
        "pass": not compile_errors and tests.returncode == 0 and source["pass"] and dynamic["pass"],
        "git_head": _git_head(),
        "config_hash": combined_file_hash(config_path),
        "source_hash": python_source_tree_hash(root),
        "checked_files": list(REQUIRED_FILES),
        "compile_errors": compile_errors,
        "pytest": {"returncode": tests.returncode, "stdout": tests.stdout[-4000:], "stderr": tests.stderr[-2000:]},
        "source_checks": source,
        "dynamic_checks": dynamic,
        "pilot_gates": {"required": list("ABCDEFG"), "evaluated": False},
        "review_pass_path": "",
        "missing_items": source["missing_files"],
        "warnings": ["Implementation pass does not replace the real 4-epoch pilot gates A-G."],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "implementation_audit_METER_OIA_V3_HECA.json"
    if write_review_pass and result["pass"]:
        review = output / "REVIEW_PASS_METER_OIA_V3_HECA.json"
        result["review_pass_path"] = str(review)
        write_json(review, {"pass": True, "git_head": result["git_head"], "config_hash": result["config_hash"], "source_hash": result["source_hash"], "audit": str(audit_path)})
    write_json(audit_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--write_review_pass", action="store_true")
    args = parser.parse_args()
    result = run_audit(args.config, args.output_dir, args.write_review_pass)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
