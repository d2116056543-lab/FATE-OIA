from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
from torch.utils.data import Subset

from fate_oia.engine.train_aie_oia import canonical_model_state_dict, make_dataset
from fate_oia.engine.train_pact_oia_probe import build_model, load_config, make_loader
from fate_oia.models.aie_oia_model import AIEOIAModel
from fate_oia.utils.aie_calibration import apply_posthoc_threshold
from fate_oia.utils.aie_metrics import aie_branch_metrics
from fate_oia.utils.pact_artifacts import sha256, write_json


REQUIRED = (
    "configs/fate_oia_train_360x640_pact_oia_v1_probe.yaml",
    "fate_oia/models/pact_shared_readout.py", "fate_oia/models/pact_context_decoder.py",
    "fate_oia/models/pact_explanation_decoder.py", "fate_oia/models/pact_predicate_agreement.py",
    "fate_oia/models/pact_reason_rereader.py", "fate_oia/models/pact_oia_model.py",
    "fate_oia/losses/pact_rank_losses.py", "fate_oia/losses/pact_loss_registry.py",
    "fate_oia/utils/pact_pareto_controller.py", "fate_oia/utils/pact_pair_queue.py",
    "fate_oia/utils/pact_bootstrap.py", "fate_oia/utils/pact_artifacts.py",
    "fate_oia/engine/train_pact_oia_probe.py", "fate_oia/engine/evaluate_pact_oia_probe.py",
    "fate_oia/engine/diagnose_pact_owner_tomography.py", "fate_oia/engine/supervise_pact_oia_probe.py",
    "fate_oia/engine/diagnose_pact_scale_conflict.py", "fate_oia/engine/profile_pact_oia.py",
    "tests/test_pact_source_migration.py", "tests/test_pact_role_firewall.py",
    "tests/test_pact_pareto_controller.py", "tests/test_pact_predicate_agreement.py",
    "tests/test_pact_rank_preservation.py", "tests/test_pact_reason_bound.py",
    "tests/test_pact_pair_coverage.py", "tests/test_pact_resume_state.py",
)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--full-replay", action="store_true")
    args = parser.parse_args(); cfg = load_config(args.config); root = Path.cwd(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    forbidden = {"feature_cache": cfg["experiment"]["feature_cache_enabled"],
                 "token_compression": cfg["experiment"]["token_compression"] != "none",
                 "val_best": cfg["experiment"]["best_selection_split"] != "test"}
    device = torch.device(args.device); checkpoint = torch.load(args.source_checkpoint, map_location="cpu")
    source = AIEOIAModel(
        dim=cfg["primary"]["dim"], selected_layers=tuple(cfg["backbone"]["selected_layers"]),
        pretrained_weights=cfg["backbone"]["pretrained_weights"], scene_config=cfg["primary"]["scene_predicates"],
        grammar_path=cfg["primary"]["reason_grammar"], probes_per_action=cfg["evidence"]["probes_per_action"],
        local_points_per_layer=cfg["evidence"]["local_points_per_layer"], max_offset=cfg["evidence"]["max_offset"],
        predicate_bias_max=cfg["evidence"]["predicate_bias_max"], probe_chunk_size=cfg["evidence"]["probe_chunk_size"],
        action_kappa=cfg["evidence"]["action_kappa"], reason_kappa=cfg["reason"]["kappa"],
    ).to(device).eval()
    source.load_state_dict(canonical_model_state_dict(checkpoint["model"]), strict=True)
    pact = build_model(cfg, device).eval(); migration = pact.migrate_from_aie_state_dict(checkpoint["model"])
    test = make_dataset(cfg, "test"); limit = len(test) if args.full_replay else min(2, len(test))
    loader = make_loader(Subset(test, list(range(limit))), 6, False, 8 if args.full_replay else 0, cfg)
    stores = {key: [] for key in ("action", "reason", "action_target", "reason_target")}; max_error = {"action": 0.0, "reason": 0.0}
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.full_replay and device.type == "cuda"):
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device); field = source.encode_images(images)
            expected = source.decode_from_field(field, action_scale=1.0, reason_scale=0.60)
            actual = pact.decode_from_field(field, semantic_share_license=1.0, action_scale=1.0,
                                            reason_budget=0.60, compatibility_mode=True)
            max_error["action"] = max(max_error["action"], float((expected["action_logits_final"] - actual["action_logits_final"]).abs().max()))
            max_error["reason"] = max(max_error["reason"], float((expected["reason_logits_final"] - actual["reason_logits_final"]).abs().max()))
            stores["action"].append(expected["action_logits_final"].cpu()); stores["reason"].append(expected["reason_logits_final"].cpu())
            stores["action_target"].append(batch["action"]); stores["reason_target"].append(batch["reason"])
            if (batch_index + 1) % 100 == 0:
                write_json(output / "source_replay_progress.json", {
                    "batches_complete": batch_index + 1, "samples_complete": min((batch_index + 1) * 6, limit),
                    "max_abs_action": max_error["action"], "max_abs_reason": max_error["reason"],
                })
    stores = {key: torch.cat(value) for key, value in stores.items()}
    threshold = checkpoint["calibration"]["threshold_prob"]
    deploy = aie_branch_metrics(apply_posthoc_threshold(stores["action"], threshold[:4]),
                                apply_posthoc_threshold(stores["reason"], threshold[4:]),
                                stores["action_target"], stores["reason_target"])
    source_replay = {"samples": limit, "full_replay": args.full_replay, "deploy": deploy,
                     "checkpoint_metrics": checkpoint.get("metrics", {})}
    tolerance = 5e-4 if args.full_replay and device.type == "cuda" else 1e-5
    equivalence = {"migration": migration, "max_abs_action": max_error["action"], "max_abs_reason": max_error["reason"],
                   "precision": "bf16" if tolerance == 5e-4 else "float32", "tolerance": tolerance,
                   "pass": max(max_error.values()) <= tolerance}
    write_json(output / "source_replay.json", source_replay); write_json(output / "migration_equivalence.json", equivalence)
    smoke = root / ".review/pact_oia_v1_probe/counterfactual_smoke/checkpoint_latest.pth"
    smoke_summary = root / ".review/pact_oia_v1_probe/counterfactual_smoke/epoch_000/counterfactual_summary.json"
    result = {"pass": not missing and not any(forbidden.values()) and equivalence["pass"] and smoke.exists(),
              "git_head": _git_head(), "config_hash": sha256(args.config), "checkpoint_hash": sha256(args.source_checkpoint),
              "checked_files": list(REQUIRED), "missing_items": missing, "forbidden": forbidden,
              "source_replay": source_replay, "migration_equivalence": equivalence,
              "real_smoke_checkpoint": str(smoke), "counterfactual_summary": str(smoke_summary),
              "warnings": [] if smoke_summary.exists() else ["counterfactual smoke summary is missing"]}
    if not smoke_summary.exists():
        result["pass"] = False
    write_json(output / "implementation_audit.json", result)
    if not result["pass"]:
        raise SystemExit(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
