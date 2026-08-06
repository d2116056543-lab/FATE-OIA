from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

import torch
import yaml

from fate_oia.losses.aie_loss_registry import exact_owner_parameter_groups
from fate_oia.models.acpr_oia_model import ACPROIAModel
from fate_oia.models.aie_calalign_foundation import AIECalAlignFoundation
from fate_oia.models.aie_oia_model import AIEOIAModel
from fate_oia.engine.train_aie_oia import make_dataset
from fate_oia.utils.aie_artifacts import write_json
from fate_oia.utils.aie_contracts import gradient_norm, scan_forbidden
from fate_oia.utils.aie_hashes import aie_source_tree_sha256, file_sha256, object_sha256
from fate_oia.utils.aie_counterfactual import AIECounterfactualEngine


REQUIRED_FILES = [
    "configs/fate_oia_train_360x640_aie_oia_v1.yaml", "configs/aie_scene_predicates.yaml", "configs/aie_reason_counter_evidence.yaml",
    "fate_oia/models/aie_calalign_foundation.py", "fate_oia/models/aie_evidence_interface.py", "fate_oia/models/aie_deformable_reread.py",
    "fate_oia/models/aie_contribution_head.py", "fate_oia/models/aie_predicate_naming.py", "fate_oia/models/aie_reason_rereader.py", "fate_oia/models/aie_oia_model.py",
    "fate_oia/datasets/aie_structured_evidence.py", "fate_oia/datasets/aie_splits.py", "fate_oia/losses/aie_losses.py", "fate_oia/losses/aie_loss_registry.py",
    "fate_oia/utils/aie_counterfactual.py", "fate_oia/utils/aie_calibration.py", "fate_oia/utils/aie_metrics.py", "fate_oia/utils/aie_artifacts.py", "fate_oia/utils/aie_contracts.py", "fate_oia/utils/aie_hashes.py",
    "fate_oia/engine/train_aie_oia.py", "fate_oia/engine/eval_aie_oia.py", "fate_oia/engine/profile_aie_oia.py", "fate_oia/engine/audit_aie_oia_implementation.py",
    "fate_oia/engine/evaluate_aie_oia_pilot.py", "fate_oia/engine/supervise_aie_oia_foreground.py",
    "scripts/FATE_OIA_aie_oia_v1_pilot.ps1", "scripts/FATE_OIA_aie_oia_v1_foreground.ps1",
    ".codex/skills/aie-oia-v1-implementation-audit/SKILL.md",
]


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _zero(model) -> None:
    for parameter in model.parameters(): parameter.grad = None


def gradient_probe(model: AIEOIAModel, image: torch.Tensor) -> dict:
    owners = exact_owner_parameter_groups(model)
    checks = {}
    losses = {
        "primary_only": lambda out: out["action_logits_primary"].sum() + out["reason_logits_primary"].sum() + out["predicate_logits"].sum(),
        "final_action": lambda out: out["action_logits_final_train"].sum(),
        "final_reason": lambda out: out["reason_logits_final_train"].sum(),
        "predicate_only": lambda out: out["predicate_logits"].sum(),
        "naming_only": lambda out: out["name_quality"].sum(),
    }
    for name, make_loss in losses.items():
        _zero(model); out = model(image); make_loss(out).backward()
        checks[name] = {owner: gradient_norm(parameters) for owner, parameters in owners.items()}
        checks[name]["dino"] = gradient_norm(model.foundation.dino.parameters())
    _zero(model); out = model(image)
    cf = AIECounterfactualEngine().run(
        model, out, torch.ones(image.shape[0], 4, device=image.device),
        [f"audit_{index}.jpg" for index in range(image.shape[0])], global_update=4, action_scale=1.0,
    )
    if cf["cf_valid_count"] > 0:
        (cf["selected_minus_control"].mean() + cf["supportive_contribution"].mean()).backward()
        checks["cf_only"] = {owner: gradient_norm(parameters) for owner, parameters in owners.items()}
        checks["cf_only"]["dino"] = gradient_norm(model.foundation.dino.parameters())
    else:
        checks["cf_only"] = {"invalid": True}
    return checks


def primary_trajectory_probe(model: AIEOIAModel, image: torch.Tensor) -> float:
    primary_only = copy.deepcopy(model).train()
    full = copy.deepcopy(model).train()
    optimizer_primary = torch.optim.AdamW([p for p in primary_only.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.05)
    optimizer_full = torch.optim.AdamW([p for p in full.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.05)
    for _ in range(2):
        optimizer_primary.zero_grad(set_to_none=True); optimizer_full.zero_grad(set_to_none=True)
        out_primary = primary_only(image)
        out_full = full(image)
        primary_loss = out_primary["action_logits_primary"].square().mean() + out_primary["reason_logits_primary"].square().mean() + out_primary["predicate_logits"].square().mean()
        full_primary_loss = out_full["action_logits_primary"].square().mean() + out_full["reason_logits_primary"].square().mean() + out_full["predicate_logits"].square().mean()
        final_loss = out_full["action_logits_final_train"].square().mean() + out_full["reason_logits_final_train"].square().mean()
        primary_loss.backward(); (full_primary_loss + final_loss).backward()
        optimizer_primary.step(); optimizer_full.step()
    names = dict(primary_only.named_parameters())
    full_names = dict(full.named_parameters())
    primary_prefixes = ("foundation.ego.", "foundation.predicate_head.", "foundation.trunk.", "foundation.predicate_reason.")
    return max(float((parameter - full_names[name]).abs().max().detach().cpu()) for name, parameter in names.items() if name.startswith(primary_prefixes))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); root = Path.cwd(); output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    missing = [path for path in REQUIRED_FILES if not (root / path).exists()]
    # The contract module contains the literal deny-list by design; scanning
    # that declaration would be a self-referential false positive.
    forbidden_targets = [
        str(root / path) for path in REQUIRED_FILES
        if (root / path).exists()
        and Path(path).suffix in {".py", ".yaml", ".ps1"}
        and path != "fate_oia/utils/aie_contracts.py"
    ]
    forbidden = scan_forbidden(forbidden_targets)
    device = torch.device(args.device)
    backbone, primary_cfg = cfg["backbone"], cfg["primary"]
    real_image = make_dataset(cfg, "train")[0]["image"].unsqueeze(0).to(device)
    source = ACPROIAModel(
        selected_layers=tuple(backbone["selected_layers"]), pretrained_weights=str(backbone["pretrained_weights"]),
        scene_config=str(primary_cfg["scene_predicates"]), grammar_path=str(primary_cfg["reason_grammar"]), threshold_enabled=False,
    ).to(device).eval()
    foundation = AIECalAlignFoundation(
        selected_layers=tuple(backbone["selected_layers"]), pretrained_weights=str(backbone["pretrained_weights"]),
        scene_config=str(primary_cfg["scene_predicates"]), grammar_path=str(primary_cfg["reason_grammar"]),
    ).to(device).eval(); foundation.load_from_acpr_state_dict(source.state_dict())
    with torch.no_grad():
        expected, actual = source(real_image), foundation(real_image)
    equivalence = max(float((actual[a] - expected[b]).abs().max().cpu()) for a, b in {
        "action_logits_primary": "action_logits_base", "reason_logits_primary": "reason_logits_base", "label_nodes": "label_nodes",
        "label_attention": "label_attention", "predicate_logits": "predicate_logits", "predicate_attention": "predicate_attention",
    }.items())
    real_dino = {
        "patch_shape": tuple(actual["patch_tokens_by_layer"].shape),
        "cls_shape": tuple(actual["cls_tokens_by_layer"].shape),
        "all_frozen": all(not parameter.requires_grad for parameter in foundation.dino.parameters()),
    }
    del source, foundation, expected, actual, real_image
    if device.type == "cuda": torch.cuda.empty_cache()
    image = torch.randn(2, 3, 360, 640, device=device)
    model = AIEOIAModel(use_mock_dino=True).to(device)
    out = model(image)
    shapes = {
        "action_logits": tuple(out["action_logits_final"].shape), "reason_logits": tuple(out["reason_logits_final"].shape),
        "probe_queries": tuple(out["probe_queries"].shape), "evidence_map": tuple(out["evidence_map"].shape),
        "evidence_token": tuple(out["evidence_token"].shape), "sampling_offsets": tuple(out["sampling_offsets"].shape),
        "sampling_weights": tuple(out["sampling_weights"].shape), "reason_private_attention": tuple(out["reason_private_attention"].shape),
    }
    expected_shapes = {
        "action_logits": (2, 4), "reason_logits": (2, 21), "probe_queries": (2, 4, 4, 384),
        "evidence_map": (2, 4, 4, 3600), "evidence_token": (2, 4, 4, 384),
        "sampling_offsets": (2, 4, 4, 3, 8, 2), "sampling_weights": (2, 4, 4, 3, 8),
        "reason_private_attention": (2, 21, 3, 3600),
    }
    gradients = gradient_probe(model, image)
    trajectory_difference = primary_trajectory_probe(model, image[:1])
    firewall = {
        "primary_to_aie_zero": gradients["primary_only"]["action_evidence"] == 0 and gradients["primary_only"]["action_contribution"] == 0 and gradients["primary_only"]["reason_private"] == 0,
        "action_to_primary_zero": gradients["final_action"]["primary"] == 0 and gradients["final_action"]["reason_private"] == 0,
        "reason_to_action_primary_zero": gradients["final_reason"]["primary"] == 0 and gradients["final_reason"]["action_evidence"] == 0 and gradients["final_reason"]["action_contribution"] == 0,
        "predicate_to_action_zero": gradients["predicate_only"]["action_evidence"] == 0,
        "dino_zero": all(values["dino"] == 0 for values in gradients.values()),
        "owners_active": gradients["primary_only"]["primary"] > 0 and gradients["final_action"]["action_evidence"] > 0 and gradients["final_action"]["action_contribution"] > 0 and gradients["final_reason"]["reason_private"] > 0,
        "cf_owner": not gradients["cf_only"].get("invalid", False) and gradients["cf_only"]["action_evidence"] > 0 and gradients["cf_only"]["action_contribution"] > 0 and gradients["cf_only"]["primary"] == 0 and gradients["cf_only"]["reason_private"] == 0 and gradients["cf_only"]["dino"] == 0,
    }
    functional = {
        "foundation_equivalence": equivalence < 1e-6,
        "dynamic_shapes": shapes == expected_shapes,
        "contribution_reconstruction": float(out["contribution_reconstruction_error"].detach().cpu()) < 1e-6,
        "contribution_nonzero": float(out["raw_contribution"].std().detach().cpu()) > 1e-3,
        "predicate_bias_bounded": float(out["predicate_bias_strength"].max().detach().cpu()) <= 0.25 + 1e-7,
        "offset_bounded": float(out["sampling_offsets"].abs().max().detach().cpu()) <= 0.25 + 1e-7,
        "maps_normalized": bool(torch.allclose(out["evidence_map"].sum(-1), torch.ones_like(out["evidence_map"].sum(-1)), atol=1e-5)),
        "real_dino_smoke": real_dino["patch_shape"] == (1, 3, 3600, 384) and real_dino["cls_shape"] == (1, 3, 384) and real_dino["all_frozen"],
        "primary_trajectory_isolation": trajectory_difference < 1e-7,
        "action_logit_cap": float(out["action_logits_final"].norm(dim=-1).max().detach().cpu()) <= float(cfg["training"]["action_logit_norm_cap"]) + 1e-5,
        "firewalls": all(firewall.values()),
    }
    status = not missing and not forbidden and all(functional.values())
    predicate_schema_hash = file_sha256("configs/aie_scene_predicates.yaml")
    counter_evidence_schema_hash = file_sha256("configs/aie_reason_counter_evidence.yaml")
    optimizer_ownership = {
        "exact_cover": len({id(parameter) for parameters in exact_owner_parameter_groups(model).values() for parameter in parameters})
        == len([parameter for parameter in model.parameters() if parameter.requires_grad]),
        "owner_parameter_counts": {owner: len(parameters) for owner, parameters in exact_owner_parameter_groups(model).items()},
    }
    failures = []
    failures.extend(f"missing:{item}" for item in missing)
    failures.extend(f"forbidden:{path}:{pattern}" for path, patterns in forbidden.items() for pattern in patterns)
    failures.extend(f"functional:{name}" for name, passed in functional.items() if not passed)
    if not optimizer_ownership["exact_cover"]:
        failures.append("optimizer_ownership:not_exact_cover")
    status = status and not failures
    payload = {
        "status": "REVIEW_PASS" if status else "REVIEW_FAIL", "pass": status, "git_head": git_value("rev-parse", "HEAD"),
        "source_head": "373aa49feac17372574fd7fb056c1d79c7c848fe", "config_hash": file_sha256(args.config),
        "source_tree_hash": aie_source_tree_sha256(),
        "predicate_schema_hash": predicate_schema_hash,
        "counter_evidence_schema_hash": counter_evidence_schema_hash,
        "schema_hash": object_sha256({"required": REQUIRED_FILES, "shapes": expected_shapes}), "checked_files": REQUIRED_FILES,
        "missing_items": missing, "forbidden_pattern_results": forbidden, "forbidden_paths": forbidden, "functional_checks": functional,
        "source_equivalence": {"max_abs_error": equivalence, "pass": equivalence < 1e-6},
        "gradient_ownership": {"owners": gradients, "firewalls": firewall},
        "optimizer_ownership": optimizer_ownership,
        "runtime": {"real_dino_smoke": real_dino, "dynamic_shapes": shapes},
        "failures": failures,
        "foundation_max_abs_error": equivalence, "real_dino_smoke": real_dino, "primary_trajectory_max_abs_difference": trajectory_difference,
        "dynamic_shapes": shapes, "owner_gradients": gradients, "firewall_checks": firewall,
        "warnings": ["REVIEW_PASS proves implementation integrity, not metric superiority."],
    }
    write_json(output_dir / "AIE_IMPLEMENTATION_REVIEW.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    raise SystemExit(0 if status else 1)


if __name__ == "__main__": main()
