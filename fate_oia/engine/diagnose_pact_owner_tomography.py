from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Subset

from fate_oia.engine.train_aie_oia import canonical_model_state_dict, make_dataset
from fate_oia.engine.train_pact_oia_probe import load_config, make_loader
from fate_oia.models.aie_oia_model import AIEOIAModel
from fate_oia.utils.aie_calibration import apply_posthoc_threshold
from fate_oia.utils.aie_metrics import aie_branch_metrics
from fate_oia.utils.pact_artifacts import write_json


def build(cfg, device):
    return AIEOIAModel(
        dim=cfg["primary"]["dim"], selected_layers=tuple(cfg["backbone"]["selected_layers"]),
        pretrained_weights=cfg["backbone"]["pretrained_weights"], scene_config=cfg["primary"]["scene_predicates"],
        grammar_path=cfg["primary"]["reason_grammar"], probes_per_action=cfg["evidence"]["probes_per_action"],
        local_points_per_layer=cfg["evidence"]["local_points_per_layer"], max_offset=cfg["evidence"]["max_offset"],
        predicate_bias_max=cfg["evidence"]["predicate_bias_max"], probe_chunk_size=cfg["evidence"]["probe_chunk_size"],
        action_kappa=cfg["evidence"]["action_kappa"], reason_kappa=cfg["reason"]["kappa"],
    ).to(device).eval()


def compose(base, foundation, action, reason):
    result = {}
    for key, value in base.items():
        source = foundation if key.startswith("foundation.") else action if key.startswith(("action_evidence.", "action_contribution.", "predicate_naming.")) else reason
        result[key] = source.get(key, value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    parser.add_argument("--joint-checkpoint", required=True); parser.add_argument("--action-checkpoint", required=True)
    parser.add_argument("--reason-map-checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args(); cfg = load_config(args.config); device = torch.device(args.device)
    checkpoints = {name: torch.load(path, map_location="cpu") for name, path in
                   (("joint", args.joint_checkpoint), ("action", args.action_checkpoint), ("reason", args.reason_map_checkpoint))}
    states = {name: canonical_model_state_dict(value["model"]) for name, value in checkpoints.items()}
    definitions = {
        "C1_joint_all": ("joint", "joint", "joint"),
        "C2_action_all": ("action", "action", "action"),
        "C3_jointF_actionA_jointR": ("joint", "action", "joint"),
        "C4_actionF_actionA_reasonR": ("action", "action", "reason"),
    }
    models = {}
    template = build(cfg, device); base = template.state_dict()
    for name, (foundation, action, reason) in definitions.items():
        model = build(cfg, device)
        model.load_state_dict(compose(base, states[foundation], states[action], states[reason]), strict=True)
        models[name] = model
    dataset = make_dataset(cfg, "test"); count = min(args.max_test_samples or len(dataset), len(dataset))
    loader = make_loader(Subset(dataset, list(range(count))), 6, False, 8, cfg)
    stores = {name: {key: [] for key in ("action", "reason")} for name in models}; targets = {"action": [], "reason": []}
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for batch in loader:
            field = models["C1_joint_all"].encode_images(batch["image"].to(device))
            for name, model in models.items():
                out = model.decode_from_field(field, action_scale=1.0, reason_scale=0.60)
                stores[name]["action"].append(out["action_logits_final"].cpu())
                stores[name]["reason"].append(out["reason_logits_final"].cpu())
            targets["action"].append(batch["action"]); targets["reason"].append(batch["reason"])
    targets = {key: torch.cat(value) for key, value in targets.items()}; results = {}
    for name, values in stores.items():
        action, reason = torch.cat(values["action"]), torch.cat(values["reason"])
        raw = aie_branch_metrics(action, reason, targets["action"], targets["reason"])
        calibration = checkpoints[definitions[name][0]]["calibration"]["threshold_prob"]
        deploy = aie_branch_metrics(apply_posthoc_threshold(action, calibration[:4]),
                                    apply_posthoc_threshold(reason, calibration[4:]), targets["action"], targets["reason"])
        results[name] = {"owners": definitions[name], "raw": raw, "deploy": deploy}
    write_json(args.output, {"samples": count, "combinations": results})
    table = ["| combination | owners F/A/R | Act_mF1 | Act_mAP | Exp_mF1 | Exp_mAP | joint |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, row in results.items():
        metric = row["deploy"]
        table.append(f"| {name} | {'/'.join(row['owners'])} | {metric['Act_mF1']:.6f} | "
                     f"{metric['Act_mAP']:.6f} | {metric['Exp_mF1']:.6f} | {metric['Exp_mAP']:.6f} | {metric['joint']:.6f} |")
    Path(args.output).with_name("owner_tomography_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
