from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Subset

from fate_oia.datasets.aie_splits import stable_split_ids
from fate_oia.datasets.aie_structured_evidence import AIEStructuredEvidenceBuilder
from fate_oia.engine.train_pact_oia_probe import build_model, load_config, load_source, make_dataset, make_loader
from fate_oia.losses import acpr_losses
from fate_oia.losses.aie_losses import (
    predicate_map_loss, predicate_masked_asl_loss, predicate_reason_alignment_pu_loss, soft_f1_loss,
)
from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.utils.aie_metrics import aie_branch_metrics
from fate_oia.utils.pact_artifacts import write_json


def _gradients(loss: Tensor, parameters: tuple[Tensor, ...]) -> tuple[Tensor | None, ...]:
    return torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)


def _geometry_from_gradients(action_grad, semantic_grad, blocks: dict[str, tuple[Tensor, ...]]) -> dict:
    rows, offset = {}, 0
    for name, block in blocks.items():
        count = len(block)
        ga = torch.cat([(torch.zeros_like(parameter) if grad is None else grad).float().flatten()
                        for parameter, grad in zip(block, action_grad[offset:offset + count])])
        gs = torch.cat([(torch.zeros_like(parameter) if grad is None else grad).float().flatten()
                        for parameter, grad in zip(block, semantic_grad[offset:offset + count])])
        offset += count
        dot = torch.dot(ga, gs)
        rows[name] = {
            "cosine": float(dot / (ga.norm() * gs.norm()).clamp_min(1e-12)),
            "signed_dot": float(dot), "action_norm": float(ga.norm()), "semantic_norm": float(gs.norm()),
        }
    return rows


def _summarize(rows: list[dict], blocks: dict[str, tuple[Tensor, ...]]) -> dict:
    summary = {}
    for name in blocks:
        values = [row[name] for row in rows]
        summary[name] = {key: float(sum(value[key] for value in values) / len(values)) for key in values[0]}
        summary[name]["negative_cosine_rate"] = sum(value["cosine"] < 0 for value in values) / len(values)
    return summary


@torch.no_grad()
def _scale_scan(model, loader, device, candidates) -> tuple[list[dict], dict]:
    stores = {candidate: {key: [] for key in ("a0", "r0", "a", "r", "ay", "ry")} for candidate in candidates}
    original_max = model.predicate_agreement.lambda_max
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            field = model.encode_images(image)
            for action_scale, reason_budget, predicate_gate in candidates:
                model.predicate_agreement.lambda_max = predicate_gate
                output = model.decode_from_field(field, semantic_share_license=1.0, action_scale=action_scale,
                                                 reason_budget=reason_budget)
                store = stores[(action_scale, reason_budget, predicate_gate)]
                for key, value in (("a0", output["action_logits_primary"]), ("r0", output["reason_logits_primary"]),
                                   ("a", output["action_logits_final"]), ("r", output["reason_logits_final"])):
                    store[key].append(value.float().cpu())
                store["ay"].append(batch["action"]); store["ry"].append(batch["reason"])
    model.predicate_agreement.lambda_max = original_max
    rows = []
    for candidate, store in stores.items():
        store = {key: torch.cat(value) for key, value in store.items()}
        primary = aie_branch_metrics(store["a0"], store["r0"], store["ay"], store["ry"])
        final = aie_branch_metrics(store["a"], store["r"], store["ay"], store["ry"])
        safe = final["Act_mAP"] >= primary["Act_mAP"] - 0.0005 and final["Exp_mAP"] >= primary["Exp_mAP"]
        rows.append({"action_scale": candidate[0], "reason_budget": candidate[1], "predicate_gate": candidate[2],
                     "primary": primary, "final": final, "safe": safe})
    safe_rows = [row for row in rows if row["safe"]]
    selected = max(safe_rows, key=lambda row: row["final"]["joint"]) if safe_rows else max(rows, key=lambda row: row["final"]["joint"])
    return rows, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--scale-samples", type=int, default=1024); parser.add_argument("--conflict-samples", type=int, default=512)
    parser.add_argument("--conflict-batch-size", type=int, default=10)
    parser.add_argument("--skip-scale", action="store_true")
    args = parser.parse_args(); cfg = load_config(args.config); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); model = build_model(cfg, device); load_source(model, args.source_checkpoint); model.eval()
    train = make_dataset(cfg, "train"); names = [sample.file_name for sample in train.samples]
    split = stable_split_ids(names, int(cfg["data"]["split_seed"]), cfg["data"]["train_calib_fraction"], cfg["data"]["train_audit_count"])
    index = {sample.file_name: i for i, sample in enumerate(train.samples)}
    ids = [index[name] for name in split["train_audit"]]
    if args.skip_scale:
        selected = json.loads((output / "selected_probe_hparams.json").read_text(encoding="utf-8"))
    else:
        scale_loader = make_loader(Subset(train, ids[:args.scale_samples]), 6, False, 8, cfg)
        candidates = list(itertools.product((0.60, 0.75, 0.90, 1.00), (0.45, 0.55, 0.60), (0.15, 0.20, 0.25)))
        rows, selected = _scale_scan(model, scale_loader, device, candidates)
        write_json(output / "pareto_scale_audit.json", {"split": "train_audit", "samples": min(args.scale_samples, len(ids)), "candidates": rows})
        write_json(output / "selected_probe_hparams.json", selected)

    structured_builder = AIEStructuredEvidenceBuilder(cfg["primary"]["scene_predicates"], cfg["data"]["bdd100k_root"])
    conflict_loader = make_loader(Subset(train, ids[:args.conflict_samples]), args.conflict_batch_size, False, 8, cfg)
    batches = []; per_action_batches = {name: [] for name in ("forward", "stop", "left", "right")}
    reason_groups = {"traffic_control": (0, 3, 4, 13, 19), "road": (1, 2),
                     "front_object": (5, 6, 7, 8, 14, 20), "lane": (9, 10, 11, 12, 15, 16, 17, 18)}
    per_reason_group_batches = {name: [] for name in reason_groups}
    blocks = {
        "shared_visual_readout": tuple(model.shared_readout.parameters()),
        "context_self_attention": tuple(model.context_decoder.label_self_attn.parameters()),
        "predicate_cross_attention": tuple(model.context_decoder.predicate_cross_attn.parameters()),
        "ego_encoder": tuple(model.ego.parameters()), "predicate_head": tuple(model.predicate_head.parameters()),
    }
    for batch_index, batch in enumerate(conflict_loader):
        batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
        structured = structured_builder.build(batch["file_name"], device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output_row = model(batch["image"], semantic_share_license=1.0, action_scale=selected["action_scale"],
                               reason_budget=selected["reason_budget"])
            action_loss = (asymmetric_loss_with_logits(output_row["action_logits_primary"], batch["action"]) +
                           0.05 * asymmetric_loss_with_logits(output_row["action_visual_logits_primary"], batch["action"]) +
                           0.05 * asymmetric_loss_with_logits(output_row["action_reason_logits_primary"], batch["action"]))
            semantic_loss = (acpr_losses.partial_label_reason_loss(output_row["reason_logits_primary"], batch["reason"], output_row["contradiction_score"]) +
                             0.04 * soft_f1_loss(output_row["reason_logits_primary"], batch["reason"]) +
                             0.08 * predicate_masked_asl_loss(
                                 output_row["semantic_predicate_logits"], structured["predicate_target"],
                                 structured["predicate_target_mask"], structured["predicate_counter_mask"],
                                 structured["predicate_reliability"]) +
                             0.04 * predicate_map_loss(output_row["semantic_predicate_attention"], structured["predicate_map_target"], structured["predicate_map_mask"]) +
                             0.03 * predicate_reason_alignment_pu_loss(
                                 output_row["semantic_predicate_probs"], batch["reason"], output_row["_grammar_positive_mask"],
                                 output_row["_grammar_contradictory_mask"], output_row["contradiction_score"]))
            action_losses = {name: asymmetric_loss_with_logits(output_row["action_logits_primary"][:, label:label + 1],
                                                               batch["action"][:, label:label + 1])
                             for label, name in enumerate(per_action_batches)}
            group_losses = {name: acpr_losses.partial_label_reason_loss(
                                output_row["reason_logits_primary"][:, labels], batch["reason"][:, labels],
                                output_row["contradiction_score"][:, labels]) for name, labels in reason_groups.items()}
        parameters = tuple(parameter for block in blocks.values() for parameter in block)
        action_gradient = _gradients(action_loss, parameters); semantic_gradient = _gradients(semantic_loss, parameters)
        batches.append(_geometry_from_gradients(action_gradient, semantic_gradient, blocks))
        for name, loss in action_losses.items():
            per_action_batches[name].append(_geometry_from_gradients(_gradients(loss, parameters), semantic_gradient, blocks))
        for name, loss in group_losses.items():
            per_reason_group_batches[name].append(_geometry_from_gradients(action_gradient, _gradients(loss, parameters), blocks))
        model.zero_grad(set_to_none=True)
        if (batch_index + 1) % 8 == 0:
            write_json(output / "conflict_localization_progress.json", {
                "batches_complete": batch_index + 1,
                "samples_complete": min((batch_index + 1) * args.conflict_batch_size, args.conflict_samples),
            })
    summary = _summarize(batches, blocks)
    per_action = {name: _summarize(rows, blocks) for name, rows in per_action_batches.items()}
    per_reason_group = {name: _summarize(rows, blocks) for name, rows in per_reason_group_batches.items()}
    evidence = max((row["negative_cosine_rate"] for row in summary.values()), default=0.0)
    write_json(output / "conflict_localization.json", {"split": "train_audit", "samples": min(args.conflict_samples, len(ids)),
               "per_parameter_block": summary, "per_action": per_action, "per_reason_group": per_reason_group,
               "max_negative_cosine_rate": evidence,
               "core_hypothesis_supported": evidence >= 0.15})


if __name__ == "__main__":
    main()
