from __future__ import annotations

from typing import Any, Iterable

import torch

from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.losses.precise_intervention_losses import packed_target_specific_interventions


EVAL_BRANCHES = (
    "action_direct", "action_reread_no_exchange", "action_ungated_exchange", "action_certified_exchange", "action_final_raw", "action_deploy",
    "reason_direct", "reason_semantic", "reason_observed", "reason_deploy",
    "action_explicit_only", "reason_explicit_only", "action_latent_only", "reason_latent_only",
    "action_exchange_off", "reason_exchange_off", "action_evidence_shuffled", "reason_evidence_shuffled",
    "action_reason_token_shuffled", "reason_reason_token_shuffled", "action_annotation_off", "reason_annotation_off",
)


@torch.no_grad()
def evaluate_precise(model: torch.nn.Module, loader: Iterable[dict[str, Any]], device: torch.device) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in EVAL_BRANCHES}
    actions, reasons, file_names = [], [], []
    evidence_reliability, evidence_presence, evidence_coordinates = [], [], []
    counterfactual_rows = []
    case_rows: list[dict[str, Any]] = []
    for batch in loader:
        output = model(batch["image"].to(device, non_blocking=True))
        branches = output["branch_logits"]
        for name in EVAL_BRANCHES:
            collected[name].append(branches[name].detach().cpu())
        actions.append(batch["action"].cpu())
        reasons.append(batch["reason"].cpu())
        file_names.extend(batch["file_name"])
        evidence_reliability.append(output["evidence_reliability"].detach().cpu())
        evidence_presence.append(torch.sigmoid(output["evidence_presence_logits"]).detach().cpu())
        evidence_coordinates.append(output["evidence_part_coordinates"].detach().cpu())
        if sum(int(row["count"]) for row in counterfactual_rows) < 64:
            intervention = packed_target_specific_interventions(model, output, batch["action"].to(device), batch["reason"].to(device), max_pairs=24)
            counterfactual_rows.append({"selected": float(intervention["selected_effect_mean"]), "control": float(intervention["control_effect_mean"]), "wrong": float(intervention["wrong_effect_mean"]), "count": float(intervention["intervention_pair_count"]), "sign": float(intervention["sign_agreement"]), **{key: intervention[key].detach().cpu() for key in intervention if "per_target_" in key}})
        for row_index, file_name in enumerate(batch["file_name"]):
            if len(case_rows) >= 32:
                break
            def take_row(value):
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == len(batch["file_name"]):
                    return value[row_index:row_index + 1]
                if isinstance(value, dict):
                    return {key: take_row(item) for key, item in value.items()}
                return value
            row_output = take_row(output)
            row_action = batch["action"][row_index:row_index + 1].to(device)
            row_reason = batch["reason"][row_index:row_index + 1].to(device)
            row_intervention = packed_target_specific_interventions(model, row_output, row_action, row_reason, max_pairs=8)
            row_branches = row_output["branch_logits"]
            case_rows.append({
                "file_name": file_name,
                "action": {name: value[0].detach().cpu().tolist() for name, value in row_branches.items() if name.startswith("action_")},
                "reason": {name: value[0].detach().cpu().tolist() for name, value in row_branches.items() if name.startswith("reason_")},
                "evidence": {
                    "field_names": [field["name"] for field in model.evidence_schema],
                    "presence": torch.sigmoid(row_output["evidence_presence_logits"])[0].detach().cpu().tolist(),
                    "reliability": row_output["evidence_reliability"][0].detach().cpu().tolist(),
                    "coordinates": row_output["evidence_part_coordinates"][0].detach().cpu().tolist(),
                    "action_attention": row_output["action_evidence_attention"][0].detach().cpu().tolist(),
                    "reason_attention": row_output["reason_evidence_attention"][0].detach().cpu().tolist(),
                },
                "counterfactual": {key: float(row_intervention[key]) for key in ("selected_effect_mean", "control_effect_mean", "wrong_effect_mean", "intervention_pair_count")},
            })
    action = torch.cat(actions)
    reason = torch.cat(reasons)
    tensors = {name: torch.cat(values) for name, values in collected.items()}
    metrics: dict[str, Any] = {}
    for name in EVAL_BRANCHES:
        target = action if name.startswith("action_") else reason
        prefix = "Act_" if name.startswith("action_") else "Exp_"
        metrics[name] = multilabel_metrics_from_logits(tensors[name], target, prefix=prefix)
    metrics["deploy_fixed_joint"] = 0.5 * metrics["action_deploy"]["Act_mF1"] + 0.5 * metrics["reason_deploy"]["Exp_mF1"]
    total_pairs = max(1.0, sum(row["count"] for row in counterfactual_rows))
    def per_target(task: str, size: int) -> list[dict[str, float]]:
        count = sum((row[f"{task}_per_target_count"] for row in counterfactual_rows), torch.zeros(size))
        denominator = count.clamp_min(1.0)
        values = {}
        for name in ("selected", "control", "wrong", "sign"):
            values[name] = sum((row[f"{task}_per_target_{name}_sum"] for row in counterfactual_rows), torch.zeros(size)) / denominator
        return [{"count": float(count[index]), **{name: float(value[index]) for name, value in values.items()}} for index in range(size)]
    metrics["counterfactual"] = {
        "selected_effect": sum(row["selected"] * row["count"] for row in counterfactual_rows) / total_pairs,
        "matched_control_effect": sum(row["control"] * row["count"] for row in counterfactual_rows) / total_pairs,
        "wrong_target_effect": sum(row["wrong"] * row["count"] for row in counterfactual_rows) / total_pairs,
        "target_specificity_margin": sum((row["selected"] - row["wrong"]) * row["count"] for row in counterfactual_rows) / total_pairs,
        "selected_control_margin": sum((row["selected"] - row["control"]) * row["count"] for row in counterfactual_rows) / total_pairs,
        "sign_agreement": sum(row["sign"] * row["count"] for row in counterfactual_rows) / total_pairs,
        "per_action": per_target("action", 4),
        "per_reason": per_target("reason", 21),
    }
    model.train()
    return metrics, {**tensors, "labels_action": action, "labels_reason": reason, "file_names": file_names, "evidence_reliability": torch.cat(evidence_reliability), "evidence_presence": torch.cat(evidence_presence), "evidence_coordinates": torch.cat(evidence_coordinates), "case_rows": case_rows}
