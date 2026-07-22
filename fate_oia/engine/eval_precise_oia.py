from __future__ import annotations

from typing import Any, Iterable

import torch

from fate_oia.metrics import multilabel_metrics_from_logits


EVAL_BRANCHES = ("action_direct", "action_reread_no_exchange", "action_ungated_exchange", "action_certified_exchange", "action_final_raw", "action_deploy", "reason_direct", "reason_semantic", "reason_observed", "reason_deploy")


@torch.no_grad()
def evaluate_precise(model: torch.nn.Module, loader: Iterable[dict[str, Any]], device: torch.device) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in EVAL_BRANCHES}
    actions, reasons = [], []
    for batch in loader:
        output = model(batch["image"].to(device, non_blocking=True))
        branches = output["branch_logits"]
        for name in EVAL_BRANCHES:
            collected[name].append(branches[name].detach().cpu())
        actions.append(batch["action"].cpu())
        reasons.append(batch["reason"].cpu())
    action = torch.cat(actions)
    reason = torch.cat(reasons)
    tensors = {name: torch.cat(values) for name, values in collected.items()}
    metrics: dict[str, Any] = {}
    for name in EVAL_BRANCHES:
        target = action if name.startswith("action_") else reason
        prefix = "Act_" if name.startswith("action_") else "Exp_"
        metrics[name] = multilabel_metrics_from_logits(tensors[name], target, prefix=prefix)
    metrics["deploy_fixed_joint"] = 0.5 * metrics["action_deploy"]["Act_mF1"] + 0.5 * metrics["reason_deploy"]["Exp_mF1"]
    model.train()
    return metrics, {**tensors, "labels_action": action, "labels_reason": reason}
