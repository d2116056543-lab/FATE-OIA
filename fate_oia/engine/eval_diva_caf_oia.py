from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from fate_oia.metrics import multilabel_metrics_from_logits


@torch.no_grad()
def evaluate_diva_caf(model, loader: DataLoader, device: str = "cuda") -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    action_logits = []
    fate_logits = []
    eva_logits = []
    reason_logits = []
    y_action = []
    y_reason = []
    for batch in loader:
        images = batch["image"].to(device)
        out = model(images=images, labels=None, train_mode=False)
        action_logits.append(out["guarded_action_logits"].detach().cpu())
        fate_logits.append(out["z_fate_action_logits"].detach().cpu())
        eva_logits.append(out["z_eva_action_logits"].detach().cpu())
        reason_logits.append(out["final_reason_logits"].detach().cpu())
        y_action.append(batch["action"].cpu())
        y_reason.append(batch["reason"].cpu())
    action_logits_t = torch.cat(action_logits)
    fate_logits_t = torch.cat(fate_logits)
    eva_logits_t = torch.cat(eva_logits)
    reason_logits_t = torch.cat(reason_logits)
    y_action_t = torch.cat(y_action)
    y_reason_t = torch.cat(y_reason)
    metrics: dict[str, Any] = {}
    metrics.update(multilabel_metrics_from_logits(action_logits_t, y_action_t, prefix="Act_"))
    metrics.update(multilabel_metrics_from_logits(fate_logits_t, y_action_t, prefix="Act_fate_"))
    metrics.update(multilabel_metrics_from_logits(eva_logits_t, y_action_t, prefix="Act_eva_"))
    metrics.update(multilabel_metrics_from_logits(reason_logits_t, y_reason_t, prefix="Exp_"))
    metrics["joint"] = 0.5 * metrics.get("Act_mF1", 0.0) + 0.5 * metrics.get("Exp_mF1", 0.0)
    tensors = {"action_logits": action_logits_t, "reason_logits": reason_logits_t, "labels_action": y_action_t, "labels_reason": y_reason_t}
    return metrics, tensors
