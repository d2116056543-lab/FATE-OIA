from __future__ import annotations

from typing import Any

import torch

from fate_oia.utils.lens_metrics import deploy_joint, multilabel_metrics


@torch.no_grad()
def evaluate_lens(model, loader, device: torch.device, *, progress: float, action_threshold: torch.Tensor | float = 0.5, reason_threshold: torch.Tensor | float = 0.5) -> tuple[dict[str, Any], dict[str, torch.Tensor | list[str]]]:
    model.eval(); store: dict[str, list] = {"action": [], "reason": [], "action_base": [], "reason_source": [], "labels_action": [], "labels_reason": [], "file_names": []}
    for batch in loader:
        image=batch["image"].to(device, non_blocking=True); out=model(image, progress=progress)
        for key, out_key in [("action", "action_logits_final"), ("reason", "reason_logits_formal"), ("action_base", "action_logits_base"), ("reason_source", "reason_logits_source")]: store[key].append(out[out_key].detach().cpu())
        store["labels_action"].append(batch["action"].cpu()); store["labels_reason"].append(batch["reason"].cpu()); store["file_names"].extend(batch["file_name"])
    merged={key:(torch.cat(value) if key != "file_names" else value) for key,value in store.items()}
    raw_action=multilabel_metrics(merged["action"],merged["labels_action"],0.5); raw_reason=multilabel_metrics(merged["reason"],merged["labels_reason"],0.5)
    deploy_action=multilabel_metrics(merged["action"],merged["labels_action"],action_threshold); deploy_reason=multilabel_metrics(merged["reason"],merged["labels_reason"],reason_threshold)
    metrics={"raw_action":raw_action,"raw_reason":raw_reason,"deploy_action":deploy_action,"deploy_reason":deploy_reason,"deploy_joint":deploy_joint(deploy_action,deploy_reason)}
    return metrics, merged
