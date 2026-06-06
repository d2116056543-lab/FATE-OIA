from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.models.diva_visual_mixture_gate import branch_safe_guarded_action


@torch.no_grad()
def evaluate_diva_caf(model, loader: DataLoader, device: str = "cuda") -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    stores = {k: [] for k in ["fate", "eva", "actor", "base_reason", "factor_reason", "final_reason", "y_action", "y_reason"]}
    for batch in loader:
        images = batch["image"].to(device)
        out = model(images=images, labels=None, train_mode=False)
        stores["fate"].append(out["z_fate_action_logits"].detach().cpu())
        stores["eva"].append(out["z_eva_action_logits"].detach().cpu())
        stores["actor"].append(out["z_actor_action_logits"].detach().cpu())
        stores["base_reason"].append(out["base_reason_logits"].detach().cpu())
        stores["factor_reason"].append(out["reason_factor_logits"].detach().cpu())
        stores["final_reason"].append(out["final_reason_logits"].detach().cpu())
        stores["y_action"].append(batch["action"].cpu())
        stores["y_reason"].append(batch["reason"].cpu())
    tensors = {k: torch.cat(v) for k, v in stores.items()}
    guarded, guard_stats = branch_safe_guarded_action(tensors["fate"], tensors["actor"], tensors["y_action"], tolerance=0.0)
    tensors["guarded"] = guarded
    metrics: dict[str, Any] = {}
    metrics.update(multilabel_metrics_from_logits(tensors["guarded"], tensors["y_action"], prefix="Act_"))
    metrics.update(multilabel_metrics_from_logits(tensors["fate"], tensors["y_action"], prefix="Act_fate_"))
    metrics.update(multilabel_metrics_from_logits(tensors["eva"], tensors["y_action"], prefix="Act_eva_"))
    metrics.update(multilabel_metrics_from_logits(tensors["actor"], tensors["y_action"], prefix="Act_actor_"))
    metrics.update(multilabel_metrics_from_logits(tensors["base_reason"], tensors["y_reason"], prefix="Exp_base_"))
    metrics.update(multilabel_metrics_from_logits(tensors["factor_reason"], tensors["y_reason"], prefix="Exp_factor_"))
    metrics.update(multilabel_metrics_from_logits(tensors["final_reason"], tensors["y_reason"], prefix="Exp_"))
    metrics["joint"] = 0.5 * metrics.get("Act_mF1", 0.0) + 0.5 * metrics.get("Exp_mF1", 0.0)
    metrics["guarded_source"] = guard_stats["guarded_source"]
    metrics["actor_minus_fate_mF1"] = guard_stats["actor_minus_fate_mF1"]
    return metrics, tensors
