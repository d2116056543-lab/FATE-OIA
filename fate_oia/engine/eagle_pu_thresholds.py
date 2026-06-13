from __future__ import annotations

import torch


def fixed_thresholds(num_labels: int, value: float = 0.5) -> torch.Tensor:
    return torch.full((num_labels,), float(value))


def global_threshold_diagnostic(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Diagnostic only: choose a single threshold from a small grid on the evaluated split.
    probs = torch.sigmoid(logits)
    best_t, best_f1 = 0.5, -1.0
    for t in torch.linspace(0.1, 0.9, 17, device=logits.device):
        pred = (probs >= t).float(); y = targets.float()
        tp = (pred*y).sum(); fp = (pred*(1-y)).sum(); fn = ((1-pred)*y).sum()
        f1 = (2*tp/(2*tp+fp+fn+1e-9)).item()
        if f1 > best_f1:
            best_f1, best_t = f1, float(t.item())
    return torch.full((logits.shape[1],), best_t, device=logits.device)


def per_label_threshold_diagnostic(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    outs = []
    for i in range(logits.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        for t in torch.linspace(0.1, 0.9, 17, device=logits.device):
            pred = (probs[:, i] >= t).float(); y = targets[:, i].float()
            tp = (pred*y).sum(); fp = (pred*(1-y)).sum(); fn = ((1-pred)*y).sum()
            f1 = (2*tp/(2*tp+fp+fn+1e-9)).item()
            if f1 > best_f1:
                best_f1, best_t = f1, float(t.item())
        outs.append(best_t)
    return torch.tensor(outs, device=logits.device)
