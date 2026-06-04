from __future__ import annotations

import torch


def _flatten_grads(grads: list[torch.Tensor | None], params: list[torch.nn.Parameter]) -> torch.Tensor:
    pieces = []
    for grad, param in zip(grads, params):
        pieces.append((torch.zeros_like(param) if grad is None else grad).reshape(-1))
    return torch.cat(pieces) if pieces else torch.empty(0)


def _unflatten_grad(flat: torch.Tensor, params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    out = []
    offset = 0
    for param in params:
        size = param.numel()
        out.append(flat[offset : offset + size].view_as(param))
        offset += size
    return out


def compute_pcgrad_lite(
    losses: list[torch.Tensor],
    parameters,
    retain_graph: bool = True,
    eps: float = 1e-12,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Compute conflict-aware PCGrad-lite gradients for shared parameters."""

    params = [p for p in parameters if p.requires_grad]
    if not params or not losses:
        return [], {"pcgrad_task_count": float(len(losses)), "pcgrad_conflict_count": 0.0, "pcgrad_mean_dot_before": 0.0}
    task_grads = []
    for loss in losses:
        grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
        task_grads.append(_flatten_grads(list(grads), params))
    projected = [grad.clone() for grad in task_grads]
    dot_values = []
    conflict_count = 0
    conflict_strengths = []
    for i in range(len(projected)):
        for j in range(len(task_grads)):
            if i == j:
                continue
            dot = torch.dot(projected[i], task_grads[j])
            dot_values.append(float(dot.detach().cpu()))
            if dot < 0:
                conflict_strengths.append(float((-dot).detach().cpu()))
                projected[i] = projected[i] - dot / (torch.dot(task_grads[j], task_grads[j]) + eps) * task_grads[j]
                conflict_count += 1
    merged = torch.stack(projected, dim=0).mean(dim=0)
    norm_before = torch.stack([grad.norm() for grad in task_grads]).mean()
    stats = {
        "pcgrad_task_count": float(len(losses)),
        "pcgrad_conflict_count": float(conflict_count),
        "pcgrad_mean_dot_before": float(sum(dot_values) / len(dot_values)) if dot_values else 0.0,
        "pcgrad_grad_norm_before": float(norm_before.detach().cpu()),
        "pcgrad_grad_norm_after": float(merged.norm().detach().cpu()),
        "pairwise_negative_dot_count": float(conflict_count),
        "projection_applied_count": float(conflict_count),
        "mean_conflict_strength": float(sum(conflict_strengths) / len(conflict_strengths)) if conflict_strengths else 0.0,
    }
    return _unflatten_grad(merged, params), stats


def assign_pcgrad_lite(parameters, projected_grads: list[torch.Tensor]) -> None:
    params = [p for p in parameters if p.requires_grad]
    for param, grad in zip(params, projected_grads):
        param.grad = grad.detach().clone()


def apply_pcgrad_lite(losses: list[torch.Tensor], parameters, retain_graph: bool = True) -> dict[str, float]:
    params = list(parameters)
    projected, stats = compute_pcgrad_lite(losses, params, retain_graph=retain_graph)
    assign_pcgrad_lite(params, projected)
    return stats
