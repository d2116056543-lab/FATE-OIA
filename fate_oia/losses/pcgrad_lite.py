from __future__ import annotations

import torch


def pcgrad_project(g1: torch.Tensor, g2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    dot = torch.dot(g1.flatten(), g2.flatten())
    if dot >= 0:
        return g1, g2, False
    denom2 = torch.dot(g2.flatten(), g2.flatten()).clamp_min(1e-12)
    denom1 = torch.dot(g1.flatten(), g1.flatten()).clamp_min(1e-12)
    p1 = g1 - dot / denom2 * g2
    p2 = g2 - dot / denom1 * g1
    return p1, p2, True


def _flatten_grads(loss: torch.Tensor, params: list[torch.nn.Parameter], retain_graph: bool) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    flat: list[torch.Tensor] = []
    for p, g in zip(params, grads):
        flat.append(torch.zeros_like(p).flatten() if g is None else g.detach().flatten())
    return torch.cat(flat) if flat else loss.new_zeros(0)


def apply_pcgrad_lite(
    losses: list[torch.Tensor],
    params: list[torch.nn.Parameter],
    retain_graph: bool = True,
    grad_accumulation_steps: int = 1,
) -> dict[str, float | int | bool]:
    params = [p for p in params if p.requires_grad]
    valid_losses = [loss for loss in losses if torch.is_tensor(loss)]
    if len(valid_losses) < 2 or not params:
        return {
            "pcgrad_task_count": len(valid_losses),
            "pcgrad_conflict_count": 0,
            "projection_applied_count": 0,
            "pcgrad_mean_dot": 0.0,
            "grad_accumulation_steps": int(grad_accumulation_steps),
            "accumulated_microbatches": 1,
            "overwrote_existing_grad": False,
        }
    scaled = [loss / max(int(grad_accumulation_steps), 1) for loss in valid_losses]
    grads = [_flatten_grads(loss, params, retain_graph=retain_graph) for loss in scaled]
    conflict = 0
    projections = 0
    dots: list[float] = []
    projected = [g.clone() for g in grads]
    for i in range(len(projected)):
        for j in range(len(projected)):
            if i == j:
                continue
            dot = torch.dot(projected[i], projected[j])
            dots.append(float(dot.detach().cpu()))
            pi, _, is_conflict = pcgrad_project(projected[i], projected[j])
            projected[i] = pi
            conflict += int(is_conflict)
            projections += int(is_conflict)
    combined = torch.stack(projected, dim=0).mean(dim=0)
    offset = 0
    overwrote = False
    for p in params:
        n = p.numel()
        grad = combined[offset : offset + n].view_as(p).to(p.device, p.dtype)
        if p.grad is None:
            p.grad = grad.clone()
        else:
            p.grad = p.grad + grad
        offset += n
    return {
        "pcgrad_task_count": len(valid_losses),
        "pcgrad_conflict_count": int(conflict),
        "conflict_count": int(conflict),
        "projection_applied_count": int(projections),
        "pcgrad_mean_dot": float(sum(dots) / max(len(dots), 1)),
        "grad_accumulation_steps": int(grad_accumulation_steps),
        "accumulated_microbatches": 1,
        "overwrote_existing_grad": bool(overwrote),
    }
