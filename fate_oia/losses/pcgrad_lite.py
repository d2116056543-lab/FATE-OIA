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


def apply_pcgrad_lite(losses: list[torch.Tensor], params: list[torch.nn.Parameter], retain_graph: bool = True) -> dict[str, float | int]:
    params = [p for p in params if p.requires_grad]
    if len(losses) < 2 or not params:
        return {"pcgrad_task_count": len(losses), "pcgrad_conflict_count": 0, "pcgrad_mean_dot": 0.0}
    grads = []
    for loss in losses:
        gs = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
        flat = []
        for p, g in zip(params, gs):
            flat.append(torch.zeros_like(p).flatten() if g is None else g.detach().flatten())
        grads.append(torch.cat(flat))
    conflict = 0
    dots = []
    projected = grads[:]
    for i in range(len(projected)):
        for j in range(len(projected)):
            if i == j:
                continue
            pi, _, is_conflict = pcgrad_project(projected[i], projected[j])
            projected[i] = pi
            conflict += int(is_conflict)
            dots.append(float(torch.dot(grads[i], grads[j]).detach().cpu()))
    combined = torch.stack(projected, dim=0).mean(dim=0)
    offset = 0
    for p in params:
        n = p.numel()
        grad = combined[offset : offset + n].view_as(p).to(p.device, p.dtype)
        p.grad = grad.clone()
        offset += n
    return {"pcgrad_task_count": len(losses), "pcgrad_conflict_count": conflict, "pcgrad_mean_dot": float(sum(dots) / max(len(dots), 1))}
