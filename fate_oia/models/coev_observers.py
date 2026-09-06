from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def inverse_grid_sample_affine(theta: Tensor, source_points: Tensor) -> Tensor:
    """Map source-image points to output coordinates for affine_grid/grid_sample theta."""
    linear=theta[...,:2];offset=theta[...,2]
    return torch.linalg.solve(linear,(source_points-offset.unsqueeze(-2)).transpose(-1,-2)).transpose(-1,-2)


class PredicateObserver(nn.Module):
    def __init__(self, dim: int = 384, count: int = 8) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, count))

    def forward(self, low8: Tensor, grid_hw: tuple[int, int]) -> Tensor:
        b, t, n, d = low8.shape
        if n != grid_hw[0] * grid_hw[1]:
            raise ValueError("predicate grid/token mismatch")
        return self.head(low8).sigmoid().permute(0, 1, 3, 2).reshape(b, t, 8, *grid_hw)


class CorrespondenceObserver(nn.Module):
    def __init__(self, dim: int = 384, qk_dim: int = 128, temperature: float = 0.07) -> None:
        super().__init__()
        self.q = nn.Linear(dim, qk_dim); self.k = nn.Linear(dim, qk_dim)
        self.dustbin = nn.Parameter(torch.zeros(())); self.temperature = temperature

    def forward(self, low8: Tensor, grid_hw: tuple[int, int] = (16, 28)) -> dict[str, Tensor]:
        b, t, n, d = low8.shape
        source_hw = (int(round(n ** 0.5)), int(round(n / max(1, round(n ** 0.5)))))
        # Token grids are supplied row-major; interpolate via the known aspect when needed.
        if n == 448:
            source_hw = (16, 28)
        elif source_hw[0] * source_hw[1] != n:
            source_hw = (32, 56) if n == 1792 else (45, 80)
        x = low8.view(b * t, source_hw[0], source_hw[1], d).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, grid_hw).permute(0, 2, 3, 1).reshape(b, t, -1, d)
        q = F.normalize(self.q(x[:, :-1]), dim=-1); k = F.normalize(self.k(x[:, 1:]), dim=-1)
        scores = torch.einsum("btid,btjd->btij", q, k) / self.temperature
        dust = self.dustbin.expand(*scores.shape[:-1], 1)
        forward = torch.cat((scores, dust), -1).softmax(-1)
        reverse = torch.cat((scores.transpose(-1, -2), dust), -1).softmax(-1)
        return {"forward": forward, "reverse": reverse, "grid_features": x}


def matched_expectation(prob: Tensor, grid_hw: tuple[int, int] = (16, 28), min_mass: float = 0.05) -> tuple[Tensor, Tensor, Tensor]:
    prob=prob.float()
    h, w = grid_hw
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=prob.device),
                            torch.linspace(-1, 1, w, device=prob.device), indexing="ij")
    coords = torch.stack((xx, yy), -1).reshape(-1, 2)
    matched = prob[..., :-1]
    mass = matched.sum(-1)
    expected = torch.einsum("...n,nd->...d", matched, coords) / mass.clamp_min(1e-8).unsqueeze(-1)
    return expected, mass, mass >= min_mass


def fit_background_affine(source: Tensor, target: Tensor, weight: Tensor,
                          iterations: int = 5, huber: float = 0.05) -> tuple[Tensor, Tensor]:
    """Batched fixed-iteration Huber IRLS affine fit in normalized coordinates."""
    source, target, weight = source.float(), target.float(), weight.float()
    prefix = source.shape[:-2]; n = source.shape[-2]
    x = torch.cat((source, torch.ones_like(source[..., :1])), -1)
    w = weight.clamp_min(0)
    valid = (w > 0).sum(-1) >= 12
    beta = torch.zeros(*prefix, 3, 2, device=source.device, dtype=torch.float32)
    eye = torch.eye(3, device=source.device).expand(*prefix, 3, 3)
    for _ in range(iterations):
        xtw = x.transpose(-1, -2) * w.unsqueeze(-2)
        gram = xtw @ x
        rhs = xtw @ target
        beta = torch.linalg.solve((gram + eye * 1e-5).float(), rhs.float())
        residual = torch.linalg.vector_norm(x @ beta - target, dim=-1)
        robust = torch.where(residual <= huber, torch.ones_like(residual), huber / residual.clamp_min(1e-8))
        w = weight * robust
    linear = beta[..., :2, :].transpose(-1, -2)
    offset = beta[..., 2, :].unsqueeze(-1)
    affine = torch.cat((linear, offset), -1)
    det = torch.linalg.det(linear)
    cond = torch.linalg.cond(((x.transpose(-1, -2) * w.unsqueeze(-2)) @ x + eye * 1e-5).float())
    valid = valid & torch.isfinite(cond) & (cond <= 1e6) & (det > 0)
    identity = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=source.device).expand_as(affine)
    return torch.where(valid[..., None, None], affine, identity), valid


class TrafficPrimitiveBuilder(nn.Module):
    def __init__(self, grid_hw: tuple[int, int] = (16, 28)) -> None:
        super().__init__(); self.grid_hw = grid_hw
        h, w = grid_hw
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        self.register_buffer("coords", torch.stack((xx, yy), -1).reshape(-1, 2), persistent=False)

    def forward(self, maps: Tensor, correspondence: dict[str, Tensor], actual_t: Tensor, frame_valid: Tensor) -> dict[str, Tensor]:
        maps=maps.float();correspondence={key:value.float() if torch.is_floating_point(value) else value for key,value in correspondence.items()}
        b, t = maps.shape[:2]
        pooled = F.adaptive_avg_pool2d(maps.flatten(0, 1), self.grid_hw).view(b, t, 8, -1)
        x = self.coords[:, 0]; y = self.coords[:, 1]
        regions = torch.stack(((x.abs() <= .34) & (y >= -.2), x < -.34, x > .34)).float()
        occ_v = torch.einsum("btpn,rn->btpr", pooled[:, :, :2], regions) / regions.sum(-1).clamp_min(1).view(1, 1, 1, -1)
        values = maps.new_zeros(b, t, 14, dtype=torch.float32)
        values[..., :3] = occ_v[:, :, 0]
        values[..., 3:6] = occ_v[:, :, 1]
        upper = (y < 0).float(); values[..., 9] = (pooled[:, :, 3] * upper).sum(-1) / upper.sum()
        values[..., 10] = (pooled[:, :, 4] * upper).sum(-1) / upper.sum()
        valid = frame_valid.unsqueeze(-1).expand_as(values).clone(); valid[:, 0, 6:9] = False; valid[:, 0, 11:] = False
        affines = maps.new_zeros(b, t - 1, 2, 3, dtype=torch.float32)
        bg_valid = torch.zeros(b, t - 1, dtype=torch.bool, device=maps.device)
        if t > 1:
            expected, mass, matched = matched_expectation(correspondence["forward"], self.grid_hw)
            reverse_expected, _, reverse_matched = matched_expectation(correspondence["reverse"], self.grid_hw)
            src = self.coords.expand(b, t - 1, -1, -1)
            reverse_map = reverse_expected.view(b * (t - 1), *self.grid_hw, 2).permute(0, 3, 1, 2)
            composed = F.grid_sample(reverse_map, expected.view(b * (t - 1), *self.grid_hw, 2),
                                     align_corners=True).permute(0, 2, 3, 1).reshape_as(expected)
            reverse_ok_map = reverse_matched.float().view(b * (t - 1), 1, *self.grid_hw)
            reverse_ok = F.grid_sample(reverse_ok_map, expected.view(b * (t - 1), *self.grid_hw, 2),
                                       mode="nearest", align_corners=True).view(b, t - 1, -1) > .5
            cycle_error = torch.linalg.vector_norm(composed - src, dim=-1)
            interior = expected.abs().amax(-1) < .999
            quality = matched & reverse_ok & (cycle_error <= .15) & interior
            foreground = pooled[:, :-1, :3].amax(2)
            texture = correspondence["grid_features"][:, :-1].float().var(-1).sqrt()
            weight = (1 - foreground).clamp_min(.05) * texture.detach() * mass.detach() * quality.float()
            affines, bg_valid = fit_background_affine(src, expected, weight)
            dt = (actual_t[:, 1:] - actual_t[:, :-1]).clamp_min(1e-4)
            delta = expected - torch.einsum("btij,btnj->btni", affines[..., :2], src) - affines[..., 2].unsqueeze(-2)
            # Occupancy-weighted image-plane motion proxies.
            radial = F.normalize(src, dim=-1, eps=1e-6); radial_mask = src.norm(dim=-1) >= .05
            vehicle = pooled[:, :-1, 0] * quality.float()
            values[:, 1:, 6] = ((delta * radial).sum(-1) * vehicle * radial_mask).sum(-1) / vehicle.sum(-1).clamp_min(1e-6) / dt
            values[:, 1:, 7] = (-delta[..., 0] * vehicle * (x < -.34)).sum(-1) / (vehicle * (x < -.34)).sum(-1).clamp_min(1e-6) / dt
            values[:, 1:, 8] = (delta[..., 0] * vehicle * (x > .34)).sum(-1) / (vehicle * (x > .34)).sum(-1).clamp_min(1e-6) / dt
            values[:, 1:, 11] = affines[..., 0, 2] / dt
            values[:, 1:, 12] = affines[..., 1, 2] / dt
            det = torch.linalg.det(affines[..., :2]); values[:, 1:, 13] = .5 * det.abs().clamp_min(1e-8).log() / dt
            motion_ok = quality.any(-1) & bg_valid & frame_valid[:, :-1] & frame_valid[:, 1:]
            valid[:, 1:, 6:9] &= motion_ok.unsqueeze(-1)
            valid[:, 1:, 11:] &= bg_valid.unsqueeze(-1)
        values[..., 6:9] = torch.tanh(values[..., 6:9] / .20)
        values[..., 11:13] = torch.tanh(values[..., 11:13] / .20)
        values[..., 13] = torch.tanh(values[..., 13] / .50)
        return {"primitive_value": values, "primitive_valid": valid,
                "primitive_support": valid.float(), "predicate_maps": maps,
                "background_affine": affines, "background_valid": bg_valid,
                "correspondence_quality_rate": quality.float().mean((-1, -2)) if t > 1 else maps.new_zeros(b)}
