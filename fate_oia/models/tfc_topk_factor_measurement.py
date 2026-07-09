from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def build_region_priors(spatial_names: list[str], grid_hw: tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    h, w = grid_hw
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h, device=device, dtype=dtype),
        torch.linspace(0, 1, w, device=device, dtype=dtype),
        indexing="ij",
    )
    priors = []
    for name in spatial_names:
        if name in {"front_center", "center_corridor"}:
            mask = torch.exp(-(((xx - 0.5) ** 2) / 0.06 + ((yy - 0.72) ** 2) / 0.18))
        elif name in {"upper_front"}:
            mask = torch.sigmoid((0.45 - yy) * 10.0) * torch.exp(-((xx - 0.5) ** 2) / 0.12)
        elif name in {"left_corridor"}:
            mask = torch.sigmoid((0.45 - xx) * 10.0) * torch.sigmoid((yy - 0.35) * 10.0)
        elif name in {"right_corridor"}:
            mask = torch.sigmoid((xx - 0.55) * 10.0) * torch.sigmoid((yy - 0.35) * 10.0)
        else:
            mask = torch.ones_like(xx)
        mask = mask.flatten()
        priors.append(mask / mask.max().clamp_min(1e-6))
    return torch.stack(priors, dim=0)


class TFCTopKFactorMeasurement(nn.Module):
    def __init__(self, dim: int = 384, topk: int = 64, grid_hw: tuple[int, int] = (45, 80)) -> None:
        super().__init__()
        self.dim = int(dim)
        self.topk = int(topk)
        self.grid_hw = grid_hw
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.logit_head = nn.Linear(dim, 1)
        self.rho_head = nn.Linear(dim, 1)
        self.region_bias_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not bool(mask.any()):
            return values.new_tensor(0.0)
        return values[:, mask].mean()

    def forward(
        self,
        patch_tokens_by_layer: torch.Tensor,
        factor_queries: torch.Tensor,
        spatial_names: list[str],
        factor_names: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        b, layers, n, d = patch_tokens_by_layer.shape
        f = factor_queries.shape[0]
        k = min(self.topk, layers * n)
        keys = F.normalize(self.key_proj(patch_tokens_by_layer), dim=-1)
        queries = F.normalize(factor_queries, dim=-1)
        scores = torch.einsum("fd,blnd->bfln", queries, keys) / (d ** 0.5)
        priors = build_region_priors(spatial_names, self.grid_hw, patch_tokens_by_layer.device, patch_tokens_by_layer.dtype)
        scores = scores + self.region_bias_scale * priors.view(1, f, 1, n)
        scores_flat = scores.flatten(2)
        top_scores, top_idx = scores_flat.topk(k=k, dim=-1)
        values_flat = self.value_proj(patch_tokens_by_layer).reshape(b, layers * n, d)
        top_values = []
        for factor_idx in range(f):
            idx = top_idx[:, factor_idx].unsqueeze(-1).expand(b, k, d)
            top_values.append(values_flat.gather(1, idx))
        top_values_t = torch.stack(top_values, dim=1)
        weights = torch.softmax(top_scores, dim=-1)
        factor_features = (weights.unsqueeze(-1) * top_values_t).sum(dim=2)
        factor_logits = self.logit_head(factor_features).squeeze(-1)
        factor_probs = torch.sigmoid(factor_logits)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1)
        rho_base = self.rho_head(factor_features).squeeze(-1)
        factor_rho = torch.sigmoid(rho_base - entropy / max(float(k), 1.0))
        top_region = priors.view(1, f, 1, n).expand(b, f, layers, n).flatten(2).gather(2, top_idx)
        names = factor_names or ["" for _ in range(f)]
        name_l = [name.lower() for name in names]
        spatial_l = [name.lower() for name in spatial_names]
        device = patch_tokens_by_layer.device
        traffic_mask = torch.tensor(
            [("traffic_light" in name or "traffic_sign" in name) for name in name_l],
            device=device,
            dtype=torch.bool,
        )
        obstacle_mask = torch.tensor(
            [("vehicle" in name or "pedestrian" in name or "rider" in name or "obstacle" in name) and "front" in spatial_l[i] for i, name in enumerate(name_l)],
            device=device,
            dtype=torch.bool,
        )
        left_mask = torch.tensor(["left_corridor" in spatial_l[i] for i in range(f)], device=device, dtype=torch.bool)
        right_mask = torch.tensor(["right_corridor" in spatial_l[i] for i in range(f)], device=device, dtype=torch.bool)
        upper_mask = torch.tensor(["upper" in spatial_l[i] for i in range(f)], device=device, dtype=torch.bool)
        traffic_upper_mask = traffic_mask & upper_mask
        red_idx = next((i for i, name in enumerate(name_l) if "traffic_light_red" in name), None)
        green_idx = next((i for i, name in enumerate(name_l) if "traffic_light_green" in name), None)
        if red_idx is not None and green_idx is not None:
            red_green_contradiction = (factor_probs[:, red_idx] * factor_probs[:, green_idx]).mean()
        else:
            red_green_contradiction = factor_probs.new_tensor(0.0)
        grounding_audit_stats = {
            "traffic_control_upper_region_mass": self._masked_mean(top_region, traffic_upper_mask),
            "obstacle_front_center_mass": self._masked_mean(top_region, obstacle_mask),
            "left_lane_corridor_mass": self._masked_mean(top_region, left_mask),
            "right_lane_corridor_mass": self._masked_mean(top_region, right_mask),
            "prototype_assignment_entropy": entropy.mean(),
            "red_green_contradiction": red_green_contradiction,
            "left_right_mirror_mass_gap": (self._masked_mean(top_region, left_mask) - self._masked_mean(top_region, right_mask)).abs(),
        }
        random_indices = []
        for factor_idx in range(f):
            prob = priors[factor_idx].view(1, n).expand(layers, n).reshape(layers * n).clamp_min(1e-6)
            prob = prob / prob.sum()
            # Sample without replacement so selected and random deletion remove
            # the same number of unique tokens. Duplicates would be collapsed by
            # deletion-time unique(), silently weakening the random baseline.
            samples = [torch.multinomial(prob, num_samples=k, replacement=False) for _ in range(b)]
            random_indices.append(torch.stack(samples, dim=0))
        random_indices_t = torch.stack(random_indices, dim=1)
        return {
            "factor_features": factor_features,
            "factor_logits": factor_logits,
            "factor_probs": factor_probs,
            "factor_rho": factor_rho,
            "topk_indices": top_idx,
            "topk_scores": top_scores,
            "random_indices": random_indices_t,
            "attention_entropy": entropy,
            "region_mass": top_region.mean(-1),
            "grounding_audit_stats": grounding_audit_stats,
        }

