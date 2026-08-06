from __future__ import annotations

import math
import time

import torch
from torch import Tensor, nn

from .aie_deformable_reread import AIEDeformableReread


class AIEEvidenceInterface(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        probes_per_action: int = 4,
        num_layers: int = 3,
        num_predicates: int = 32,
        grid_hw: tuple[int, int] = (45, 80),
        local_points_per_layer: int = 8,
        max_offset: float = 0.25,
        predicate_bias_max: float = 0.25,
        probe_chunk_size: int = 16,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.probes_per_action = probes_per_action
        self.num_layers = num_layers
        self.predicate_bias_max = float(predicate_bias_max)
        self.probe_chunk_size = int(probe_chunk_size)
        self.role_embeddings = nn.Parameter(torch.empty(probes_per_action, dim))
        nn.init.orthogonal_(self.role_embeddings)
        with torch.no_grad():
            # Keep the four initialization roles visible beside a unit-variance action token.
            self.role_embeddings.mul_(0.50 * math.sqrt(dim))
        self.probe_query_norm = nn.LayerNorm(dim)
        self.layer_projections = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.layer_norms = nn.ModuleList(nn.RMSNorm(dim) for _ in range(num_layers))
        self.position_projection = nn.Linear(8, dim)
        self.query_projection = nn.Linear(dim, dim)
        self.key_projections = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.value_projections = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.layer_router = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_layers))
        self.predicate_keys = nn.Parameter(torch.randn(num_predicates, 64) * 0.02)
        self.probe_predicate_projection = nn.Linear(dim, 64)
        self.predicate_strength = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.deformable = AIEDeformableReread(
            dim=dim,
            grid_hw=grid_hw,
            num_layers=num_layers,
            points_per_layer=local_points_per_layer,
            max_offset=max_offset,
        )
        self.fusion_norm = nn.LayerNorm(dim)
        self.fusion_ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.group_attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.grid_hw = grid_hw

    def _position_features(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        h, w = self.grid_hw
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=device, dtype=dtype),
            torch.linspace(0, 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        distance = torch.sqrt((xx - 0.5).square() + (yy - 1.0).square())
        front = torch.exp(-(((xx - 0.5).square()) / 0.08 + ((yy - 0.75).square()) / 0.20))
        left = torch.sigmoid((0.45 - xx) * 10) * torch.sigmoid((yy - 0.35) * 10)
        right = torch.sigmoid((xx - 0.55) * 10) * torch.sigmoid((yy - 0.35) * 10)
        upper = torch.sigmoid((0.45 - yy) * 10)
        bottom = yy
        return torch.stack((xx, yy, distance, front, left, right, upper, bottom), -1).reshape(h * w, 8)

    def _condition(self, field: Tensor) -> Tensor:
        position = self.position_projection(self._position_features(field.device, field.dtype))
        layers = [norm(proj(field[:, i])) + position for i, (proj, norm) in enumerate(zip(self.layer_projections, self.layer_norms))]
        return torch.stack(layers, dim=1)

    def forward(
        self,
        action_nodes_primary: Tensor,
        patch_tokens_by_layer: Tensor,
        predicate_attention: Tensor,
        predicate_probs: Tensor,
        *,
        predicate_bias_enabled: bool = True,
        local_reread_enabled: bool = True,
        group_attention_enabled: bool = True,
        profile: bool = False,
    ) -> dict[str, Tensor]:
        def stamp() -> float:
            if profile and patch_tokens_by_layer.is_cuda:
                torch.cuda.synchronize(patch_tokens_by_layer.device)
            return time.perf_counter()

        global_start = stamp()
        b, layers, patches, _ = patch_tokens_by_layer.shape
        if layers != self.num_layers:
            raise ValueError("AIE evidence interface received the wrong number of DINO layers")
        probes = self.probe_query_norm(action_nodes_primary.detach()[:, :, None, :] + self.role_embeddings[None, None, :, :])
        # The final branch owns no path into the primary ego encoder.
        conditioned = self._condition(patch_tokens_by_layer.detach())
        q = self.query_projection(probes)
        layer_mix = torch.softmax(self.layer_router(probes), dim=-1)
        scores, values, layer_attentions = [], [], []
        for layer in range(layers):
            key = self.key_projections[layer](conditioned[:, layer])
            value = self.value_projections[layer](conditioned[:, layer])
            flat_query = q.reshape(b, self.action_dim * self.probes_per_action, self.dim)
            score = torch.cat(
                [
                    torch.einsum("bqd,bnd->bqn", flat_query[:, start : start + self.probe_chunk_size], key)
                    for start in range(0, flat_query.shape[1], self.probe_chunk_size)
                ],
                dim=1,
            ).reshape(b, self.action_dim, self.probes_per_action, patches) / math.sqrt(self.dim)
            scores.append(score)
            values.append(value)
            layer_attentions.append(torch.softmax(score, dim=-1))
        score_stack = torch.stack(scores, dim=3)
        attention_stack = torch.stack(layer_attentions, dim=3)
        global_attention = (layer_mix[..., None] * attention_stack).sum(3)
        global_token = sum(
            torch.einsum("bakn,bnd->bakd", layer_mix[..., layer, None] * attention_stack[:, :, :, layer], values[layer])
            for layer in range(layers)
        )
        pred_attn = predicate_attention.detach().clamp_min(1e-8)
        pred_prob = predicate_probs.detach().clamp(0, 1)
        compatibility = torch.sigmoid(
            torch.einsum("bakd,pd->bakp", self.probe_predicate_projection(probes), self.predicate_keys)
        )
        weighted_compat = compatibility * pred_prob[:, None, None, :]
        normalized_compat = weighted_compat / weighted_compat.sum(-1, keepdim=True).clamp_min(1e-8)
        predicate_log_prior = torch.einsum("bakp,bpn->bakn", normalized_compat, pred_attn.log())
        strength = self.predicate_bias_max * torch.sigmoid(self.predicate_strength(global_token)).squeeze(-1)
        visual_score = (layer_mix[..., None] * score_stack).sum(3)
        combined_score = visual_score + (strength[..., None] * predicate_log_prior if predicate_bias_enabled else 0.0)
        evidence_map = torch.softmax(combined_score, dim=-1)
        global_end = stamp()
        local_out = self.deformable(probes, conditioned, evidence_map)
        local_end = stamp()
        local = local_out["local_token"] if local_reread_enabled else torch.zeros_like(global_token)
        fused = global_token + local
        fused = self.fusion_norm(fused + self.fusion_ffn(fused))
        grouped = fused.reshape(b * self.action_dim, self.probes_per_action, self.dim)
        if group_attention_enabled:
            grouped = grouped + self.group_attention(grouped, grouped, grouped, need_weights=False)[0]
        evidence = grouped.reshape(b, self.action_dim, self.probes_per_action, self.dim)
        result = {
            "probe_queries": probes,
            "conditioned_patch_layers": conditioned,
            "global_attention": global_attention,
            "global_token": global_token,
            "layer_mixture": layer_mix,
            "evidence_token": evidence,
            "evidence_map": evidence_map,
            "reference_point": local_out["reference_point"],
            "sampling_offsets": local_out["sampling_offsets"],
            "sampling_weights": local_out["sampling_weights"],
            "predicate_compatibility": normalized_compat,
            "predicate_compatibility_raw": compatibility,
            "predicate_bias_strength": strength if predicate_bias_enabled else torch.zeros_like(strength),
        }
        if profile:
            result["_profile_evidence_global_time"] = global_end - global_start
            result["_profile_evidence_local_time"] = local_end - global_end
        return result
