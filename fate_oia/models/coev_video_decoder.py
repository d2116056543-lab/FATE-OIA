from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SpatialClassBlock(nn.Module):
    def __init__(self, dim: int = 384, heads: int = 6) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, q: Tensor, fields: Tensor) -> Tensor:
        q = q + self.cross(self.n1(q), fields, fields, need_weights=False)[0]
        q = q + self.self_attn(self.n2(q), self.n2(q), self.n2(q), need_weights=False)[0]
        return q + self.ff(q)


class CoEVVideoDecoder(nn.Module):
    def __init__(self, dim: int = 384, temporal_layers: int = 3, reason_bias_verified: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.class_queries = nn.Parameter(torch.randn(25, dim) * .02)
        self.frame_blocks = nn.ModuleList([SpatialClassBlock(dim) for _ in range(2)])
        layer = nn.TransformerEncoderLayer(dim, 6, dim * 4, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, temporal_layers)
        self.time_proj = nn.Linear(32, dim)
        self.layer_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.layer_id = nn.Parameter(torch.randn(3, dim) * .02)
        self.spatial_proj = nn.Linear(4, dim)
        self.class_interaction = nn.MultiheadAttention(dim, 6, batch_first=True)
        self.reread_q = nn.Linear(dim, dim); self.reread_k = nn.Linear(dim, dim); self.reread_v = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)
        support = torch.zeros(25, 8)
        support[0, [0, 1, 6]] = 1; support[1, [0, 1, 6]] = 1
        support[2, [0, 1, 7, 6]] = 1; support[3, [0, 1, 7, 6]] = 1
        if reason_bias_verified:
            support[4 + 0, 4] = 1; support[4 + 1, 0] = 1; support[4 + 2, 6] = 1
            support[4 + 3, [3, 4]] = 1; support[4 + 4, 5] = 1; support[4 + 5, 0] = 1
            support[4 + 6, 1] = 1; support[4 + 7, 2] = 1
            support[4 + 10, [0, 1]] = 1; support[4 + 16, [0, 1]] = 1
        self.register_buffer("class_fact_support", support)

    def _spatial(self, hw: tuple[int, int], device: torch.device, dtype: torch.dtype) -> Tensor:
        y,x=torch.meshgrid(torch.linspace(-1,1,hw[0],device=device,dtype=dtype),
                           torch.linspace(-1,1,hw[1],device=device,dtype=dtype),indexing="ij")
        return self.spatial_proj(torch.stack((x,y,torch.sin(math.pi*x),torch.sin(math.pi*y)),-1).reshape(-1,4))

    def class_spatial_support(self, hw: tuple[int,int], device: torch.device, dtype: torch.dtype) -> Tensor:
        y,x=torch.meshgrid(torch.linspace(-1,1,hw[0],device=device,dtype=dtype),
                           torch.linspace(-1,1,hw[1],device=device,dtype=dtype),indexing="ij")
        support=torch.ones(25,*hw,device=device,dtype=dtype)
        support[0]=support[1]=((x.abs()<=.45)&(y>=-.4)).to(dtype)
        support[2]=(x<=0).to(dtype);support[3]=(x>=0).to(dtype)
        return support

    def _decorate(self, field: Tensor, layer_index: int, hw: tuple[int,int], time: Tensor | None = None) -> Tensor:
        value=self.layer_proj[layer_index](field)+self.layer_id[layer_index]+self._spatial(hw,field.device,field.dtype)
        if time is not None:
            value=value+self.time_proj(self._time_features(time)).unsqueeze(1)
        return value

    def read_frame(self, task_fields: list[Tensor], grid_hw: tuple[int,int]) -> Tensor:
        b = task_fields[0].shape[0]
        q = self.class_queries.unsqueeze(0).expand(b, -1, -1)
        fields = torch.cat([self._decorate(field,index,grid_hw) for index,field in enumerate(task_fields)], 1)
        for block in self.frame_blocks:
            q = block(q, fields)
        return q

    def _time_features(self, t: Tensor) -> Tensor:
        freq = (2.0 ** torch.arange(16, device=t.device, dtype=t.dtype)) * math.pi / 5.0
        phase = t.unsqueeze(-1) * freq
        return torch.cat((phase.sin(), phase.cos()), -1)

    def _full_reread(self, q: Tensor, field_chunks: list[Tensor], bias_chunks: list[Tensor] | None,
                     valid_chunks: list[Tensor] | None = None) -> Tensor:
        keys = torch.cat([self.reread_k(x) for x in field_chunks], 1)
        values = torch.cat([self.reread_v(x) for x in field_chunks], 1)
        scores = torch.einsum("bld,bnd->bln", self.reread_q(q), keys) / math.sqrt(self.dim)
        if bias_chunks:
            scores = scores + torch.cat(bias_chunks, -1)
        if valid_chunks:
            key_valid = torch.cat(valid_chunks, -1)
            scores = scores.masked_fill(~key_valid.unsqueeze(1), torch.finfo(scores.dtype).min)
        return q + torch.einsum("bln,bnd->bld", scores.softmax(-1), values)

    def read_history(self, frame_queries: Tensor, actual_t: Tensor, valid: Tensor,
                     history_fields: list[Tensor], target_fields: list[Tensor],
                     history_hw: tuple[int,int], target_hw: tuple[int,int],
                     predicate_maps: Tensor | None = None) -> dict[str, Tensor | int]:
        b, t, labels, d = frame_queries.shape
        x = frame_queries + self.time_proj(self._time_features(actual_t)).unsqueeze(2)
        x = x.permute(0, 2, 1, 3).reshape(b * labels, t, d)
        padding = (~valid).unsqueeze(1).expand(b, labels, t).reshape(b * labels, t)
        x = self.temporal(x, src_key_padding_mask=padding)
        last = x[:, -1].view(b, labels, d)
        last = last + self.class_interaction(last, last, last, need_weights=False)[0]
        chunks = [self._decorate(history_fields[layer][:,i],layer,history_hw,actual_t[:,i])
                  for i in range(14) for layer in range(3)]
        chunks += [self._decorate(target_fields[layer],layer,target_hw,actual_t[:,-1]) for layer in range(3)]
        key_valid = [valid[:, i:i+1].expand(-1, history_fields[0].shape[2])
                     for i in range(14) for _ in history_fields]
        key_valid += [torch.ones(b, x.shape[1], dtype=torch.bool, device=x.device) for x in target_fields]
        bias_chunks = None
        if predicate_maps is not None:
            bias_chunks = []
            for i in range(15):
                for f in range(3):
                    n = chunks[i * 3 + f].shape[1]
                    support = torch.einsum("lp,bphw->blhw", self.class_fact_support, predicate_maps[:, i])
                    support = support * self.class_spatial_support(support.shape[-2:],support.device,support.dtype)
                    side = int(round(n ** .5)); hw = (32, 56) if n == 1792 else (45, 80)
                    support = F.interpolate(support, hw, mode="bilinear", align_corners=False).flatten(2)
                    bias_chunks.append(.5 * torch.log(.05 + support.clamp_min(0)))
        last = self._full_reread(last, chunks, bias_chunks, key_valid)
        last = self._full_reread(last, chunks[-3:], None)
        return {"q_video": self.out_norm(last), "full_history_kv_length": sum(x.shape[1] for x in chunks)}

    def forward(self, fields: dict[str, Tensor], actual_t: Tensor, valid: Tensor,
                predicate_maps: Tensor | None = None) -> dict[str, Tensor | int]:
        hfields = [fields[f"history_task{x}"] for x in (4, 8, 12)]
        tfields = [fields[f"target_task{x}"] for x in (4, 8, 12)]
        per_frame = []
        for i in range(14):
            per_frame.append(self.read_frame([x[:, i] for x in hfields],fields["history_grid_hw"]))
        per_frame.append(self.read_frame(tfields,fields["target_grid_hw"]))
        return self.read_history(torch.stack(per_frame, 1), actual_t, valid, hfields, tfields,
                                 fields["history_grid_hw"],fields["target_grid_hw"],predicate_maps)
