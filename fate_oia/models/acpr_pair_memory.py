from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PairMiningThresholds:
    action_sim_min: float = 0.35
    visual_sim_min: float = 0.05
    predicate_sim_min: float = 0.05
    contradiction_min: float = 0.15
    tail_action_sim_min: float = 0.20
    tail_visual_sim_min: float = -0.05
    tail_predicate_sim_min: float = -0.05
    tail_contradiction_min: float = 0.05
    semi_hard_band: float = 0.15
    fallback_easy_pair_weight: float = 0.15


class ACPRPairMemory(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        memory_size: int = 8192,
        tail_multiplier: float = 2.0,
        memory_device: str = "cpu",
    ) -> None:
        super().__init__()
        self.memory_size = int(memory_size)
        self.tail_multiplier = float(tail_multiplier)
        self.memory_device = str(memory_device)
        self.proj = nn.Linear(dim, dim)
        self._memory_tensors: dict[str, torch.Tensor] = {}
        self._file_names: list[str] = ["" for _ in range(self.memory_size)]
        self._write_idx = 0
        self._count = 0
        self._reason_dim = 0
        self._embed_dim = int(dim)

    @property
    def memory_count(self) -> int:
        return int(self._count)

    def _storage_device(self, reference: torch.Tensor) -> torch.device:
        requested = self.memory_device.lower()
        if requested in {"same", "input", "auto"}:
            return reference.device
        if requested.startswith("cuda"):
            if reference.is_cuda:
                return reference.device
            if torch.cuda.is_available():
                return torch.device(requested)
            return torch.device("cpu")
        return torch.device("cpu")

    def _ensure_memory(
        self,
        *,
        global_embed: torch.Tensor,
        predicate_probs: torch.Tensor,
        action_targets: torch.Tensor,
        reason_targets: torch.Tensor,
        reason_embeddings_detached: torch.Tensor,
    ) -> None:
        storage_device = self._storage_device(global_embed)
        reason_dim = int(reason_targets.shape[1])
        embed_dim = int(global_embed.shape[-1])
        predicate_dim = int(predicate_probs.shape[-1])
        action_dim = int(action_targets.shape[-1])
        need_alloc = (
            not self._memory_tensors
            or self._memory_tensors["global_embed"].device != storage_device
            or self._memory_tensors["global_embed"].shape[1] != embed_dim
            or self._memory_tensors["predicate_probs"].shape[1] != predicate_dim
            or self._memory_tensors["action_targets"].shape[1] != action_dim
            or self._memory_tensors["reason_targets"].shape[1] != reason_dim
        )
        if not need_alloc:
            return
        self._reason_dim = reason_dim
        self._embed_dim = embed_dim
        self._write_idx = 0
        self._count = 0
        self._file_names = ["" for _ in range(self.memory_size)]
        self._memory_tensors = {
            "global_embed": torch.empty(self.memory_size, embed_dim, device=storage_device, dtype=global_embed.dtype),
            "predicate_probs": torch.empty(self.memory_size, predicate_dim, device=storage_device, dtype=predicate_probs.dtype),
            "action_targets": torch.empty(self.memory_size, action_dim, device=storage_device, dtype=action_targets.dtype),
            "reason_targets": torch.empty(self.memory_size, reason_dim, device=storage_device, dtype=reason_targets.dtype),
            "contradiction_scores": torch.empty(self.memory_size, reason_dim, device=storage_device, dtype=reason_targets.dtype),
            "reason_logits_detached": torch.empty(self.memory_size, reason_dim, device=storage_device, dtype=reason_targets.dtype),
            "reason_embeddings_detached": torch.empty(
                self.memory_size,
                reason_dim,
                int(reason_embeddings_detached.shape[-1]),
                device=storage_device,
                dtype=reason_embeddings_detached.dtype,
            ),
        }

    def _ordered_indices(self, device: torch.device | None = None) -> torch.Tensor:
        if self._count <= 0:
            return torch.empty(0, dtype=torch.long, device=device or torch.device("cpu"))
        storage_device = next(iter(self._memory_tensors.values())).device
        idx_device = device or storage_device
        start = (self._write_idx - self._count) % self.memory_size
        return (torch.arange(self._count, device=idx_device, dtype=torch.long) + start) % self.memory_size

    def memory_view(self, device: torch.device | None = None) -> dict[str, torch.Tensor | list[str]]:
        if self._count <= 0 or not self._memory_tensors:
            return {}
        tensor_device = next(iter(self._memory_tensors.values())).device
        indices_on_storage = self._ordered_indices(tensor_device)
        out: dict[str, torch.Tensor | list[str]] = {}
        for key, tensor in self._memory_tensors.items():
            value = tensor.index_select(0, indices_on_storage)
            out[key] = value if device is None or value.device == device else value.to(device, non_blocking=True)
        indices_cpu = indices_on_storage.detach().cpu().tolist()
        out["file_names"] = [self._file_names[int(i)] for i in indices_cpu]
        return out

    def forward(self, label_nodes: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(label_nodes.mean(1)), dim=-1)

    @torch.no_grad()
    def enqueue(
        self,
        file_names: list[str],
        global_embed: torch.Tensor,
        predicate_probs: torch.Tensor,
        action_targets: torch.Tensor,
        reason_targets: torch.Tensor,
        contradiction_scores: torch.Tensor | None = None,
        reason_logits_detached: torch.Tensor | None = None,
        reason_embeddings_detached: torch.Tensor | None = None,
    ) -> None:
        b = int(action_targets.shape[0])
        reason_dim = int(reason_targets.shape[1])
        device = action_targets.device
        if contradiction_scores is None:
            contradiction_scores = torch.zeros(b, reason_dim, device=device, dtype=reason_targets.dtype)
        if reason_logits_detached is None:
            reason_logits_detached = torch.zeros(b, reason_dim, device=device, dtype=reason_targets.dtype)
        if reason_embeddings_detached is None:
            d = int(global_embed.shape[-1])
            reason_embeddings_detached = torch.zeros(b, reason_dim, d, device=device, dtype=global_embed.dtype)
        # fixed-capacity ring buffer: keep historical negatives on the requested
        # memory device and overwrite in-place instead of rebuilding history.
        self._ensure_memory(
            global_embed=global_embed,
            predicate_probs=predicate_probs,
            action_targets=action_targets,
            reason_targets=reason_targets,
            reason_embeddings_detached=reason_embeddings_detached,
        )
        storage_device = next(iter(self._memory_tensors.values())).device
        payload = {
            "global_embed": F.normalize(global_embed.detach(), dim=-1).to(storage_device, non_blocking=True),
            "predicate_probs": predicate_probs.detach().to(storage_device, non_blocking=True),
            "action_targets": action_targets.detach().to(storage_device, non_blocking=True),
            "reason_targets": reason_targets.detach().to(storage_device, non_blocking=True),
            "contradiction_scores": contradiction_scores.detach().to(storage_device, non_blocking=True),
            "reason_logits_detached": reason_logits_detached.detach().to(storage_device, non_blocking=True),
            "reason_embeddings_detached": F.normalize(reason_embeddings_detached.detach(), dim=-1).to(storage_device, non_blocking=True),
        }
        for row in range(b):
            slot = self._write_idx
            for key, value in payload.items():
                self._memory_tensors[key][slot].copy_(value[row], non_blocking=True)
            self._file_names[slot] = file_names[row]
            self._write_idx = (self._write_idx + 1) % self.memory_size
            self._count = min(self._count + 1, self.memory_size)

    @staticmethod
    def _empty(device: torch.device, reason_dim: int = 21, embed_dim: int = 384) -> dict[str, torch.Tensor | int | list[float] | list[int] | list[str]]:
        empty_l = torch.empty(0, dtype=torch.long, device=device)
        empty_f = torch.empty(0, dtype=torch.float32, device=device)
        empty_b = torch.empty(0, dtype=torch.bool, device=device)
        return {
            "pair_pos_indices": empty_l,
            "pair_neg_indices": empty_l,
            "pair_neg_memory_indices": empty_l,
            "pair_reason_ids": empty_l,
            "pair_weights": empty_f,
            "pair_neg_logits_detached": empty_f,
            "pair_neg_embedding_detached": torch.empty(0, embed_dim, dtype=torch.float32, device=device),
            "pair_neg_is_memory": empty_b,
            "pair_action_sim": empty_f,
            "pair_visual_sim": empty_f,
            "pair_predicate_sim": empty_f,
            "pair_contradiction": empty_f,
            "pair_hinge_raw": empty_f,
            "pair_hard_mask": empty_b,
            "pair_semi_hard_mask": empty_b,
            "pair_easy_mask": empty_b,
            "pair_active_mask": empty_b,
            "positive_pairs": torch.empty(0, 2, dtype=torch.long, device=device),
            "contrast_pairs": torch.empty(0, 2, dtype=torch.long, device=device),
            "pair_count": 0,
            "active_pair_count": 0,
            "hard_pair_count": 0,
            "semi_hard_pair_count": 0,
            "easy_pair_count": 0,
            "zero_loss_pair_count": 0,
            "margin_satisfied_count": 0,
            "tail_pair_count": 0,
            "tail_active_pair_count": 0,
            "pair_memory_count": 0,
            "pair_no_candidate_count": 0,
            "pair_gate_filtered_count": 0,
            "pair_count_per_reason": [0 for _ in range(reason_dim)],
            "active_pair_count_per_reason": [0 for _ in range(reason_dim)],
            "hard_pair_count_per_reason": [0 for _ in range(reason_dim)],
            "semi_hard_pair_count_per_reason": [0 for _ in range(reason_dim)],
            "easy_pair_count_per_reason": [0 for _ in range(reason_dim)],
            "margin_mean_per_reason": [0.0 for _ in range(reason_dim)],
            "active_margin_mean_per_reason": [0.0 for _ in range(reason_dim)],
            "file_names": [],
        }

    def mine(
        self,
        batch_file_names: list[str],
        batch_global_embed: torch.Tensor,
        batch_predicate_probs: torch.Tensor,
        batch_action_targets: torch.Tensor,
        batch_reason_targets: torch.Tensor,
        contradiction_score: torch.Tensor,
        tail_indices: list[int] | None = None,
        reason_logits_current: torch.Tensor | None = None,
        reason_embeddings_current: torch.Tensor | None = None,
        epoch: int = 0,
        max_pairs: int | None = None,
        max_pairs_per_reason: int = 8,
        max_tail_pairs_per_reason: int = 12,
        max_memory_scan: int = 2048,
        margin: float = 0.25,
        thresholds: PairMiningThresholds | None = None,
    ) -> dict[str, torch.Tensor | int | list[float] | list[int] | list[str]]:
        thresholds = thresholds or PairMiningThresholds()
        if reason_logits_current is None:
            reason_logits_current = torch.zeros_like(batch_reason_targets)
        if reason_embeddings_current is None:
            d = int(batch_global_embed.shape[-1])
            reason_embeddings_current = torch.zeros(batch_reason_targets.shape[0], batch_reason_targets.shape[1], d, device=batch_reason_targets.device, dtype=batch_global_embed.dtype)
        return self.mine_pairs(
            batch_global_embed,
            batch_action_targets,
            batch_reason_targets,
            tail_indices,
            global_embedding=batch_global_embed,
            predicate_probs=batch_predicate_probs,
            contradiction_scores=contradiction_score,
            reason_logits_current=reason_logits_current,
            reason_embeddings_current=reason_embeddings_current,
            file_names=batch_file_names,
            max_pairs=max_pairs,
            max_pairs_per_reason=max_pairs_per_reason,
            max_tail_pairs_per_reason=max_tail_pairs_per_reason,
            max_memory_scan=max_memory_scan,
            margin=margin,
            thresholds=thresholds,
        )

    def mine_pairs(
        self,
        embeddings: torch.Tensor,
        action: torch.Tensor,
        reason: torch.Tensor,
        tail_indices: list[int] | None = None,
        global_embedding: torch.Tensor | None = None,
        predicate_probs: torch.Tensor | None = None,
        contradiction_scores: torch.Tensor | None = None,
        reason_logits_current: torch.Tensor | None = None,
        reason_embeddings_current: torch.Tensor | None = None,
        file_names: list[str] | None = None,
        max_pairs: int | None = None,
        max_pairs_per_reason: int = 8,
        max_tail_pairs_per_reason: int = 12,
        max_memory_scan: int = 2048,
        margin: float = 0.25,
        thresholds: PairMiningThresholds | None = None,
    ) -> dict[str, torch.Tensor | int | list[float] | list[int] | list[str]]:
        thresholds = thresholds or PairMiningThresholds()
        b, reason_dim = int(reason.shape[0]), int(reason.shape[1])
        device = reason.device
        embed_dim = int((reason_embeddings_current.shape[-1] if reason_embeddings_current is not None else embeddings.shape[-1]))
        if b < 1:
            return self._empty(device, reason_dim, embed_dim)
        max_pairs = int(max_pairs or min(self.memory_size, 256))
        emb = F.normalize(global_embedding if global_embedding is not None else embeddings, dim=-1)
        pred = F.normalize(predicate_probs if predicate_probs is not None else torch.zeros(b, 1, device=device), dim=-1)
        contradiction_scores = torch.zeros_like(reason) if contradiction_scores is None else contradiction_scores
        reason_logits_current = torch.zeros_like(reason) if reason_logits_current is None else reason_logits_current.detach()
        if reason_embeddings_current is None:
            reason_embeddings_current = torch.zeros(b, reason_dim, embed_dim, device=device, dtype=emb.dtype)
        reason_embeddings_current = F.normalize(reason_embeddings_current.detach(), dim=-1)
        action_sim_batch = F.cosine_similarity(action[:, None, :], action[None, :, :], dim=-1).clamp(-1, 1)
        visual_sim_batch = emb @ emb.t()
        predicate_sim_batch = pred @ pred.t()

        mem = self.memory_view(device)
        has_mem = bool(mem)
        if has_mem:
            mem_embed = mem["global_embed"].to(device)  # type: ignore[union-attr]
            mem_pred = mem["predicate_probs"].to(device)  # type: ignore[union-attr]
            mem_action = mem["action_targets"].to(device)  # type: ignore[union-attr]
            mem_reason = mem["reason_targets"].to(device)  # type: ignore[union-attr]
            mem_contra = mem["contradiction_scores"].to(device)  # type: ignore[union-attr]
            mem_logits = mem["reason_logits_detached"].to(device)  # type: ignore[union-attr]
            mem_reason_emb = mem["reason_embeddings_detached"].to(device)  # type: ignore[union-attr]
            mem_files = list(mem.get("file_names", []))  # type: ignore[arg-type]
            action_sim_mem = F.cosine_similarity(action[:, None, :], mem_action[None, :, :], dim=-1).clamp(-1, 1)
            visual_sim_mem = emb @ F.normalize(mem_embed, dim=-1).t()
            predicate_sim_mem = pred @ F.normalize(mem_pred, dim=-1).t()
        else:
            mem_reason = mem_logits = mem_reason_emb = mem_contra = None
            mem_files: list[str] = []
            action_sim_mem = visual_sim_mem = predicate_sim_mem = None

        tail = set(int(x) for x in (tail_indices or []))
        files = file_names or [str(i) for i in range(b)]
        candidate_chunks: list[dict[str, torch.Tensor]] = []
        no_candidate_count = 0
        gate_filtered_count = 0

        def thresholds_for(r: int) -> tuple[float, float, float, float]:
            if r in tail:
                return (
                    thresholds.tail_action_sim_min,
                    thresholds.tail_visual_sim_min,
                    thresholds.tail_predicate_sim_min,
                    thresholds.tail_contradiction_min,
                )
            return (
                thresholds.action_sim_min,
                thresholds.visual_sim_min,
                thresholds.predicate_sim_min,
                thresholds.contradiction_min,
            )

        def same_file_mask(pos_idx: torch.Tensor, candidate_files: list[str]) -> torch.Tensor:
            rows = []
            pos_cpu = pos_idx.detach().cpu().tolist()
            for pi in pos_cpu:
                rows.append([candidate_files[j] == files[int(pi)] for j in range(len(candidate_files))])
            if not rows:
                return torch.zeros(pos_idx.numel(), len(candidate_files), dtype=torch.bool, device=device)
            return torch.tensor(rows, dtype=torch.bool, device=device)

        def build_candidates(
            r: int,
            pos_idx: torch.Tensor,
            neg_idx: torch.Tensor,
            a_scores: torch.Tensor,
            v_scores: torch.Tensor,
            p_scores: torch.Tensor,
            c_scores: torch.Tensor,
            z_neg: torch.Tensor,
            candidate_files: list[str],
            *,
            is_memory: bool,
        ) -> dict[str, torch.Tensor] | None:
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                return None
            same = same_file_mask(pos_idx, candidate_files)
            z_pos = reason_logits_current[pos_idx, r].detach().unsqueeze(1)
            hinge = float(margin) - z_pos + z_neg.detach().unsqueeze(0)
            eligible = ~same
            if not eligible.any():
                return None
            a_min, v_min, p_min, c_min = thresholds_for(r)
            gate_pass = (
                eligible
                & (a_scores >= a_min)
                & (v_scores >= v_min)
                & (p_scores >= p_min)
                & (c_scores >= c_min)
            )
            hard = gate_pass & (hinge > 0.0)
            semi = gate_pass & (hinge >= -thresholds.semi_hard_band) & (hinge <= 0.0)
            easy = gate_pass & (hinge < -thresholds.semi_hard_band)
            select_mask = hard | semi
            if not select_mask.any() and r in tail and easy.any():
                select_mask = easy
            if not select_mask.any() and r in tail:
                rejected = eligible & ~gate_pass
                if rejected.any():
                    select_mask = rejected
            if not select_mask.any():
                return {
                    "empty": torch.tensor([1], dtype=torch.long, device=device),
                    "filtered": (eligible & ~gate_pass).sum().view(1),
                }

            score = hinge.masked_fill(~select_mask, -torch.inf).flatten()
            per_reason_limit = max_tail_pairs_per_reason if r in tail else max_pairs_per_reason
            k = min(int(per_reason_limit), int(torch.isfinite(score).sum().item()))
            if k <= 0:
                return None
            flat_idx = torch.topk(score, k=k, largest=True).indices
            n = int(neg_idx.numel())
            pos_local = torch.div(flat_idx, n, rounding_mode="floor")
            neg_local = flat_idx.remainder(n)
            pos_t = pos_idx[pos_local]
            raw_neg_t = neg_idx[neg_local]
            rid_t = torch.full((k,), int(r), dtype=torch.long, device=device)
            is_mem_t = torch.full((k,), bool(is_memory), dtype=torch.bool, device=device)
            neg_t = torch.full((k,), -1, dtype=torch.long, device=device) if is_memory else raw_neg_t.long()
            mem_t = raw_neg_t.long() if is_memory else torch.full((k,), -1, dtype=torch.long, device=device)
            return {
                "pos": pos_t.long(),
                "neg": neg_t.long(),
                "mem": mem_t.long(),
                "rid": rid_t,
                "is_memory": is_mem_t,
                "a": a_scores[pos_local, neg_local].float(),
                "v": v_scores[pos_local, neg_local].float(),
                "p": p_scores[pos_local, neg_local].float(),
                "c": c_scores[pos_local, neg_local].clamp(0, 1).float(),
                "hinge": hinge[pos_local, neg_local].float(),
                "filtered": (eligible & ~gate_pass).sum().view(1),
            }

        for r in range(reason_dim):
            pos_idx = torch.where(reason[:, r] > 0.5)[0]
            if pos_idx.numel() == 0:
                continue
            chunks_before = len(candidate_chunks)
            # In-batch candidates are tiny but still selected through the same tensor path.
            neg_idx = torch.where(reason[:, r] <= 0.5)[0]
            if neg_idx.numel():
                batch_chunk = build_candidates(
                    r,
                    pos_idx,
                    neg_idx,
                    action_sim_batch[pos_idx][:, neg_idx],
                    visual_sim_batch[pos_idx][:, neg_idx],
                    predicate_sim_batch[pos_idx][:, neg_idx],
                    contradiction_scores[neg_idx, r].detach().clamp(0, 1).unsqueeze(0).expand(pos_idx.numel(), neg_idx.numel()),
                    reason_logits_current[neg_idx, r],
                    [files[int(i)] for i in neg_idx.detach().cpu().tolist()],
                    is_memory=False,
                )
                if batch_chunk is not None:
                    gate_filtered_count += int(batch_chunk.get("filtered", torch.zeros(1, device=device)).sum().item())
                    if "pos" in batch_chunk:
                        candidate_chunks.append(batch_chunk)
            if has_mem and mem_reason is not None and mem_logits is not None:
                mem_neg_idx = torch.where(mem_reason[:, r] <= 0.5)[0]
                if mem_neg_idx.numel():
                    # Recent detached negatives are enough for hard-pair mining and avoid
                    # scanning an ever-growing queue in Python.
                    scan = max(1, int(max_memory_scan))
                    if mem_neg_idx.numel() > scan:
                        mem_neg_idx = mem_neg_idx[-scan:]
                    mem_chunk = build_candidates(
                        r,
                        pos_idx,
                        mem_neg_idx,
                        action_sim_mem[pos_idx][:, mem_neg_idx],  # type: ignore[index]
                        visual_sim_mem[pos_idx][:, mem_neg_idx],  # type: ignore[index]
                        predicate_sim_mem[pos_idx][:, mem_neg_idx],  # type: ignore[index]
                        mem_contra[mem_neg_idx, r].detach().clamp(0, 1).unsqueeze(0).expand(pos_idx.numel(), mem_neg_idx.numel()),  # type: ignore[index]
                        mem_logits[mem_neg_idx, r],  # type: ignore[index]
                        [mem_files[int(i)] for i in mem_neg_idx.detach().cpu().tolist()],
                        is_memory=True,
                    )
                    if mem_chunk is not None:
                        gate_filtered_count += int(mem_chunk.get("filtered", torch.zeros(1, device=device)).sum().item())
                        if "pos" in mem_chunk:
                            candidate_chunks.append(mem_chunk)
            if len(candidate_chunks) == chunks_before:
                no_candidate_count += int(pos_idx.numel())
            if sum(int(x["pos"].numel()) for x in candidate_chunks if "pos" in x) >= max_pairs:
                break

        if not candidate_chunks:
            empty = self._empty(device, reason_dim, embed_dim)
            empty["pair_no_candidate_count"] = no_candidate_count
            empty["pair_gate_filtered_count"] = gate_filtered_count
            empty["file_names"] = files
            return empty

        pos_t = torch.cat([x["pos"] for x in candidate_chunks if "pos" in x]).long()
        neg_t = torch.cat([x["neg"] for x in candidate_chunks if "pos" in x]).long()
        mem_t = torch.cat([x["mem"] for x in candidate_chunks if "pos" in x]).long()
        rid_t = torch.cat([x["rid"] for x in candidate_chunks if "pos" in x]).long()
        is_mem = torch.cat([x["is_memory"] for x in candidate_chunks if "pos" in x]).bool()
        action_t = torch.cat([x["a"] for x in candidate_chunks if "pos" in x]).float()
        visual_t = torch.cat([x["v"] for x in candidate_chunks if "pos" in x]).float()
        pred_t = torch.cat([x["p"] for x in candidate_chunks if "pos" in x]).float()
        contra_t = torch.cat([x["c"] for x in candidate_chunks if "pos" in x]).float()
        hinge_t = torch.cat([x["hinge"] for x in candidate_chunks if "pos" in x]).float()
        if hinge_t.numel() > max_pairs:
            order_score = hinge_t + (hinge_t > 0).float() * 1000.0
            order = torch.topk(order_score, k=max_pairs, largest=True).indices
            pos_t, neg_t, mem_t, rid_t, is_mem = pos_t[order], neg_t[order], mem_t[order], rid_t[order], is_mem[order]
            action_t, visual_t, pred_t, contra_t, hinge_t = action_t[order], visual_t[order], pred_t[order], contra_t[order], hinge_t[order]

        hard_mask = hinge_t > 0.0
        semi_mask = (hinge_t >= -thresholds.semi_hard_band) & (hinge_t <= 0.0)
        easy_mask = hinge_t < -thresholds.semi_hard_band
        tail_rids = torch.tensor(sorted(tail), dtype=torch.long, device=device) if tail else torch.empty(0, dtype=torch.long, device=device)
        fallback_easy_mask = easy_mask & ((rid_t[..., None] == tail_rids).any(dim=-1) if tail_rids.numel() else torch.zeros_like(easy_mask))
        active_mask = hard_mask | semi_mask | fallback_easy_mask
        base = torch.where(hard_mask, torch.ones_like(hinge_t), torch.where(semi_mask, torch.full_like(hinge_t, 0.5), torch.full_like(hinge_t, thresholds.fallback_easy_pair_weight)))
        tail_factor = torch.where(
            (rid_t[..., None] == tail_rids).any(dim=-1) if tail_rids.numel() else torch.zeros_like(active_mask),
            torch.full_like(hinge_t, self.tail_multiplier),
            torch.ones_like(hinge_t),
        )
        weights = base * (0.5 + contra_t) * tail_factor
        weights = weights * active_mask.float()

        neg_logits = torch.zeros(hinge_t.numel(), dtype=torch.float32, device=device)
        neg_embeddings = torch.zeros(hinge_t.numel(), embed_dim, dtype=reason_embeddings_current.dtype, device=device)
        if has_mem and mem_logits is not None and mem_reason_emb is not None:
            mem_rows = torch.where(is_mem)[0]
            if mem_rows.numel():
                mi = mem_t[mem_rows].long()
                rr = rid_t[mem_rows].long()
                neg_logits[mem_rows] = mem_logits[mi, rr].detach()
                neg_embeddings[mem_rows] = mem_reason_emb[mi, rr].detach()
        batch_rows = torch.where(~is_mem)[0]
        if batch_rows.numel():
            ni = neg_t[batch_rows].long()
            rr = rid_t[batch_rows].long()
            neg_logits[batch_rows] = reason_logits_current[ni, rr].detach()
            neg_embeddings[batch_rows] = reason_embeddings_current[ni, rr].detach()

        per_count = [0 for _ in range(reason_dim)]
        per_active = [0 for _ in range(reason_dim)]
        per_hard = [0 for _ in range(reason_dim)]
        per_semi = [0 for _ in range(reason_dim)]
        per_easy = [0 for _ in range(reason_dim)]
        margins: list[list[float]] = [[] for _ in range(reason_dim)]
        active_margins: list[list[float]] = [[] for _ in range(reason_dim)]
        for i, r in enumerate(rid_t.tolist()):
            per_count[r] += 1
            per_hard[r] += int(bool(hard_mask[i]))
            per_semi[r] += int(bool(semi_mask[i]))
            per_easy[r] += int(bool(easy_mask[i]))
            margins[r].append(float(hinge_t[i].detach().cpu()))
            if bool(active_mask[i]):
                per_active[r] += 1
                active_margins[r].append(float(hinge_t[i].detach().cpu()))

        def mean_or_zero(vals: list[float]) -> float:
            return float(sum(vals) / len(vals)) if vals else 0.0

        return {
            "pair_pos_indices": pos_t,
            "pair_neg_indices": neg_t,
            "pair_neg_memory_indices": mem_t,
            "pair_reason_ids": rid_t,
            "pair_weights": weights,
            "pair_neg_logits_detached": neg_logits,
            "pair_neg_embedding_detached": neg_embeddings,
            "pair_neg_is_memory": is_mem,
            "pair_action_sim": action_t,
            "pair_visual_sim": visual_t,
            "pair_predicate_sim": pred_t,
            "pair_contradiction": contra_t,
            "pair_hinge_raw": hinge_t,
            "pair_hard_mask": hard_mask,
            "pair_semi_hard_mask": semi_mask,
            "pair_easy_mask": easy_mask,
            "pair_active_mask": active_mask,
            "positive_pairs": torch.stack([pos_t, pos_t], dim=1),
            "contrast_pairs": torch.stack([pos_t, neg_t.clamp_min(0)], dim=1),
            "pair_count": int(hinge_t.numel()),
            "active_pair_count": int(active_mask.sum().item()),
            "hard_pair_count": int(hard_mask.sum().item()),
            "semi_hard_pair_count": int(semi_mask.sum().item()),
            "easy_pair_count": int(easy_mask.sum().item()),
            "zero_loss_pair_count": int((hinge_t <= 0.0).sum().item()),
            "margin_satisfied_count": int((hinge_t <= 0.0).sum().item()),
            "tail_pair_count": int(sum(1 for r in rid_t.tolist() if r in tail)),
            "tail_active_pair_count": int(sum(1 for i, r in enumerate(rid_t.tolist()) if r in tail and bool(active_mask[i]))),
            "pair_memory_count": int(is_mem.sum().item()),
            "pair_no_candidate_count": no_candidate_count,
            "pair_gate_filtered_count": gate_filtered_count,
            "pair_count_per_reason": per_count,
            "active_pair_count_per_reason": per_active,
            "hard_pair_count_per_reason": per_hard,
            "semi_hard_pair_count_per_reason": per_semi,
            "easy_pair_count_per_reason": per_easy,
            "margin_mean_per_reason": [mean_or_zero(x) for x in margins],
            "active_margin_mean_per_reason": [mean_or_zero(x) for x in active_margins],
            "file_names": files,
        }
