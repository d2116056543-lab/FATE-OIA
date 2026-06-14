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
    def __init__(self, dim: int = 384, memory_size: int = 8192, tail_multiplier: float = 2.0) -> None:
        super().__init__()
        self.memory_size = int(memory_size)
        self.tail_multiplier = float(tail_multiplier)
        self.proj = nn.Linear(dim, dim)
        self._memory: dict[str, torch.Tensor | list[str]] = {}

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

        payload = {
            "file_names": list(file_names),
            "global_embed": F.normalize(global_embed.detach().cpu(), dim=-1),
            "predicate_probs": predicate_probs.detach().cpu(),
            "action_targets": action_targets.detach().cpu(),
            "reason_targets": reason_targets.detach().cpu(),
            "contradiction_scores": contradiction_scores.detach().cpu(),
            "reason_logits_detached": reason_logits_detached.detach().cpu(),
            "reason_embeddings_detached": F.normalize(reason_embeddings_detached.detach().cpu(), dim=-1),
        }
        if not self._memory:
            self._memory = payload
            return
        keep = max(0, int(self.memory_size) - len(file_names))
        merged: dict[str, torch.Tensor | list[str]] = {}
        merged["file_names"] = (self._memory.get("file_names", [])[-keep:] if keep else []) + payload["file_names"]  # type: ignore[operator]
        for key in [
            "global_embed",
            "predicate_probs",
            "action_targets",
            "reason_targets",
            "contradiction_scores",
            "reason_logits_detached",
            "reason_embeddings_detached",
        ]:
            old = self._memory[key]
            old = old[-keep:] if keep else old[:0]  # type: ignore[index]
            merged[key] = torch.cat([old, payload[key]], dim=0)  # type: ignore[arg-type]
        self._memory = merged

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

        mem = self._memory
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
        candidates: list[dict[str, float | int | bool]] = []
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

        for r in range(reason_dim):
            pos_idx = torch.where(reason[:, r] > 0.5)[0]
            if pos_idx.numel() == 0:
                continue
            per_reason_limit = max_tail_pairs_per_reason if r in tail else max_pairs_per_reason
            reason_candidates: list[dict[str, float | int | bool]] = []
            rejected_candidates: list[dict[str, float | int | bool]] = []
            a_min, v_min, p_min, c_min = thresholds_for(r)
            for pi in pos_idx.tolist():
                # In-batch candidates.
                neg_idx = torch.where(reason[:, r] <= 0.5)[0]
                for ni in neg_idx.tolist():
                    if ni == pi or files[ni] == files[pi]:
                        continue
                    a = float(action_sim_batch[pi, ni].detach().cpu())
                    v = float(visual_sim_batch[pi, ni].detach().cpu())
                    p = float(predicate_sim_batch[pi, ni].detach().cpu())
                    c = float(contradiction_scores[ni, r].detach().clamp(0, 1).cpu())
                    z_pos = float(reason_logits_current[pi, r].detach().cpu())
                    z_neg = float(reason_logits_current[ni, r].detach().cpu())
                    hinge = float(margin - z_pos + z_neg)
                    candidate = {"pos": pi, "neg": ni, "mem": -1, "rid": r, "is_memory": False, "a": a, "v": v, "p": p, "c": c, "hinge": hinge}
                    if a < a_min or v < v_min or p < p_min or c < c_min:
                        gate_filtered_count += 1
                        if r in tail:
                            rejected_candidates.append(candidate)
                        continue
                    reason_candidates.append(candidate)
                # Memory candidates.
                if has_mem and mem_reason is not None and mem_logits is not None:
                    mem_neg_idx = torch.where(mem_reason[:, r] <= 0.5)[0]
                    for mi in mem_neg_idx.tolist():
                        if mi < len(mem_files) and mem_files[mi] == files[pi]:
                            continue
                        a = float(action_sim_mem[pi, mi].detach().cpu())  # type: ignore[index]
                        v = float(visual_sim_mem[pi, mi].detach().cpu())  # type: ignore[index]
                        p = float(predicate_sim_mem[pi, mi].detach().cpu())  # type: ignore[index]
                        c = float(mem_contra[mi, r].detach().clamp(0, 1).cpu())  # type: ignore[index]
                        z_pos = float(reason_logits_current[pi, r].detach().cpu())
                        z_neg = float(mem_logits[mi, r].detach().cpu())  # type: ignore[index]
                        hinge = float(margin - z_pos + z_neg)
                        candidate = {"pos": pi, "neg": -1, "mem": mi, "rid": r, "is_memory": True, "a": a, "v": v, "p": p, "c": c, "hinge": hinge}
                        if a < a_min or v < v_min or p < p_min or c < c_min:
                            gate_filtered_count += 1
                            if r in tail:
                                rejected_candidates.append(candidate)
                            continue
                        reason_candidates.append(candidate)
            if not reason_candidates:
                if r in tail and rejected_candidates:
                    reason_candidates = sorted(rejected_candidates, key=lambda x: float(x["hinge"]), reverse=True)[:per_reason_limit]
                else:
                    no_candidate_count += int(pos_idx.numel())
                    continue
            hard = [x for x in reason_candidates if float(x["hinge"]) > 0.0]
            semi = [x for x in reason_candidates if -thresholds.semi_hard_band <= float(x["hinge"]) <= 0.0]
            easy = [x for x in reason_candidates if float(x["hinge"]) < -thresholds.semi_hard_band]
            selected = sorted(hard, key=lambda x: float(x["hinge"]), reverse=True)
            selected += sorted(semi, key=lambda x: float(x["hinge"]), reverse=True)
            if not selected and r in tail and easy:
                selected = sorted(easy, key=lambda x: float(x["hinge"]), reverse=True)[:per_reason_limit]
            else:
                selected = selected[:per_reason_limit]
            candidates.extend(selected)
            if len(candidates) >= max_pairs:
                break

        if not candidates:
            empty = self._empty(device, reason_dim, embed_dim)
            empty["pair_no_candidate_count"] = no_candidate_count
            empty["pair_gate_filtered_count"] = gate_filtered_count
            empty["file_names"] = files
            return empty

        candidates = sorted(candidates, key=lambda x: (float(x["hinge"]) > 0.0, float(x["hinge"])), reverse=True)[:max_pairs]
        pos_t = torch.tensor([int(c["pos"]) for c in candidates], dtype=torch.long, device=device)
        neg_t = torch.tensor([int(c["neg"]) for c in candidates], dtype=torch.long, device=device)
        mem_t = torch.tensor([int(c["mem"]) for c in candidates], dtype=torch.long, device=device)
        rid_t = torch.tensor([int(c["rid"]) for c in candidates], dtype=torch.long, device=device)
        is_mem = torch.tensor([bool(c["is_memory"]) for c in candidates], dtype=torch.bool, device=device)
        action_t = torch.tensor([float(c["a"]) for c in candidates], dtype=torch.float32, device=device)
        visual_t = torch.tensor([float(c["v"]) for c in candidates], dtype=torch.float32, device=device)
        pred_t = torch.tensor([float(c["p"]) for c in candidates], dtype=torch.float32, device=device)
        contra_t = torch.tensor([float(c["c"]) for c in candidates], dtype=torch.float32, device=device)
        hinge_t = torch.tensor([float(c["hinge"]) for c in candidates], dtype=torch.float32, device=device)
        hard_mask = hinge_t > 0.0
        semi_mask = (hinge_t >= -thresholds.semi_hard_band) & (hinge_t <= 0.0)
        easy_mask = hinge_t < -thresholds.semi_hard_band
        fallback_easy_mask = easy_mask & torch.tensor([int(c["rid"]) in tail for c in candidates], dtype=torch.bool, device=device)
        active_mask = hard_mask | semi_mask | fallback_easy_mask
        base = torch.where(hard_mask, torch.ones_like(hinge_t), torch.where(semi_mask, torch.full_like(hinge_t, 0.5), torch.full_like(hinge_t, thresholds.fallback_easy_pair_weight)))
        tail_factor = torch.tensor([self.tail_multiplier if int(c["rid"]) in tail else 1.0 for c in candidates], dtype=torch.float32, device=device)
        weights = base * (0.5 + contra_t) * tail_factor
        weights = weights * active_mask.float()

        neg_logits = torch.zeros(len(candidates), dtype=torch.float32, device=device)
        neg_embeddings = torch.zeros(len(candidates), embed_dim, dtype=reason_embeddings_current.dtype, device=device)
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
            "pair_count": int(len(candidates)),
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
