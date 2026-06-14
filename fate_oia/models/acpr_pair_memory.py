from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ACPRPairMemory(nn.Module):
    def __init__(self, dim: int = 384, memory_size: int = 8192, tail_multiplier: float = 2.0) -> None:
        super().__init__()
        self.memory_size = memory_size
        self.tail_multiplier = tail_multiplier
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
    ) -> None:
        payload = {
            "file_names": list(file_names),
            "global_embed": F.normalize(global_embed.detach().cpu(), dim=-1),
            "predicate_probs": predicate_probs.detach().cpu(),
            "action_targets": action_targets.detach().cpu(),
            "reason_targets": reason_targets.detach().cpu(),
        }
        if not self._memory:
            self._memory = payload
            return
        keep = max(0, int(self.memory_size) - len(file_names))
        merged: dict[str, torch.Tensor | list[str]] = {}
        merged["file_names"] = (self._memory.get("file_names", [])[-keep:] if keep else []) + payload["file_names"]  # type: ignore[operator]
        for key in ["global_embed", "predicate_probs", "action_targets", "reason_targets"]:
            old = self._memory[key]
            old = old[-keep:] if keep else old[:0]  # type: ignore[index]
            merged[key] = torch.cat([old, payload[key]], dim=0)  # type: ignore[arg-type]
        self._memory = merged

    def mine(
        self,
        batch_file_names: list[str],
        batch_global_embed: torch.Tensor,
        batch_predicate_probs: torch.Tensor,
        batch_action_targets: torch.Tensor,
        batch_reason_targets: torch.Tensor,
        contradiction_score: torch.Tensor,
        tail_indices: list[int] | None = None,
        max_pairs: int | None = None,
    ) -> dict[str, torch.Tensor | int | list[str]]:
        if not self._memory:
            return self.mine_pairs(
                batch_global_embed,
                batch_action_targets,
                batch_reason_targets,
                tail_indices,
                global_embedding=batch_global_embed,
                predicate_probs=batch_predicate_probs,
                contradiction_scores=contradiction_score,
                file_names=batch_file_names,
                max_pairs=max_pairs,
            )
        device = batch_reason_targets.device
        mem_embed = self._memory["global_embed"].to(device)  # type: ignore[union-attr]
        mem_pred = self._memory["predicate_probs"].to(device)  # type: ignore[union-attr]
        mem_action = self._memory["action_targets"].to(device)  # type: ignore[union-attr]
        mem_reason = self._memory["reason_targets"].to(device)  # type: ignore[union-attr]
        tail = set(int(x) for x in (tail_indices or []))
        max_pairs = int(max_pairs or min(self.memory_size, 4096))
        action_sim = F.cosine_similarity(batch_action_targets[:, None, :], mem_action[None, :, :], dim=-1).clamp(-1, 1)
        visual_sim = F.normalize(batch_global_embed, dim=-1) @ F.normalize(mem_embed, dim=-1).t()
        predicate_sim = F.normalize(batch_predicate_probs, dim=-1) @ F.normalize(mem_pred, dim=-1).t()
        pos_out: list[int] = []
        neg_out: list[int] = []
        rid_out: list[int] = []
        wt_out: list[float] = []
        hard_count = 0
        for r in range(batch_reason_targets.shape[1]):
            pos_idx = torch.where(batch_reason_targets[:, r] > 0.5)[0]
            neg_idx = torch.where(mem_reason[:, r] <= 0.5)[0]
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue
            for pi in pos_idx.tolist():
                score = 0.40 * action_sim[pi, neg_idx] + 0.30 * visual_sim[pi, neg_idx] + 0.20 * predicate_sim[pi, neg_idx]
                c = contradiction_score[pi, r].detach().clamp(0, 1)
                top = neg_idx[torch.argsort(score + 0.30 * c, descending=True)[:2]]
                for mi in top.tolist():
                    pos_out.append(pi)
                    neg_out.append(mi)
                    rid_out.append(r)
                    if float(c.cpu()) > 0.5:
                        hard_count += 1
                    wt_out.append((2.0 if r in tail else 1.0) * (0.5 + float(c.cpu())))
                    if len(pos_out) >= max_pairs:
                        break
                if len(pos_out) >= max_pairs:
                    break
            if len(pos_out) >= max_pairs:
                break
        if not pos_out:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return {"pair_pos_indices": empty, "pair_neg_indices": empty, "pair_neg_memory_indices": empty, "pair_reason_ids": empty, "pair_weights": torch.empty(0, device=device), "positive_pairs": torch.empty(0, 2, dtype=torch.long, device=device), "contrast_pairs": torch.empty(0, 2, dtype=torch.long, device=device), "pair_count": 0, "hard_negative_count": 0, "tail_pair_count": 0}
        pos_t = torch.tensor(pos_out, dtype=torch.long, device=device)
        neg_t = torch.tensor(neg_out, dtype=torch.long, device=device)
        rid_t = torch.tensor(rid_out, dtype=torch.long, device=device)
        pair_action = action_sim[pos_t, neg_t].detach()
        pair_visual = visual_sim[pos_t, neg_t].detach()
        pair_pred = predicate_sim[pos_t, neg_t].detach()
        return {
            "pair_pos_indices": pos_t,
            "pair_neg_indices": neg_t,
            "pair_neg_memory_indices": neg_t,
            "pair_reason_ids": rid_t,
            "pair_weights": torch.tensor(wt_out, dtype=torch.float32, device=device),
            "positive_pairs": torch.stack([pos_t, pos_t], dim=1),
            "contrast_pairs": torch.stack([pos_t, neg_t], dim=1),
            "pair_action_sim": pair_action,
            "pair_visual_sim": pair_visual,
            "pair_predicate_sim": pair_pred,
            "pair_contradiction": contradiction_score[pos_t, rid_t].detach(),
            "pair_count": len(pos_out),
            "hard_negative_count": hard_count,
            "tail_pair_count": int(sum(1 for r in rid_out if r in tail)),
            "file_names": batch_file_names,
        }

    def mine_pairs(
        self,
        embeddings: torch.Tensor,
        action: torch.Tensor,
        reason: torch.Tensor,
        tail_indices: list[int] | None = None,
        global_embedding: torch.Tensor | None = None,
        predicate_probs: torch.Tensor | None = None,
        contradiction_scores: torch.Tensor | None = None,
        file_names: list[str] | None = None,
        max_pairs: int | None = None,
    ) -> dict[str, torch.Tensor | int | list[str]]:
        b = int(reason.shape[0])
        device = reason.device
        max_pairs = int(max_pairs or min(self.memory_size, 4096))
        if b < 2:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return {"pair_pos_indices": empty, "pair_neg_indices": empty, "pair_reason_ids": empty, "pair_weights": torch.empty(0, device=device), "positive_pairs": torch.empty(0, 2, dtype=torch.long, device=device), "contrast_pairs": torch.empty(0, 2, dtype=torch.long, device=device), "pair_count": 0, "hard_negative_count": 0, "tail_pair_count": 0}
        emb = F.normalize(global_embedding if global_embedding is not None else embeddings, dim=-1)
        pred = F.normalize(predicate_probs if predicate_probs is not None else torch.zeros(b, 1, device=device), dim=-1)
        action_sim = F.cosine_similarity(action[:, None, :], action[None, :, :], dim=-1).clamp(-1, 1)
        visual_sim = emb @ emb.t()
        predicate_sim = pred @ pred.t()
        tail = set(int(x) for x in (tail_indices or []))
        pos_out: list[int] = []
        neg_out: list[int] = []
        rid_out: list[int] = []
        wt_out: list[float] = []
        hard_count = 0
        for r in range(reason.shape[1]):
            pos_idx = torch.where(reason[:, r] > 0.5)[0]
            neg_idx = torch.where(reason[:, r] <= 0.5)[0]
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue
            for pi in pos_idx.tolist():
                score = 0.40 * action_sim[pi, neg_idx] + 0.30 * visual_sim[pi, neg_idx] + 0.20 * predicate_sim[pi, neg_idx]
                if contradiction_scores is not None:
                    score = score + 0.30 * contradiction_scores[neg_idx, r]
                top = neg_idx[torch.argsort(score, descending=True)[:2]]
                for ni in top.tolist():
                    pos_out.append(pi); neg_out.append(ni); rid_out.append(r)
                    c = float(contradiction_scores[ni, r].detach().cpu()) if contradiction_scores is not None else 0.5
                    if c > 0.5:
                        hard_count += 1
                    wt_out.append((2.0 if r in tail else 1.0) * (0.5 + c))
                    if len(pos_out) >= max_pairs:
                        break
                if len(pos_out) >= max_pairs:
                    break
            if len(pos_out) >= max_pairs:
                break
        if not pos_out:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return {"pair_pos_indices": empty, "pair_neg_indices": empty, "pair_reason_ids": empty, "pair_weights": torch.empty(0, device=device), "positive_pairs": torch.empty(0, 2, dtype=torch.long, device=device), "contrast_pairs": torch.empty(0, 2, dtype=torch.long, device=device), "pair_count": 0, "hard_negative_count": 0, "tail_pair_count": 0}
        pair_reason_ids = torch.tensor(rid_out, dtype=torch.long, device=device)
        pos_t = torch.tensor(pos_out, dtype=torch.long, device=device)
        neg_t = torch.tensor(neg_out, dtype=torch.long, device=device)
        legacy_pairs = torch.stack([pos_t, neg_t], dim=1)
        return {
            "pair_pos_indices": pos_t,
            "pair_neg_indices": neg_t,
            "pair_reason_ids": pair_reason_ids,
            "pair_weights": torch.tensor(wt_out, dtype=torch.float32, device=device),
            "positive_pairs": legacy_pairs,
            "contrast_pairs": legacy_pairs,
            "pair_action_sim": action_sim[pos_t, neg_t].detach(),
            "pair_visual_sim": visual_sim[pos_t, neg_t].detach(),
            "pair_predicate_sim": predicate_sim[pos_t, neg_t].detach(),
            "pair_contradiction": contradiction_scores[neg_t, pair_reason_ids].detach() if contradiction_scores is not None else torch.zeros(len(pos_out), device=device),
            "pair_count": len(pos_out),
            "hard_negative_count": hard_count,
            "tail_pair_count": int(sum(1 for r in rid_out if r in tail)),
            "file_names": file_names or [],
        }
